"""问题二第二检测点候选区域与最坏定位直径搜索程序。

主体口径：构造Omega1和变半径保证接收候选域，直接枚举S2、G和第二次
测角误差，以第一问定义的定位区域直径执行max/min选点。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from q2_geometry import (
    DEGENERATE,
    EMPTY,
    NUMERIC_UNCERTAIN,
    UNBOUNDED,
    VALID_POLYGON,
    Observation,
    solve_localization,
)


SCRIPT_DIR = Path(__file__).resolve().parent


Point = tuple[float, float]


@dataclass(frozen=True)
class ModelConfig:
    s1_x: float = 0.0
    s1_y: float = 0.0
    theta1_deg: float = 0.0
    angle_error_deg: float = 1.0
    target_center_x: float = 0.0
    target_center_y: float = 0.0
    target_radius: float = 1800.0
    receive_radius_min: float = 1000.0
    receive_radius_max: float = 1500.0
    bearing_failure_radius: float = 5.0

    # 候选域筛选网格。角度方向较窄，优先对角度加密。
    constraint_angle_step_deg: float = 0.025
    refined_constraint_angle_step_deg: float = 0.005
    verification_constraint_angle_step_deg: float = 0.0005
    candidate_step: float = 100.0

    # 最坏定位直径的粗搜情景网格。
    source_angle_step_deg: float = 0.5
    source_radial_step: float = 250.0
    epsilon_step_deg: float = 0.5

    # 中等精度情景网格及多级空间细化。
    medium_source_angle_step_deg: float = 0.25
    medium_source_radial_step: float = 125.0
    medium_epsilon_step_deg: float = 0.25
    refinement_steps: tuple[float, ...] = (20.0, 5.0, 1.0, 0.5, 0.1)
    refinement_radii: tuple[float, ...] = (120.0, 30.0, 6.0, 2.0, 0.5)
    refinement_top_k: tuple[int, ...] = (8, 8, 6, 4, 4)

    # 前列候选点最终复算。
    final_source_angle_step_deg: float = 0.1
    final_source_radial_step: float = 100.0
    final_epsilon_step_deg: float = 0.1
    finalist_count: int = 10

    # 最终推荐点使用更密情景复核。
    verification_source_angle_step_deg: float = 0.05
    verification_source_radial_step: float = 50.0
    verification_epsilon_step_deg: float = 0.05
    verification_candidate_count: int = 4

    feasibility_tolerance: float = 1.0e-8
    near_equivalent_report_tolerance: float = 0.01

    @property
    def s1(self) -> Point:
        return (self.s1_x, self.s1_y)

    @property
    def target_center(self) -> Point:
        return (self.target_center_x, self.target_center_y)


@dataclass
class ScenarioResult:
    worst_diameter: float
    worst_source: Point | None
    worst_epsilon_deg: float | None
    worst_bearing_deg: float | None
    worst_status: str
    normal_scenarios: int
    optical_scenarios: int
    unique_q1_calls: int


@dataclass
class CandidateResult:
    x: float
    y: float
    u: float
    v: float
    movement_distance: float
    safe_margin_checked: float
    worst_diameter: float
    worst_source_x: float | None
    worst_source_y: float | None
    worst_epsilon_deg: float | None
    worst_bearing_deg: float | None
    worst_crossing_angle_deg: float | None
    worst_status: str
    normal_scenarios: int
    optical_scenarios: int
    unique_q1_calls: int
    stage: str

    @property
    def point(self) -> Point:
        return (self.x, self.y)


def _frange(start: float, stop: float, step: float) -> list[float]:
    """生成包含两端点的确定性等距序列。"""

    if step <= 0.0:
        raise ValueError("步长必须为正。")
    if stop < start:
        return []
    count = max(1, int(math.ceil((stop - start) / step)))
    values = [start + (stop - start) * i / count for i in range(count + 1)]
    return values


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _bearing(from_point: Point, to_point: Point) -> float:
    return math.degrees(
        math.atan2(to_point[1] - from_point[1], to_point[0] - from_point[0])
    ) % 360.0


def crossing_angle(first_station: Point, second_station: Point, source: Point) -> float:
    """返回两条观测线在源点处的较小夹角，范围为[0°,180°]。"""

    first = (first_station[0] - source[0], first_station[1] - source[1])
    second = (second_station[0] - source[0], second_station[1] - source[1])
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length == 0.0 or second_length == 0.0:
        return 0.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (
        first_length * second_length
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _point_on_ray(origin: Point, angle_deg: float, radius: float) -> Point:
    angle_rad = math.radians(angle_deg)
    return (
        origin[0] + radius * math.cos(angle_rad),
        origin[1] + radius * math.sin(angle_rad),
    )


def ray_source_interval(config: ModelConfig, angle_deg: float) -> tuple[float, float] | None:
    """求给定方向射线在 Omega1 中允许的径向区间。

    目标圆可能截断射线，故不能简单地对所有方向使用 (5,1500]。
    为了数值采样，把严格下界5 m替换为略大于5 m的下一个浮点数。
    """

    angle_rad = math.radians(angle_deg)
    ux, uy = math.cos(angle_rad), math.sin(angle_rad)
    px = config.s1_x - config.target_center_x
    py = config.s1_y - config.target_center_y

    b = px * ux + py * uy
    c = px * px + py * py - config.target_radius**2
    discriminant = b * b - c
    if discriminant < 0.0:
        return None

    root = math.sqrt(max(0.0, discriminant))
    circle_lower = -b - root
    circle_upper = -b + root
    open_five = math.nextafter(config.bearing_failure_radius, math.inf)
    lower = max(open_five, 0.0, circle_lower)
    upper = min(config.receive_radius_max, circle_upper)
    if upper < lower:
        return None
    return lower, upper


def build_source_samples(
    config: ModelConfig,
    angle_step_deg: float,
    radial_step: float | None,
    critical_only: bool,
) -> list[Point]:
    """在 Omega1 内构造源位置样本。

    critical_only=True 用于候选域圆盘约束：每条射线保留径向端点及1000 m分界点。
    False 用于目标函数情景枚举：按给定径向步长填充整段。
    """

    angles = _frange(
        config.theta1_deg - config.angle_error_deg,
        config.theta1_deg + config.angle_error_deg,
        angle_step_deg,
    )
    points: list[Point] = []
    seen: set[tuple[int, int]] = set()

    for angle in angles:
        interval = ray_source_interval(config, angle)
        if interval is None:
            continue
        lower, upper = interval

        if critical_only:
            radii = [lower, upper]
            if lower <= config.receive_radius_min <= upper:
                radii.append(config.receive_radius_min)
        else:
            if radial_step is None:
                raise ValueError("目标函数源样本必须给出径向步长。")
            radii = _frange(lower, upper, radial_step)
            if lower <= config.receive_radius_min <= upper:
                radii.append(config.receive_radius_min)

        for radius in radii:
            point = _point_on_ray(config.s1, angle, radius)
            key = (round(point[0] * 1.0e6), round(point[1] * 1.0e6))
            if key not in seen:
                points.append(point)
                seen.add(key)

    if not points:
        raise ValueError("第一次观测与目标圆域、1500米接收距离不相容，Omega1为空。")
    return points


def receive_lower_bound(config: ModelConfig, source: Point) -> float:
    return max(config.receive_radius_min, _distance(config.s1, source))


def safe_margin(
    config: ModelConfig, candidate: Point, constraint_sources: Iterable[Point]
) -> float:
    """返回已检查约束中的最小余量；非负表示全部圆盘约束通过。"""

    return min(
        receive_lower_bound(config, source) - _distance(candidate, source)
        for source in constraint_sources
    )


def candidate_bounding_box(
    config: ModelConfig, constraint_sources: Iterable[Point]
) -> tuple[float, float, float, float]:
    """利用全部已检查圆盘的包围盒交集缩小第二点搜索范围。"""

    lower_x = -math.inf
    upper_x = math.inf
    lower_y = -math.inf
    upper_y = math.inf
    for source in constraint_sources:
        radius = receive_lower_bound(config, source)
        lower_x = max(lower_x, source[0] - radius)
        upper_x = min(upper_x, source[0] + radius)
        lower_y = max(lower_y, source[1] - radius)
        upper_y = min(upper_y, source[1] + radius)

    if lower_x > upper_x or lower_y > upper_y:
        raise ValueError("保证接收候选区域为空。")
    return lower_x, upper_x, lower_y, upper_y


def generate_candidate_grid(
    config: ModelConfig,
    constraint_sources: list[Point],
    step: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> list[tuple[Point, float]]:
    """在圆盘交集的包围盒中生成并筛选候选点，不施加1800 m圆域约束。"""

    lower_x, upper_x, lower_y, upper_y = (
        bounds if bounds is not None else candidate_bounding_box(config, constraint_sources)
    )
    xs = _frange(lower_x, upper_x, step)
    ys = _frange(lower_y, upper_y, step)

    candidates: list[tuple[Point, float]] = []
    for x in xs:
        for y in ys:
            point = (x, y)
            if _distance(point, config.s1) <= config.feasibility_tolerance:
                continue
            margin = safe_margin(config, point, constraint_sources)
            if margin >= -config.feasibility_tolerance:
                candidates.append((point, margin))
    return candidates


def _degenerate_diameter(vertices: list[Point]) -> float:
    if len(vertices) <= 1:
        return 0.0
    return max(
        _distance(vertices[i], vertices[j])
        for i in range(len(vertices) - 1)
        for j in range(i + 1, len(vertices))
    )


def evaluate_candidate(
    config: ModelConfig,
    candidate: Point,
    source_samples: list[Point],
    epsilon_step_deg: float,
) -> ScenarioResult:
    """严格按 G 与 epsilon2 两层情景枚举计算候选点的最大定位直径。"""

    epsilon_values = _frange(
        -config.angle_error_deg,
        config.angle_error_deg,
        epsilon_step_deg,
    )
    worst_diameter = -1.0
    worst_source: Point | None = None
    worst_epsilon: float | None = None
    worst_bearing: float | None = None
    worst_status = ""
    normal_scenarios = 0
    optical_scenarios = 0

    # 相同候选点、相同第二读数产生完全相同的Q1输入。缓存只避免重复计算，
    # 外层仍按 G 和 epsilon2 枚举并保留最坏情景见证。
    q1_cache: dict[float, tuple[float, str]] = {}

    for source in source_samples:
        if _distance(candidate, source) <= config.bearing_failure_radius:
            optical_scenarios += 1
            scenario_diameter = 0.0
            if scenario_diameter > worst_diameter:
                worst_diameter = scenario_diameter
                worst_source = source
                worst_epsilon = None
                worst_bearing = None
                worst_status = "OPTICAL_WITHIN_5M"
            continue

        true_bearing = _bearing(candidate, source)
        for epsilon in epsilon_values:
            normal_scenarios += 1
            measured_bearing = (true_bearing + epsilon) % 360.0
            cache_key = round(measured_bearing, 12)
            if cache_key not in q1_cache:
                result = solve_localization(
                    [
                        Observation(
                            config.s1_x, config.s1_y, config.theta1_deg
                        ),
                        Observation(candidate[0], candidate[1], measured_bearing),
                    ],
                    error_deg=config.angle_error_deg,
                    known_feasible_point=source,
                )

                if result.status == VALID_POLYGON:
                    if result.diameter is None:
                        raise RuntimeError("Q1返回正常多边形但没有直径。")
                    value = result.diameter
                elif result.status == DEGENERATE:
                    value = _degenerate_diameter(result.vertices)
                elif result.status in (UNBOUNDED, NUMERIC_UNCERTAIN):
                    value = math.inf
                elif result.status == EMPTY:
                    raise RuntimeError(
                        "合法源与允许误差生成的两次角域交集为空，请检查模型或数值实现。"
                    )
                else:
                    raise RuntimeError(f"Q1返回未处理状态：{result.status}")
                q1_cache[cache_key] = (value, result.status)

            scenario_diameter, status = q1_cache[cache_key]
            if scenario_diameter > worst_diameter:
                worst_diameter = scenario_diameter
                worst_source = source
                worst_epsilon = epsilon
                worst_bearing = measured_bearing
                worst_status = status

            # 一旦出现无界情景，该候选点的最大值已经确定为无穷。
            if math.isinf(worst_diameter):
                return ScenarioResult(
                    worst_diameter=math.inf,
                    worst_source=worst_source,
                    worst_epsilon_deg=worst_epsilon,
                    worst_bearing_deg=worst_bearing,
                    worst_status=worst_status,
                    normal_scenarios=normal_scenarios,
                    optical_scenarios=optical_scenarios,
                    unique_q1_calls=len(q1_cache),
                )

    return ScenarioResult(
        worst_diameter=max(0.0, worst_diameter),
        worst_source=worst_source,
        worst_epsilon_deg=worst_epsilon,
        worst_bearing_deg=worst_bearing,
        worst_status=worst_status,
        normal_scenarios=normal_scenarios,
        optical_scenarios=optical_scenarios,
        unique_q1_calls=len(q1_cache),
    )


def local_coordinates(config: ModelConfig, point: Point) -> tuple[float, float]:
    dx = point[0] - config.s1_x
    dy = point[1] - config.s1_y
    angle = math.radians(config.theta1_deg)
    e = (math.cos(angle), math.sin(angle))
    n = (-math.sin(angle), math.cos(angle))
    return dx * e[0] + dy * e[1], dx * n[0] + dy * n[1]


def evaluate_grid(
    config: ModelConfig,
    candidates: list[tuple[Point, float]],
    source_samples: list[Point],
    epsilon_step_deg: float,
    stage: str,
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    total = len(candidates)
    for index, (candidate, margin) in enumerate(candidates, start=1):
        scenario = evaluate_candidate(
            config, candidate, source_samples, epsilon_step_deg
        )
        u, v = local_coordinates(config, candidate)
        source = scenario.worst_source
        results.append(
            CandidateResult(
                x=candidate[0],
                y=candidate[1],
                u=u,
                v=v,
                movement_distance=_distance(config.s1, candidate),
                safe_margin_checked=margin,
                worst_diameter=scenario.worst_diameter,
                worst_source_x=None if source is None else source[0],
                worst_source_y=None if source is None else source[1],
                worst_epsilon_deg=scenario.worst_epsilon_deg,
                worst_bearing_deg=scenario.worst_bearing_deg,
                worst_crossing_angle_deg=(
                    None
                    if source is None or scenario.worst_epsilon_deg is None
                    else crossing_angle(config.s1, candidate, source)
                ),
                worst_status=scenario.worst_status,
                normal_scenarios=scenario.normal_scenarios,
                optical_scenarios=scenario.optical_scenarios,
                unique_q1_calls=scenario.unique_q1_calls,
                stage=stage,
            )
        )
        if index == total or index % max(1, total // 10) == 0:
            print(f"[{stage}] 已评价 {index}/{total} 个候选点。", flush=True)
    return results


def _candidate_sort_key(result: CandidateResult) -> tuple[float, float]:
    return result.worst_diameter, result.movement_distance


def build_refinement_candidates(
    config: ModelConfig,
    seed_results: list[CandidateResult],
    constraint_sources: list[Point],
    radius: float,
    step: float,
    top_k: int,
) -> list[tuple[Point, float]]:
    """围绕当前前列点生成更细空间网格，并立即检查接收约束。"""

    finite = [item for item in seed_results if math.isfinite(item.worst_diameter)]
    if not finite:
        return []
    seeds = _unique_ranked(finite)[:top_k]

    points: dict[tuple[int, int], tuple[Point, float]] = {}
    offsets = _frange(-radius, radius, step)
    for seed in seeds:
        for dx in offsets:
            for dy in offsets:
                candidate = (seed.x + dx, seed.y + dy)
                if _distance(candidate, config.s1) <= config.feasibility_tolerance:
                    continue
                margin = safe_margin(config, candidate, constraint_sources)
                if margin >= -config.feasibility_tolerance:
                    key = (round(candidate[0] * 1.0e6), round(candidate[1] * 1.0e6))
                    points[key] = (candidate, margin)
    return list(points.values())


def _unique_ranked(results: list[CandidateResult]) -> list[CandidateResult]:
    """按目标值和移动距离排序，并删除重复坐标。"""

    unique: dict[tuple[int, int], CandidateResult] = {}
    for item in sorted(results, key=_candidate_sort_key):
        key = (round(item.x * 1.0e6), round(item.y * 1.0e6))
        unique.setdefault(key, item)
    return list(unique.values())


def select_best(
    results: list[CandidateResult], near_tolerance: float
) -> tuple[CandidateResult, list[CandidateResult]]:
    """严格按最坏直径取最小值，同时列出当前精度下的近似等价点。

    移动距离只用于打破浮点计算意义下的真正并列，不允许用“近似并列”
    替换主目标的最小值。near_tolerance仅用于结果报告，不参与推荐点选择。
    """

    finite = [item for item in results if math.isfinite(item.worst_diameter)]
    if not finite:
        raise RuntimeError("没有有限定位直径的候选点。")
    minimum_diameter = min(item.worst_diameter for item in finite)
    exact_ties = [
        item
        for item in finite
        if math.isclose(
            item.worst_diameter,
            minimum_diameter,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
    ]
    best = min(exact_ties, key=lambda item: item.movement_distance)
    near_equivalent = [
        item
        for item in finite
        if item.worst_diameter <= minimum_diameter + near_tolerance
    ]
    return best, sorted(near_equivalent, key=_candidate_sort_key)


def verify_best_candidate(
    config: ModelConfig,
    candidate: Point,
    final_constraint_sources: list[Point],
) -> float:
    margin = safe_margin(config, candidate, final_constraint_sources)
    if margin < -config.feasibility_tolerance:
        raise RuntimeError(
            f"推荐点未通过加密候选域复核，最小接收余量为 {margin:.6f} m。"
        )
    return margin


def _json_number(value: float | None) -> float | str | None:
    if value is None:
        return None
    if math.isinf(value):
        return "Infinity"
    if math.isnan(value):
        return "NaN"
    return value


def candidate_to_dict(result: CandidateResult) -> dict:
    data = asdict(result)
    data["worst_diameter"] = _json_number(result.worst_diameter)
    return data


def write_candidate_csv(path: Path, results: list[CandidateResult]) -> None:
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=_candidate_sort_key):
            row = asdict(result)
            if math.isinf(result.worst_diameter):
                row["worst_diameter"] = "Infinity"
            writer.writerow(row)


def run_model(config: ModelConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (
        len(config.refinement_steps)
        == len(config.refinement_radii)
        == len(config.refinement_top_k)
    ):
        raise ValueError("空间细化步长、半径和种子数的长度必须一致。")

    print("1/7 构造第一次源可行域与候选域约束样本。", flush=True)
    constraint_sources = build_source_samples(
        config,
        config.constraint_angle_step_deg,
        radial_step=None,
        critical_only=True,
    )
    source_samples = build_source_samples(
        config,
        config.source_angle_step_deg,
        config.source_radial_step,
        critical_only=False,
    )

    print("2/7 生成保证接收的第二点粗网格。", flush=True)
    candidates = generate_candidate_grid(
        config, constraint_sources, config.candidate_step
    )
    if not candidates:
        raise RuntimeError("当前候选步长下没有找到保证接收的第二点，请减小candidate_step。")

    print("3/7 对粗网格逐点枚举 G 与 epsilon2。", flush=True)
    coarse_results = evaluate_grid(
        config,
        candidates,
        source_samples,
        config.epsilon_step_deg,
        stage="coarse",
    )

    print("4/7 依次执行20、5、1、0.5、0.1米空间细化。", flush=True)
    refined_constraint_sources = build_source_samples(
        config,
        config.refined_constraint_angle_step_deg,
        radial_step=None,
        critical_only=True,
    )
    medium_source_samples = build_source_samples(
        config,
        config.medium_source_angle_step_deg,
        config.medium_source_radial_step,
        critical_only=False,
    )

    refinement_results: list[CandidateResult] = []
    refinement_summary: list[dict] = []
    current_results = coarse_results
    for level, (step, radius, top_k) in enumerate(
        zip(
            config.refinement_steps,
            config.refinement_radii,
            config.refinement_top_k,
            strict=True,
        ),
        start=1,
    ):
        refine_candidates = build_refinement_candidates(
            config,
            current_results,
            refined_constraint_sources,
            radius=radius,
            step=step,
            top_k=top_k,
        )
        if not refine_candidates:
            raise RuntimeError(f"{step:g}米空间细化没有产生可行候选点。")

        # 第一级只负责从大范围进入局部，继续使用粗情景；其余级别使用中等精度情景。
        if level == 1:
            level_sources = source_samples
            level_epsilon_step = config.epsilon_step_deg
        else:
            level_sources = medium_source_samples
            level_epsilon_step = config.medium_epsilon_step_deg

        stage = f"refine_{step:g}m"
        current_results = evaluate_grid(
            config,
            refine_candidates,
            level_sources,
            level_epsilon_step,
            stage=stage,
        )
        refinement_results.extend(current_results)
        level_best, _ = select_best(
            current_results, config.near_equivalent_report_tolerance
        )
        refinement_summary.append(
            {
                "stage": stage,
                "step_m": step,
                "candidate_count": len(current_results),
                "best_x": level_best.x,
                "best_y": level_best.y,
                "best_worst_diameter_m": level_best.worst_diameter,
                "safe_margin_checked_m": level_best.safe_margin_checked,
            }
        )

    all_search_results = coarse_results + refinement_results

    # 只能比较在同一情景网格上得到的目标值。不同细化阶段使用的G和epsilon2
    # 网格不同，若混排会让较粗网格的低估值挤占最终复算名额。
    finalists = _unique_ranked(current_results)[: config.finalist_count]
    if not finalists:
        raise RuntimeError("全部已检查候选点的定位区域均无界。")

    print("5/7 对前列候选点使用最终情景网格统一复算。", flush=True)
    final_source_samples = build_source_samples(
        config,
        config.final_source_angle_step_deg,
        config.final_source_radial_step,
        critical_only=False,
    )
    finalist_candidates = [
        (
            item.point,
            verify_best_candidate(config, item.point, refined_constraint_sources),
        )
        for item in finalists
    ]
    finalist_results = evaluate_grid(
        config,
        finalist_candidates,
        final_source_samples,
        config.final_epsilon_step_deg,
        stage="final_check",
    )
    finite_finalists = [
        item for item in finalist_results if math.isfinite(item.worst_diameter)
    ]
    if not finite_finalists:
        raise RuntimeError("全部前列候选点在加密复算中均出现无界定位区域。")

    print("6/7 使用更密候选域和情景网格复核前列点。", flush=True)
    verification_constraint_sources = build_source_samples(
        config,
        config.verification_constraint_angle_step_deg,
        radial_step=None,
        critical_only=True,
    )
    verification_source_samples = build_source_samples(
        config,
        config.verification_source_angle_step_deg,
        config.verification_source_radial_step,
        critical_only=False,
    )
    verification_seeds = _unique_ranked(finite_finalists)[
        : config.verification_candidate_count
    ]
    verification_candidates = [
        (
            item.point,
            verify_best_candidate(
                config, item.point, verification_constraint_sources
            ),
        )
        for item in verification_seeds
    ]
    verification_results = evaluate_grid(
        config,
        verification_candidates,
        verification_source_samples,
        config.verification_epsilon_step_deg,
        stage="verification",
    )
    best, near_equivalent = select_best(
        verification_results, config.near_equivalent_report_tolerance
    )

    final_grid_best, _ = select_best(
        finite_finalists, config.near_equivalent_report_tolerance
    )
    final_to_verification_shift = _distance(final_grid_best.point, best.point)
    final_to_verification_change = best.worst_diameter - final_grid_best.worst_diameter

    print("7/7 写出结构化结果。", flush=True)
    csv_path = output_dir / "q2_candidates.csv"
    json_path = output_dir / "q2_result.json"
    output_results = (
        all_search_results + finalist_results + verification_results
    )
    write_candidate_csv(csv_path, output_results)

    result = {
        "model_statement": {
            "objective": (
                "理论上对每个S2在全部G与epsilon2情景中取最大定位直径，"
                "再在候选区域内取最小值；程序用逐级加密的确定性网格逼近"
            ),
            "selection_rule": "推荐点严格取已复核候选点中的最小值；移动距离只处理浮点意义下的真正并列",
            "candidate_region": "对Omega1中全部G取圆盘B(G,max(1000,|G-S1|))的交集，并删除S1",
            "second_point_limited_to_target_circle": False,
            "near_distance_rule": "距离不超过5米时光学精确定位；大于5米时正常测向，包括5至20米",
            "minimum_enclosing_circle_used": False,
            "possible_reading_set_used": False,
        },
        "config": asdict(config),
        "counts": {
            "constraint_source_samples": len(constraint_sources),
            "refined_constraint_source_samples": len(refined_constraint_sources),
            "verification_constraint_source_samples": len(
                verification_constraint_sources
            ),
            "objective_source_samples_coarse": len(source_samples),
            "objective_source_samples_medium": len(medium_source_samples),
            "objective_source_samples_final": len(final_source_samples),
            "objective_source_samples_verification": len(
                verification_source_samples
            ),
            "coarse_candidates": len(coarse_results),
            "refined_candidates": len(refinement_results),
            "finalists_rechecked": len(finalist_results),
            "verification_candidates": len(verification_results),
        },
        "candidate_bounding_box_checked": candidate_bounding_box(
            config, verification_constraint_sources
        ),
        "spatial_refinement": refinement_summary,
        "recommended_point": candidate_to_dict(best),
        "near_equivalent_finalists": [
            candidate_to_dict(item) for item in near_equivalent
        ],
        "near_equivalent_report_tolerance_m": (
            config.near_equivalent_report_tolerance
        ),
        "final_verification_comparison": {
            "point_shift_m": final_to_verification_shift,
            "worst_diameter_change_m": final_to_verification_change,
        },
        "evidence_boundary": (
            "该结果是当前确定性候选点、源位置和测角误差网格上的推荐方案。"
            "程序已完成多级空间细化、前列点同网格比较和独立加密复核，"
            "但不把有限网格结果称为连续空间严格全局最优。"
        ),
        "outputs": {
            "candidate_csv": csv_path.name,
        },
    }
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def run_self_tests() -> None:
    """最小模型—代码一致性测试。"""

    # 与第一问已验证算例保持一致：三个角域交成边长20 m的等边三角形。
    root_three = math.sqrt(3.0)
    triangle = solve_localization(
        [
            Observation(-610.0, -10.0 / root_three, 1.0),
            Observation(
                310.0, -10.0 / root_three - 300.0 * root_three, 121.0
            ),
            Observation(
                300.0, 20.0 / root_three + 300.0 * root_three, 241.0
            ),
        ]
    )
    assert triangle.status == VALID_POLYGON
    assert triangle.diameter is not None
    assert math.isclose(triangle.diameter, 20.0, abs_tol=1.0e-6)

    assert (
        solve_localization([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]).status
        == UNBOUNDED
    )
    assert (
        solve_localization([(0.0, 0.0, 0.0), (-10.0, 0.0, 180.0)]).status
        == EMPTY
    )
    assert (
        solve_localization([(0.0, 0.0, 0.0), (0.0, 0.0, 180.0)]).status
        == DEGENERATE
    )

    default = ModelConfig()
    interval = ray_source_interval(default, 0.0)
    assert interval is not None
    assert interval[0] > 5.0
    assert math.isclose(interval[1], 1500.0, abs_tol=1.0e-9)

    constraints = build_source_samples(
        default, angle_step_deg=0.1, radial_step=None, critical_only=True
    )
    assert safe_margin(default, (500.0, 300.0), constraints) >= -1.0e-8
    assert safe_margin(default, (-1000.0, 0.0), constraints) < 0.0

    # 第二点不受目标1800 m圆限制：该点在目标圆外，但仍满足此算例的接收保证。
    edge_case = ModelConfig(s1_x=1700.0, s1_y=0.0, theta1_deg=0.0)
    edge_constraints = build_source_samples(
        edge_case, angle_step_deg=0.1, radial_step=None, critical_only=True
    )
    outside_target_candidate = (2500.0, 0.0)
    assert _distance(outside_target_candidate, edge_case.target_center) > 1800.0
    assert safe_margin(
        edge_case, outside_target_candidate, edge_constraints
    ) >= -1.0e-8

    # 5 m分支：无需虚构第二示向度，直接得到零定位直径。
    optical_source = [(104.0, 0.0)]
    optical_result = evaluate_candidate(
        default, (100.0, 0.0), optical_source, epsilon_step_deg=0.5
    )
    assert optical_result.worst_diameter == 0.0
    assert optical_result.optical_scenarios == 1
    assert optical_result.normal_scenarios == 0

    # 5至20 m仍按正常测向，而不是自动记零。
    normal_source = [(115.0, 1.0)]
    normal_result = evaluate_candidate(
        default, (100.0, 0.0), normal_source, epsilon_step_deg=1.0
    )
    assert normal_result.normal_scenarios > 0
    assert normal_result.optical_scenarios == 0

    assert math.isclose(
        crossing_angle((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
        45.0,
        abs_tol=1.0e-12,
    )

    # 典型侧向第二点与合法源产生正常有界交会区域。
    finite_source = [(1000.0, 0.0)]
    finite_result = evaluate_candidate(
        default, (500.0, 300.0), finite_source, epsilon_step_deg=1.0
    )
    assert math.isfinite(finite_result.worst_diameter)
    assert finite_result.worst_diameter > 0.0

    # 近似等价只用于报告，不能让移动更短但直径更大的点替代真正的min。
    common = {
        "x": 0.0,
        "y": 0.0,
        "u": 0.0,
        "v": 0.0,
        "safe_margin_checked": 1.0,
        "worst_source_x": 1.0,
        "worst_source_y": 0.0,
        "worst_epsilon_deg": 0.0,
        "worst_bearing_deg": 0.0,
        "worst_crossing_angle_deg": 90.0,
        "worst_status": VALID_POLYGON,
        "normal_scenarios": 1,
        "optical_scenarios": 0,
        "unique_q1_calls": 1,
        "stage": "test",
    }
    exact_minimum = CandidateResult(
        **common, movement_distance=100.0, worst_diameter=10.0
    )
    shorter_but_worse = CandidateResult(
        **common, movement_distance=1.0, worst_diameter=10.005
    )
    selected, near = select_best(
        [shorter_but_worse, exact_minimum], near_tolerance=0.01
    )
    assert selected is exact_minimum
    assert len(near) == 2

    print("All Q2 self-tests passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="问题二：第二检测点候选区域与最坏定位直径最小化"
    )
    parser.add_argument("--s1-x", type=float, default=0.0)
    parser.add_argument("--s1-y", type=float, default=0.0)
    parser.add_argument("--theta1", type=float, default=0.0, help="第一次示向度/度")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="结果输出目录，默认是Q2脚本目录",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return

    config = ModelConfig(
        s1_x=args.s1_x,
        s1_y=args.s1_y,
        theta1_deg=args.theta1,
    )
    run_model(config, args.output_dir)


if __name__ == "__main__":
    main()
