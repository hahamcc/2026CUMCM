"""问题一：测向角域交会、定位多边形直径与同直径圆盘覆盖判定。

核心模型与论文一致：
1. 每个示向度及其 ±1° 误差形成两个闭半平面；
2. 枚举全部边界线对，保留满足所有半平面约束的交点；
3. 枚举多边形顶点对求区域直径；
4. 以一对最远顶点的中点为唯一候选圆心，检查全部顶点。

本文件只依赖 Python 标准库。直接运行可执行等边三角形示例；使用
``python q1_algorithm.py --self-test`` 可运行内置测试。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


Point = tuple[float, float]


@dataclass(frozen=True)
class Observation:
    """一次同源测向观测。角度以正东为 0°，逆时针为正。"""

    x: float
    y: float
    bearing_deg: float

    @property
    def station(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class HalfPlane:
    """标准形式 a*x + b*y <= c，同时保存边界线的点和方向。"""

    a: float
    b: float
    c: float
    line_point: Point
    line_direction: Point
    observation_index: int
    boundary_name: str


@dataclass(frozen=True)
class NumericTolerance:
    parallel: float = 1.0e-10
    feasible: float = 1.0e-8
    duplicate: float = 1.0e-7
    geometry: float = 1.0e-9
    cover: float = 1.0e-8


@dataclass
class LocalizationResult:
    status: str
    vertices: list[Point]
    diameter: float | None = None
    farthest_pair: tuple[Point, Point] | None = None
    circle_center: Point | None = None
    circle_radius: float | None = None
    max_center_distance: float | None = None
    coverage_margin: float | None = None
    covered: bool | None = None
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


VALID_POLYGON = "VALID_POLYGON"
EMPTY = "EMPTY"
UNBOUNDED = "UNBOUNDED"
DEGENERATE = "DEGENERATE"
INVALID_INPUT = "INVALID_INPUT"


def cross(a: Point, b: Point) -> float:
    """二维向量叉积。"""

    return a[0] * b[1] - a[1] * b[0]


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def subtract(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def distance_squared(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def direction(angle_deg: float) -> Point:
    angle_rad = math.radians(angle_deg)
    return (math.cos(angle_rad), math.sin(angle_rad))


def _as_observation(value: Observation | Sequence[float]) -> Observation:
    if isinstance(value, Observation):
        obs = value
    else:
        if len(value) != 3:
            raise ValueError("每条观测必须为 (x, y, bearing_deg)。")
        obs = Observation(float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(v) for v in (obs.x, obs.y, obs.bearing_deg)):
        raise ValueError("检测点坐标和示向度必须是有限数。")
    return obs


def build_halfplanes(
    observations: Iterable[Observation | Sequence[float]],
    error_deg: float = 1.0,
) -> tuple[list[Observation], list[HalfPlane]]:
    """把每条 ±error_deg 测向约束转换成两个向前闭半平面。"""

    obs_list = [_as_observation(item) for item in observations]
    if not obs_list:
        raise ValueError("至少需要一条观测。")
    if not math.isfinite(error_deg) or not (0.0 < error_deg < 90.0):
        raise ValueError("误差半张角必须在 (0°, 90°) 内。")

    halfplanes: list[HalfPlane] = []
    for index, obs in enumerate(obs_list):
        station = obs.station
        lower = direction(obs.bearing_deg - error_deg)
        upper = direction(obs.bearing_deg + error_deg)

        # lower × (X-S) >= 0
        # 等价于 (lower_y, -lower_x)·X <= (lower_y, -lower_x)·S
        lower_normal = (lower[1], -lower[0])
        halfplanes.append(
            HalfPlane(
                a=lower_normal[0],
                b=lower_normal[1],
                c=dot(lower_normal, station),
                line_point=station,
                line_direction=lower,
                observation_index=index,
                boundary_name="lower",
            )
        )

        # upper × (X-S) <= 0
        # 等价于 (-upper_y, upper_x)·X <= (-upper_y, upper_x)·S
        upper_normal = (-upper[1], upper[0])
        halfplanes.append(
            HalfPlane(
                a=upper_normal[0],
                b=upper_normal[1],
                c=dot(upper_normal, station),
                line_point=station,
                line_direction=upper,
                observation_index=index,
                boundary_name="upper",
            )
        )

    return obs_list, halfplanes


def satisfies_halfplane(point: Point, halfplane: HalfPlane, eps: float) -> bool:
    return (
        halfplane.a * point[0] + halfplane.b * point[1]
        <= halfplane.c + eps
    )


def satisfies_all(point: Point, halfplanes: Sequence[HalfPlane], eps: float) -> bool:
    return all(satisfies_halfplane(point, hp, eps) for hp in halfplanes)


def line_intersection(
    first: HalfPlane,
    second: HalfPlane,
    parallel_eps: float,
) -> Point | None:
    """返回两条支撑线的唯一交点；平行或重合时返回 None。"""

    p = first.line_point
    q = second.line_point
    u = first.line_direction
    v = second.line_direction
    denominator = cross(u, v)
    if abs(denominator) <= parallel_eps:
        return None
    t = cross(subtract(q, p), v) / denominator
    return (p[0] + t * u[0], p[1] + t * u[1])


def _append_if_new(points: list[Point], candidate: Point, eps: float) -> None:
    threshold2 = eps * eps
    if all(distance_squared(candidate, existing) > threshold2 for existing in points):
        points.append(candidate)


def enumerate_feasible_intersections(
    halfplanes: Sequence[HalfPlane],
    tolerance: NumericTolerance,
) -> list[Point]:
    """两两求交，并用全部约束筛选、去重。"""

    vertices: list[Point] = []
    for i in range(len(halfplanes) - 1):
        for j in range(i + 1, len(halfplanes)):
            point = line_intersection(
                halfplanes[i], halfplanes[j], tolerance.parallel
            )
            if point is None:
                continue
            if satisfies_all(point, halfplanes, tolerance.feasible):
                _append_if_new(vertices, point, tolerance.duplicate)
    return vertices


def _add_linear_bound(
    coefficient: float,
    right_side: float,
    lower_x: float,
    upper_x: float,
    eps: float,
) -> tuple[bool, float, float]:
    """把 coefficient*x <= right_side 合并到 x 的可行区间。"""

    if abs(coefficient) <= eps:
        return (right_side >= -eps, lower_x, upper_x)
    bound = right_side / coefficient
    if coefficient > 0.0:
        upper_x = min(upper_x, bound)
    else:
        lower_x = max(lower_x, bound)
    return (lower_x <= upper_x + eps, lower_x, upper_x)


def find_feasible_point(
    halfplanes: Sequence[HalfPlane], eps: float
) -> tuple[bool, Point | None]:
    """用二维 Fourier-Motzkin 消元检查半平面组可行性。

    该函数只用于识别 EMPTY，不参与正常多边形的顶点计算。
    """

    lower_lines: list[tuple[float, float]] = []  # y >= slope*x + intercept
    upper_lines: list[tuple[float, float]] = []  # y <= slope*x + intercept
    lower_x = -math.inf
    upper_x = math.inf

    for hp in halfplanes:
        if hp.b > eps:
            upper_lines.append((-hp.a / hp.b, hp.c / hp.b))
        elif hp.b < -eps:
            lower_lines.append((-hp.a / hp.b, hp.c / hp.b))
        else:
            ok, lower_x, upper_x = _add_linear_bound(
                hp.a, hp.c, lower_x, upper_x, eps
            )
            if not ok:
                return False, None

    for lower_slope, lower_intercept in lower_lines:
        for upper_slope, upper_intercept in upper_lines:
            # lower_slope*x + lower_intercept
            # <= upper_slope*x + upper_intercept
            ok, lower_x, upper_x = _add_linear_bound(
                lower_slope - upper_slope,
                upper_intercept - lower_intercept,
                lower_x,
                upper_x,
                eps,
            )
            if not ok:
                return False, None

    if math.isfinite(lower_x) and math.isfinite(upper_x):
        x = 0.5 * (lower_x + upper_x)
    elif math.isfinite(lower_x):
        x = max(0.0, lower_x)
    elif math.isfinite(upper_x):
        x = min(0.0, upper_x)
    else:
        x = 0.0

    y_lower = max((m * x + q for m, q in lower_lines), default=-math.inf)
    y_upper = min((m * x + q for m, q in upper_lines), default=math.inf)
    if y_lower > y_upper + eps:
        return False, None
    if math.isfinite(y_lower) and math.isfinite(y_upper):
        y = 0.5 * (y_lower + y_upper)
    elif math.isfinite(y_lower):
        y = max(0.0, y_lower)
    elif math.isfinite(y_upper):
        y = min(0.0, y_upper)
    else:
        y = 0.0

    witness = (x, y)
    if not satisfies_all(witness, halfplanes, 10.0 * eps):
        return False, None
    return True, witness


def has_unbounded_direction(
    halfplanes: Sequence[HalfPlane], eps: float
) -> bool:
    """判断衰退锥中是否存在非零方向，即可行域是否可能无界。"""

    if not halfplanes:
        return True
    candidate_directions: list[Point] = []
    for hp in halfplanes:
        # a*d_x + b*d_y = 0 的两个单位方向。
        candidate_directions.append((hp.b, -hp.a))
        candidate_directions.append((-hp.b, hp.a))

    for candidate in candidate_directions:
        if all(hp.a * candidate[0] + hp.b * candidate[1] <= eps for hp in halfplanes):
            return True
    return False


def sort_vertices_counterclockwise(vertices: Sequence[Point]) -> list[Point]:
    center_x = sum(point[0] for point in vertices) / len(vertices)
    center_y = sum(point[1] for point in vertices) / len(vertices)
    return sorted(
        vertices,
        key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x),
    )


def polygon_signed_area(vertices: Sequence[Point]) -> float:
    if len(vertices) < 3:
        return 0.0
    total = 0.0
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        total += cross(first, second)
    return 0.5 * total


def diameter_and_coverage(
    vertices: Sequence[Point], cover_eps: float = 1.0e-8
) -> tuple[float, tuple[Point, Point], Point, float, float, float, bool]:
    """枚举顶点对求直径，并完成同直径圆盘覆盖判定。"""

    if len(vertices) < 2:
        raise ValueError("至少需要两个不同顶点才能计算直径。")

    best_d2 = -1.0
    farthest_pair: tuple[Point, Point] | None = None
    for i in range(len(vertices) - 1):
        for j in range(i + 1, len(vertices)):
            current_d2 = distance_squared(vertices[i], vertices[j])
            if current_d2 > best_d2:
                best_d2 = current_d2
                farthest_pair = (vertices[i], vertices[j])

    assert farthest_pair is not None
    diameter = math.sqrt(best_d2)
    first, second = farthest_pair
    center = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    radius = diameter / 2.0
    max_center_distance = max(
        math.sqrt(distance_squared(vertex, center)) for vertex in vertices
    )
    margin = radius - max_center_distance
    covered = max_center_distance <= radius + cover_eps
    return (
        diameter,
        farthest_pair,
        center,
        radius,
        max_center_distance,
        margin,
        covered,
    )


def solve_localization(
    observations: Iterable[Observation | Sequence[float]],
    error_deg: float = 1.0,
    tolerance: NumericTolerance | None = None,
) -> LocalizationResult:
    """执行第一问的完整算法。"""

    tol = tolerance or NumericTolerance()
    try:
        _, halfplanes = build_halfplanes(observations, error_deg)
    except (TypeError, ValueError) as exc:
        return LocalizationResult(INVALID_INPUT, [], message=str(exc))

    feasible, _ = find_feasible_point(halfplanes, tol.feasible)
    if not feasible:
        return LocalizationResult(EMPTY, [], message="全部测向角域没有公共点。")

    vertices = enumerate_feasible_intersections(halfplanes, tol)
    if has_unbounded_direction(halfplanes, tol.geometry):
        return LocalizationResult(
            UNBOUNDED,
            sort_vertices_counterclockwise(vertices) if vertices else [],
            message="测向角域交集无界，因而没有有限的区域直径。",
        )

    if len(vertices) < 3:
        return LocalizationResult(
            DEGENERATE,
            vertices,
            message="角域交集退化为点或线段，不是正常二维定位多边形。",
        )

    vertices = sort_vertices_counterclockwise(vertices)
    if abs(polygon_signed_area(vertices)) <= tol.geometry:
        return LocalizationResult(
            DEGENERATE,
            vertices,
            message="角域交集面积近似为零，不是正常二维定位多边形。",
        )

    (
        diameter,
        farthest_pair,
        center,
        radius,
        max_center_distance,
        margin,
        covered,
    ) = diameter_and_coverage(vertices, tol.cover)

    return LocalizationResult(
        status=VALID_POLYGON,
        vertices=vertices,
        diameter=diameter,
        farthest_pair=farthest_pair,
        circle_center=center,
        circle_radius=radius,
        max_center_distance=max_center_distance,
        coverage_margin=margin,
        covered=covered,
        message="计算成功。",
    )


def equilateral_example() -> list[Observation]:
    """返回论文中符合 ±1° 测向条件的等边三角形反例。"""

    root3 = math.sqrt(3.0)
    return [
        Observation(-610.0, -10.0 / root3, 1.0),
        Observation(310.0, -10.0 / root3 - 300.0 * root3, 121.0),
        Observation(300.0, 20.0 / root3 + 300.0 * root3, 241.0),
    ]


def _point_in_observation_wedge(
    point: Point,
    observation: Observation,
    error_deg: float = 1.0,
    eps: float = 1.0e-8,
) -> bool:
    _, halfplanes = build_halfplanes([observation], error_deg)
    return satisfies_all(point, halfplanes, eps)


def run_self_tests() -> None:
    """不依赖第三个测试文件的最小回归测试。"""

    result = solve_localization(equilateral_example())
    assert result.status == VALID_POLYGON
    assert len(result.vertices) == 3
    assert math.isclose(result.diameter or 0.0, 20.0, rel_tol=0.0, abs_tol=1e-6)
    assert result.covered is False

    square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    diameter, _, center, radius, _, margin, covered = diameter_and_coverage(square)
    assert math.isclose(diameter, 2.0 * math.sqrt(2.0), abs_tol=1e-12)
    assert math.isclose(center[0], 0.0, abs_tol=1e-12)
    assert math.isclose(center[1], 0.0, abs_tol=1e-12)
    assert math.isclose(radius, math.sqrt(2.0), abs_tol=1e-12)
    assert abs(margin) <= 1e-12
    assert covered is True

    origin_east = Observation(0.0, 0.0, 0.0)
    point_359_5 = direction(359.5)
    point_180 = direction(180.0)
    assert _point_in_observation_wedge(point_359_5, origin_east)
    assert not _point_in_observation_wedge(point_180, origin_east)

    unbounded = solve_localization([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)])
    assert unbounded.status == UNBOUNDED

    empty = solve_localization([(0.0, 0.0, 0.0), (-10.0, 0.0, 180.0)])
    assert empty.status == EMPTY

    degenerate = solve_localization([(0.0, 0.0, 0.0), (0.0, 0.0, 180.0)])
    assert degenerate.status == DEGENERATE

    reordered = solve_localization(list(reversed(equilateral_example())))
    assert reordered.status == VALID_POLYGON
    assert math.isclose(reordered.diameter or 0.0, result.diameter or 0.0, abs_tol=1e-7)
    assert reordered.covered == result.covered


def _print_result(result: LocalizationResult) -> None:
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="问题一几何定位算法")
    parser.add_argument(
        "--self-test", action="store_true", help="运行内置回归测试"
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="可选：把示例计算结果另存为 JSON 文件",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        print("All Q1 self-tests passed.")
        return

    result = solve_localization(equilateral_example())
    _print_result(result)
    if args.json:
        args.json.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
