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
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mpmath as mp


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
    line_angle_deg: float
    observation_index: int
    boundary_name: str


@dataclass(frozen=True)
class NumericTolerance:
    # 这些量用于触发复核，不代表扩大题目的真实可行域。
    parallel_trigger: float = 1.0e-10
    feasible_relative: float = 1.0e-12
    duplicate_relative: float = 1.0e-11
    area_relative: float = 1.0e-12
    cover_relative: float = 1.0e-12
    high_precision_dps: int = 80


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
    coverage_status: str | None = None
    precision_note: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


VALID_POLYGON = "VALID_POLYGON"
EMPTY = "EMPTY"
UNBOUNDED = "UNBOUNDED"
DEGENERATE = "DEGENERATE"
INVALID_INPUT = "INVALID_INPUT"
NUMERIC_UNCERTAIN = "NUMERIC_UNCERTAIN"

COVERED = "COVERED"
NOT_COVERED = "NOT_COVERED"

PASS = "PASS"
FAIL = "FAIL"
UNCERTAIN = "UNCERTAIN"


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
                line_angle_deg=obs.bearing_deg - error_deg,
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
                line_angle_deg=obs.bearing_deg + error_deg,
                observation_index=index,
                boundary_name="upper",
            )
        )

    return obs_list, halfplanes


def _relative_band(values: Sequence[float], relative: float) -> float:
    return relative * max(sys.float_info.min, *(abs(value) for value in values))


def classify_halfplane_float(
    point: Point, halfplane: HalfPlane, relative: float
) -> str:
    """以三分判定检查半平面：明确满足、明确违反或临界。"""

    left = halfplane.a * point[0] + halfplane.b * point[1]
    residual = halfplane.c - left
    band = _relative_band((halfplane.c, left, 1.0), relative)
    if residual > band:
        return PASS
    if residual < -band:
        return FAIL
    return UNCERTAIN


def satisfies_halfplane(point: Point, halfplane: HalfPlane, eps: float) -> bool:
    """供可行性审计和简单点测使用；候选顶点使用三分判定。"""

    return halfplane.a * point[0] + halfplane.b * point[1] <= halfplane.c + eps


def satisfies_all(point: Point, halfplanes: Sequence[HalfPlane], eps: float) -> bool:
    return all(satisfies_halfplane(point, hp, eps) for hp in halfplanes)


def _mp_line_data(halfplane: HalfPlane) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    angle = mp.radians(mp.mpf(str(halfplane.line_angle_deg)))
    return (
        mp.mpf(str(halfplane.line_point[0])),
        mp.mpf(str(halfplane.line_point[1])),
        mp.cos(angle),
        mp.sin(angle),
    )


def _mp_halfplane_coefficients(
    halfplane: HalfPlane,
) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    px, py, ux, uy = _mp_line_data(halfplane)
    if halfplane.boundary_name == "lower":
        a, b = uy, -ux
    else:
        a, b = -uy, ux
    return a, b, a * px + b * py


def _angles_are_parallel(first: HalfPlane, second: HalfPlane) -> bool:
    difference = mp.fmod(
        mp.mpf(str(first.line_angle_deg)) - mp.mpf(str(second.line_angle_deg)),
        mp.mpf("180"),
    )
    return difference == 0


def _mp_line_intersection(
    first: HalfPlane, second: HalfPlane, dps: int
) -> tuple[mp.mpf, mp.mpf] | None:
    with mp.workdps(dps):
        if _angles_are_parallel(first, second):
            return None
        px, py, ux, uy = _mp_line_data(first)
        qx, qy, vx, vy = _mp_line_data(second)
        denominator = ux * vy - uy * vx
        if denominator == 0:
            return None
        t = ((qx - px) * vy - (qy - py) * vx) / denominator
        return px + t * ux, py + t * uy


def line_intersection(
    first: HalfPlane,
    second: HalfPlane,
    tolerance: NumericTolerance,
) -> tuple[Point | None, bool]:
    """求支撑线交点；小行列式触发高精度，而不是直接当作平行。"""

    p = first.line_point
    q = second.line_point
    u = first.line_direction
    v = second.line_direction
    denominator = cross(u, v)
    if abs(denominator) > tolerance.parallel_trigger:
        t = cross(subtract(q, p), v) / denominator
        return (p[0] + t * u[0], p[1] + t * u[1]), False

    high_precision_point = _mp_line_intersection(
        first, second, tolerance.high_precision_dps
    )
    if high_precision_point is None:
        return None, True
    x, y = high_precision_point
    if not (mp.isfinite(x) and mp.isfinite(y)):
        raise ArithmeticError("近平行边界的交点无法可靠表示。")
    return (float(x), float(y)), True


def _append_if_new(points: list[Point], candidate: Point, relative: float) -> None:
    def is_distinct(existing: Point) -> bool:
        scale = max(
            1.0,
            abs(candidate[0]),
            abs(candidate[1]),
            abs(existing[0]),
            abs(existing[1]),
        )
        threshold2 = (relative * scale) ** 2
        return distance_squared(candidate, existing) > threshold2

    if all(is_distinct(existing) for existing in points):
        points.append(candidate)


def _classify_candidate_high_precision(
    first: HalfPlane,
    second: HalfPlane,
    halfplanes: Sequence[HalfPlane],
    active_indices: tuple[int, int],
    dps: int,
) -> tuple[str, Point | None]:
    """重新求交并复核非活动约束；真正临界时返回 UNCERTAIN。"""

    with mp.workdps(dps):
        point = _mp_line_intersection(first, second, dps)
        if point is None:
            return FAIL, None
        x, y = point
        unresolved = False
        tiny = mp.power(10, -(dps - 20))
        for index, halfplane in enumerate(halfplanes):
            if index in active_indices:
                continue
            a, b, c = _mp_halfplane_coefficients(halfplane)
            left = a * x + b * y
            residual = c - left
            band = tiny * max(mp.mpf("1"), abs(c), abs(left))
            if residual < -band:
                return FAIL, None
            if residual != 0 and abs(residual) <= band:
                unresolved = True
        float_point = (float(x), float(y))
        return (UNCERTAIN if unresolved else PASS), float_point


def enumerate_feasible_intersections(
    halfplanes: Sequence[HalfPlane],
    tolerance: NumericTolerance,
) -> tuple[list[Point], bool]:
    """两两求交，并用全部约束筛选、去重。"""

    vertices: list[Point] = []
    has_uncertain_candidate = False
    for i in range(len(halfplanes) - 1):
        for j in range(i + 1, len(halfplanes)):
            point, used_high_precision = line_intersection(
                halfplanes[i], halfplanes[j], tolerance
            )
            if point is None:
                continue

            judgment = PASS
            if not used_high_precision:
                for index, halfplane in enumerate(halfplanes):
                    if index in (i, j):
                        continue
                    current = classify_halfplane_float(
                        point, halfplane, tolerance.feasible_relative
                    )
                    if current == FAIL:
                        judgment = FAIL
                        break
                    if current == UNCERTAIN:
                        judgment = UNCERTAIN

            if used_high_precision or judgment == UNCERTAIN:
                judgment, refined_point = _classify_candidate_high_precision(
                    halfplanes[i],
                    halfplanes[j],
                    halfplanes,
                    (i, j),
                    tolerance.high_precision_dps,
                )
                if refined_point is not None:
                    point = refined_point

            if judgment == UNCERTAIN:
                has_uncertain_candidate = True
            elif judgment == PASS:
                _append_if_new(vertices, point, tolerance.duplicate_relative)
    return vertices, has_uncertain_candidate


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


def relative_area_measure(vertices: Sequence[Point]) -> float:
    """返回无量纲面积指标 2*Area/D^2，避免固定面积阈值依赖尺度。"""

    if len(vertices) < 3:
        return 0.0
    scale2 = max(
        distance_squared(vertices[i], vertices[j])
        for i in range(len(vertices) - 1)
        for j in range(i + 1, len(vertices))
    )
    if scale2 <= sys.float_info.min:
        return 0.0
    return 2.0 * abs(polygon_signed_area(vertices)) / scale2


def _high_precision_coverage(
    vertices: Sequence[Point], dps: int
) -> tuple[bool | None, str]:
    """对覆盖临界情形进行高精度复核。"""

    with mp.workdps(dps):
        points = [(mp.mpf(str(x)), mp.mpf(str(y))) for x, y in vertices]
        best_d2 = mp.mpf("-1")
        pair = (0, 1)
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                current = dx * dx + dy * dy
                if current > best_d2:
                    best_d2 = current
                    pair = (i, j)

        first, second = points[pair[0]], points[pair[1]]
        center = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
        radius2 = best_d2 / 4
        max_distance2 = max(
            (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2
            for point in points
        )
        gap2 = radius2 - max_distance2
        scale = max(mp.mpf("1"), abs(radius2), abs(max_distance2))
        band = mp.power(10, -(dps - 20)) * scale
        if gap2 > band:
            return True, COVERED
        if gap2 < -band:
            return False, NOT_COVERED
        if gap2 == 0:
            # 对输入十进制坐标重新计算后得到严格相等，可按闭圆盘处理。
            return True, COVERED
        return None, NUMERIC_UNCERTAIN


def diameter_and_coverage(
    vertices: Sequence[Point],
    cover_relative: float = 1.0e-12,
    high_precision_dps: int = 80,
) -> tuple[
    float,
    tuple[Point, Point],
    Point,
    float,
    float,
    float,
    bool | None,
    str,
    str,
]:
    """枚举顶点对求直径，并以三分判定完成圆盘覆盖检查。"""

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
    max_center_distance2 = max(distance_squared(vertex, center) for vertex in vertices)
    max_center_distance = math.sqrt(max_center_distance2)
    margin = radius - max_center_distance
    radius2 = best_d2 / 4.0
    gap2 = radius2 - max_center_distance2
    band = _relative_band((radius2, max_center_distance2), cover_relative)
    if gap2 > band:
        covered, coverage_status = True, COVERED
        precision_note = "双精度结果明显位于覆盖侧。"
    elif gap2 < -band:
        covered, coverage_status = False, NOT_COVERED
        precision_note = "双精度结果明显位于不覆盖侧。"
    else:
        covered, coverage_status = _high_precision_coverage(
            vertices, high_precision_dps
        )
        precision_note = (
            f"覆盖余量接近零，已使用 {high_precision_dps} 位十进制精度复核。"
        )
    return (
        diameter,
        farthest_pair,
        center,
        radius,
        max_center_distance,
        margin,
        covered,
        coverage_status,
        precision_note,
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

    feasible, _ = find_feasible_point(halfplanes, tol.feasible_relative)
    if not feasible:
        return LocalizationResult(EMPTY, [], message="全部测向角域没有公共点。")

    try:
        vertices, has_uncertain_candidate = enumerate_feasible_intersections(
            halfplanes, tol
        )
    except (ArithmeticError, OverflowError) as exc:
        return LocalizationResult(NUMERIC_UNCERTAIN, [], message=str(exc))

    if has_unbounded_direction(halfplanes, tol.feasible_relative):
        return LocalizationResult(
            UNBOUNDED,
            sort_vertices_counterclockwise(vertices) if vertices else [],
            message="测向角域交集无界，因而没有有限的区域直径。",
        )

    if has_uncertain_candidate:
        return LocalizationResult(
            NUMERIC_UNCERTAIN,
            sort_vertices_counterclockwise(vertices) if vertices else [],
            message="至少一个候选交点接近非活动半平面边界，需进一步复核。",
        )

    if len(vertices) < 3:
        return LocalizationResult(
            DEGENERATE,
            vertices,
            message="角域交集退化为点或线段，不是正常二维定位多边形。",
        )

    vertices = sort_vertices_counterclockwise(vertices)
    area = abs(polygon_signed_area(vertices))
    relative_area = relative_area_measure(vertices)
    if area == 0.0:
        return LocalizationResult(
            DEGENERATE,
            vertices,
            message="角域交集面积为零，不是正常二维定位多边形。",
        )
    if relative_area <= tol.area_relative:
        return LocalizationResult(
            NUMERIC_UNCERTAIN,
            vertices,
            message=(
                "定位区域极度狭长，尺度归一化面积接近零，"
                "不能仅凭固定面积容差判为退化。"
            ),
        )

    (
        diameter,
        farthest_pair,
        center,
        radius,
        max_center_distance,
        margin,
        covered,
        coverage_status,
        precision_note,
    ) = diameter_and_coverage(
        vertices, tol.cover_relative, tol.high_precision_dps
    )

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
        coverage_status=coverage_status,
        precision_note=precision_note,
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
    assert result.coverage_status == NOT_COVERED

    square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    diameter, _, center, radius, _, margin, covered, status, _ = (
        diameter_and_coverage(square)
    )
    assert math.isclose(diameter, 2.0 * math.sqrt(2.0), abs_tol=1e-12)
    assert math.isclose(center[0], 0.0, abs_tol=1e-12)
    assert math.isclose(center[1], 0.0, abs_tol=1e-12)
    assert math.isclose(radius, math.sqrt(2.0), abs_tol=1e-12)
    assert abs(margin) <= 1e-12
    assert covered is True
    assert status == COVERED

    # 圆周内外的微小扰动不得被一个宽松绝对容差混为一谈。
    inside = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0 - 1.0e-9)]
    outside = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0 + 1.0e-9)]
    assert diameter_and_coverage(inside)[7] == COVERED
    assert diameter_and_coverage(outside)[7] == NOT_COVERED

    origin_east = Observation(0.0, 0.0, 0.0)
    point_359_5 = direction(359.5)
    point_180 = direction(180.0)
    assert _point_in_observation_wedge(point_359_5, origin_east)
    assert not _point_in_observation_wedge(point_180, origin_east)

    # 夹角虽极小但不为零的边界必须求交，不能按固定阈值直接跳过。
    _, near_parallel_halfplanes = build_halfplanes(
        [(0.0, 0.0, 1.0), (0.0, 1.0, 1.0 + 1.0e-12)]
    )
    near_point, used_high_precision = line_intersection(
        near_parallel_halfplanes[0],
        near_parallel_halfplanes[2],
        NumericTolerance(),
    )
    assert used_high_precision
    assert near_point is not None
    assert abs(near_point[0]) > 1.0e10

    # 尺度归一化面积对相似图形应保持不变。
    thin = [(0.0, 0.0), (10.0, 0.0), (10.0, 1.0e-5), (0.0, 1.0e-5)]
    thin_scaled = [(1.0e6 * x, 1.0e6 * y) for x, y in thin]
    assert math.isclose(
        relative_area_measure(thin),
        relative_area_measure(thin_scaled),
        rel_tol=1.0e-12,
    )

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
