from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog


Point = tuple[float, float]

VALID_POLYGON = "VALID_POLYGON"
EMPTY = "EMPTY"
UNBOUNDED = "UNBOUNDED"
DEGENERATE = "DEGENERATE"
INVALID_INPUT = "INVALID_INPUT"
NUMERIC_UNCERTAIN = "NUMERIC_UNCERTAIN"


@dataclass(frozen=True)
class Observation:
    """一次测向观测，方位角以正东为0度、逆时针为正。"""

    x: float
    y: float
    bearing_deg: float

    @property
    def station(self) -> Point:
        return self.x, self.y


@dataclass(frozen=True)
class HalfPlane:
    """标准半平面 a*x+b*y<=c。"""

    a: float
    b: float
    c: float


@dataclass
class LocalizationResult:
    status: str
    vertices: list[Point]
    diameter: float | None
    message: str


def _direction(angle_deg: float) -> Point:
    angle_rad = math.radians(angle_deg)
    return math.cos(angle_rad), math.sin(angle_rad)


def _dot(first: Point, second: Point) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _as_observation(value: Observation | Sequence[float]) -> Observation:
    if isinstance(value, Observation):
        observation = value
    else:
        if len(value) != 3:
            raise ValueError("每条观测必须包含x、y和示向度。")
        observation = Observation(float(value[0]), float(value[1]), float(value[2]))
    if not all(
        math.isfinite(item)
        for item in (observation.x, observation.y, observation.bearing_deg)
    ):
        raise ValueError("检测点坐标和示向度必须是有限数。")
    return observation


def build_halfplanes(
    observations: Iterable[Observation | Sequence[float]],
    error_deg: float,
) -> list[HalfPlane]:
    """把测向角域转换为半平面约束。"""

    observation_list = [_as_observation(item) for item in observations]
    if not observation_list:
        raise ValueError("至少需要一条测向观测。")
    if not math.isfinite(error_deg) or not 0.0 < error_deg < 90.0:
        raise ValueError("测角误差半张角必须位于0至90度之间。")

    halfplanes: list[HalfPlane] = []
    for observation in observation_list:
        station = observation.station
        lower = _direction(observation.bearing_deg - error_deg)
        upper = _direction(observation.bearing_deg + error_deg)

        # lower × (X-S) >= 0
        lower_normal = (lower[1], -lower[0])
        halfplanes.append(
            HalfPlane(
                lower_normal[0],
                lower_normal[1],
                _dot(lower_normal, station),
            )
        )

        # upper × (X-S) <= 0
        upper_normal = (-upper[1], upper[0])
        halfplanes.append(
            HalfPlane(
                upper_normal[0],
                upper_normal[1],
                _dot(upper_normal, station),
            )
        )
    return halfplanes


def _constraint_tolerance(point: Point, halfplane: HalfPlane) -> float:
    left = halfplane.a * point[0] + halfplane.b * point[1]
    return 1.0e-9 * max(1.0, abs(left), abs(halfplane.c))


def _satisfies_all(point: Point, halfplanes: Sequence[HalfPlane]) -> bool:
    for halfplane in halfplanes:
        left = halfplane.a * point[0] + halfplane.b * point[1]
        if left > halfplane.c + _constraint_tolerance(point, halfplane):
            return False
    return True


def _line_intersection(
    first: HalfPlane, second: HalfPlane
) -> Point | None:
    determinant = first.a * second.b - second.a * first.b
    if determinant == 0.0:
        return None
    x = (first.c * second.b - second.c * first.b) / determinant
    y = (first.a * second.c - second.a * first.c) / determinant
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _append_distinct(vertices: list[Point], candidate: Point) -> None:
    scale = max(1.0, abs(candidate[0]), abs(candidate[1]))
    tolerance = 1.0e-8 * scale
    if all(_distance(candidate, existing) > tolerance for existing in vertices):
        vertices.append(candidate)


def enumerate_vertices(halfplanes: Sequence[HalfPlane]) -> list[Point]:
    vertices: list[Point] = []
    for first_index in range(len(halfplanes) - 1):
        for second_index in range(first_index + 1, len(halfplanes)):
            candidate = _line_intersection(
                halfplanes[first_index], halfplanes[second_index]
            )
            if candidate is not None and _satisfies_all(candidate, halfplanes):
                _append_distinct(vertices, candidate)
    return vertices


def _find_feasible_point(halfplanes: Sequence[HalfPlane]) -> tuple[bool, Point | None]:
    """用线性规划检查半平面交集是否非空。"""

    matrix = np.asarray([[item.a, item.b] for item in halfplanes], dtype=float)
    bounds = np.asarray([item.c for item in halfplanes], dtype=float)
    result = linprog(
        c=np.zeros(2, dtype=float),
        A_ub=matrix,
        b_ub=bounds,
        bounds=[(None, None), (None, None)],
        method="highs",
    )
    if not result.success or result.x is None:
        return False, None
    witness = float(result.x[0]), float(result.x[1])
    return _satisfies_all(witness, halfplanes), witness


def _has_unbounded_direction(halfplanes: Sequence[HalfPlane]) -> bool:
    """检查衰退锥中是否存在非零方向。"""

    candidate_directions: list[Point] = []
    for halfplane in halfplanes:
        candidate_directions.extend(
            [
                (halfplane.b, -halfplane.a),
                (-halfplane.b, halfplane.a),
            ]
        )

    for direction in candidate_directions:
        length = math.hypot(*direction)
        if length == 0.0:
            continue
        unit = direction[0] / length, direction[1] / length
        if all(
            halfplane.a * unit[0] + halfplane.b * unit[1] <= 1.0e-12
            for halfplane in halfplanes
        ):
            return True
    return False


def _sort_counterclockwise(vertices: Sequence[Point]) -> list[Point]:
    center_x = sum(point[0] for point in vertices) / len(vertices)
    center_y = sum(point[1] for point in vertices) / len(vertices)
    return sorted(
        vertices,
        key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x),
    )


def _polygon_area(vertices: Sequence[Point]) -> float:
    if len(vertices) < 3:
        return 0.0
    total = 0.0
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        total += first[0] * second[1] - first[1] * second[0]
    return abs(total) / 2.0


def polygon_diameter(vertices: Sequence[Point]) -> float:
    if len(vertices) <= 1:
        return 0.0
    return max(
        _distance(vertices[first], vertices[second])
        for first in range(len(vertices) - 1)
        for second in range(first + 1, len(vertices))
    )


def solve_localization(
    observations: Iterable[Observation | Sequence[float]],
    error_deg: float = 1.0,
    known_feasible_point: Point | None = None,
) -> LocalizationResult:
    """求测向角域交集状态及其直径。

    问题二枚举真实源位置时，该源点必然属于两次允许角域，可作为已知可行点传入，
    从而避免对每个情景重复求解一次线性规划。独立调用时仍执行完整可行性检查。
    """

    try:
        halfplanes = build_halfplanes(observations, error_deg)
    except (TypeError, ValueError) as error:
        return LocalizationResult(INVALID_INPUT, [], None, str(error))

    if known_feasible_point is None:
        feasible, _ = _find_feasible_point(halfplanes)
    else:
        feasible = _satisfies_all(known_feasible_point, halfplanes)
    if not feasible:
        return LocalizationResult(EMPTY, [], None, "全部测向角域没有公共点。")

    vertices = enumerate_vertices(halfplanes)
    if _has_unbounded_direction(halfplanes):
        return LocalizationResult(
            UNBOUNDED,
            _sort_counterclockwise(vertices) if vertices else [],
            None,
            "测向角域交集无界。",
        )

    if len(vertices) < 3:
        return LocalizationResult(
            DEGENERATE,
            vertices,
            polygon_diameter(vertices),
            "角域交集退化为线段或点。",
        )

    ordered = _sort_counterclockwise(vertices)
    if _polygon_area(ordered) <= 1.0e-12:
        return LocalizationResult(
            DEGENERATE,
            ordered,
            polygon_diameter(ordered),
            "角域交集面积为零。",
        )

    return LocalizationResult(
        VALID_POLYGON,
        ordered,
        polygon_diameter(ordered),
        "计算成功。",
    )
