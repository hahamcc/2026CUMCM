#!/usr/bin/env python3
"""CUMCM 2026 B 题问题三：基础版 B / 增强版 B+ 自动搜索定位清除程序。

安全约定：
1. 默认只打印离线计划，不连接模拟器。
2. 演练必须显式使用 --connect --mode practice，并在终端输入确认词。
3. 正式模式默认由 FORMAL_UNLOCKED=False 硬锁，防止误用正式机会。
4. 本文件只依赖 Python 3.13 标准库。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


# ============================ 运行安全锁 ============================

# 演练阶段必须保持 False。正式测试前由全队确认代码与演练结果后手工改为 True。
FORMAL_UNLOCKED = False
PRACTICE_CONFIRM_TEXT = "PRACTICE-Q3"
FORMAL_CONFIRM_TEXT = "USE-ONE-Q3-FORMAL-ATTEMPT"


# ============================ 题面参数 ============================

ARENA_ID = "default"
DEFAULT_BASE_URL = "http://127.0.0.1:2026"
TARGET_RADIUS_M = 1800.0
MIN_RECEIVE_RADIUS_M = 1000.0
MAX_RECEIVE_RADIUS_M = 1500.0
BEARING_ERROR_DEG = 1.0
CLEAR_RADIUS_M = 20.0
SURVEY_RING_RADIUS_M = 1150.0


# ============================ 程序参数 ============================

# 圆盘使用外切正多边形近似，始终保持“真实位置集合的外包”。
CIRCLE_SIDES = 360

# 只用于防止浮点边界误判，不代表新的物理误差。
SAFE_MEASURE_NUMERIC_MARGIN_M = 0.20
CLEAR_NUMERIC_MARGIN_M = 0.10

# 补测和保底参数均为演练阶段的初始值，必须根据演练记录复核。
GRID_CELL_M = 28.0
GRID_AFTER_DIRECTION_COUNT = 6
LOW_GAIN_RATIO = 0.90
LOW_GAIN_STREAK_TRIGGER = 2
GRID_POINT_WARNING = 500
MAX_TOTAL_ACTIONS = 10_000
REAL_TIME_EXIT_RESERVE_S = 20.0

# B+ 不删除七点托底任务；只允许经完整时间比较的外侧小清除宏任务产生少量绕路。
BPLUS_MAX_EXTRA_CHANNELS_PER_STOP = 2
BPLUS_REQUIRED_RADIUS_RATIO = 0.80
BPLUS_INSERTION_BUFFER_S = 6.0
SEGMENT_ENDPOINT_GAP_M = 1.0
EXCLUSION_NUMERIC_MARGIN_M = 0.10

HTTP_TIMEOUT_S = 5.0
HTTP_RETRIES = 3
HTTP_RETRY_DELAYS_S = (0.4, 1.0, 2.0)

Point = tuple[float, float]


class Q3Error(RuntimeError):
    """问题三程序可说明的错误。"""


class ActionRejected(Q3Error):
    """模拟器形成了响应，但没有接受动作。"""


class SafetyStop(Q3Error):
    """触发防误操作、时间或请求数量保护。"""


class GeometryError(Q3Error):
    """保守几何区域出现不应发生的异常。"""


class ChannelStatus(str, Enum):
    UNKNOWN = "unknown_unexcluded"
    ACTIVE = "found_uncleared"
    CLEARED = "cleared"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class Observation:
    position: Point
    bearing_deg: float


@dataclass
class ChannelRecord:
    channel: int
    status: ChannelStatus = ChannelStatus.UNKNOWN
    observations: list[Observation] = field(default_factory=list)
    no_signal_points: list[Point] = field(default_factory=list)
    clear_fail_points: list[Point] = field(default_factory=list)
    polygon: list[Point] | None = None
    low_gain_streak: int = 0
    small_clear_plan_cache: tuple[ClearPlan, ...] | None = None


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float


@dataclass(frozen=True)
class CandidateScore:
    point: Point
    worst_radius: float
    predicted_finish_time_s: float | None
    safe_margin_m: float


@dataclass(frozen=True)
class ClearPlan:
    """经几何证明的一点或两点清除方案。

    两点方案中的存储顺序不是执行顺序；执行时会同时比较两种排列。
    """

    points: tuple[Point, ...]
    proof: str


@dataclass(frozen=True)
class ActionPreview:
    """B+ 滚动路径中某个频道的下一个安全宏任务。"""

    channel: int
    kind: str
    points: tuple[Point, ...]
    priority: int


@dataclass
class ActionStats:
    move_distance_m: float = 0.0
    measure_count: int = 0
    switch_count: int = 0
    clear_success_count: int = 0
    clear_failure_count: int = 0
    bplus_edge_stop_count: int = 0
    bplus_extra_measure_count: int = 0
    bplus_opportunity_clear_count: int = 0
    outside_opportunity_count: int = 0
    single_clear_macro_count: int = 0
    two_clear_macro_count: int = 0
    two_clear_first_success_count: int = 0
    opportunity_detour_distance_m: float = 0.0
    rolling_replan_count: int = 0

    @property
    def action_count(self) -> int:
        return self.measure_count + self.clear_success_count + self.clear_failure_count


# ============================ 基础几何 ============================


def add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def subtract(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def scale(a: Point, factor: float) -> Point:
    return a[0] * factor, a[1] * factor


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(a: Point) -> Point:
    length = norm(a)
    if length <= 1.0e-12:
        return 1.0, 0.0
    return a[0] / length, a[1] / length


def bearing_deg(origin: Point, target: Point) -> float:
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0])) % 360.0


def survey_points() -> list[Point]:
    return [(0.0, 0.0), *outer_survey_points()]


def outer_survey_points() -> list[Point]:
    points: list[Point] = []
    for index in range(6):
        angle = index * math.pi / 3.0
        points.append(
            (
                SURVEY_RING_RADIUS_M * math.cos(angle),
                SURVEY_RING_RADIUS_M * math.sin(angle),
            )
        )
    return points


def outer_survey_routes() -> list[tuple[int, int, list[Point]]]:
    """六个起点乘两个方向；每条路线均含全部外围点且长度相同。"""

    outer = outer_survey_points()
    routes: list[tuple[int, int, list[Point]]] = []
    for start in range(6):
        for direction in (1, -1):
            order = [outer[(start + direction * step) % 6] for step in range(6)]
            routes.append((start, direction, order))
    return routes


def regular_outer_polygon(center: Point, radius: float, sides: int = CIRCLE_SIDES) -> list[Point]:
    """返回圆的外切正多边形，保证真实圆盘包含在多边形内。"""

    if sides < 8:
        raise ValueError("外切多边形边数至少为 8。")
    vertex_radius = radius / math.cos(math.pi / sides)
    return [
        (
            center[0] + vertex_radius * math.cos((2 * k + 1) * math.pi / sides),
            center[1] + vertex_radius * math.sin((2 * k + 1) * math.pi / sides),
        )
        for k in range(sides)
    ]


def _deduplicate_polygon(vertices: Sequence[Point], tolerance: float = 1.0e-8) -> list[Point]:
    result: list[Point] = []
    for point in vertices:
        if not result or distance(point, result[-1]) > tolerance:
            result.append(point)
    if len(result) > 1 and distance(result[0], result[-1]) <= tolerance:
        result.pop()
    return result


def clip_halfplane(
    polygon: Sequence[Point],
    a: float,
    b: float,
    c: float,
    tolerance: float = 1.0e-8,
) -> list[Point]:
    """Sutherland-Hodgman：保留 a*x+b*y<=c 的部分。"""

    if not polygon:
        return []

    def residual(point: Point) -> float:
        return a * point[0] + b * point[1] - c

    output: list[Point] = []
    previous = polygon[-1]
    previous_value = residual(previous)
    previous_inside = previous_value <= tolerance

    for current in polygon:
        current_value = residual(current)
        current_inside = current_value <= tolerance
        if previous_inside != current_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1.0e-18:
                ratio = previous_value / denominator
                output.append(
                    (
                        previous[0] + ratio * (current[0] - previous[0]),
                        previous[1] + ratio * (current[1] - previous[1]),
                    )
                )
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside

    return _deduplicate_polygon(output)


def clip_outer_disk(
    polygon: Sequence[Point],
    center: Point,
    radius: float,
    sides: int = CIRCLE_SIDES,
) -> list[Point]:
    """用圆的外切正多边形约束裁剪，结果仍是实际圆盘交集的外包。"""

    result = list(polygon)
    for index in range(sides):
        angle = 2.0 * math.pi * index / sides
        a = math.cos(angle)
        b = math.sin(angle)
        c = a * center[0] + b * center[1] + radius
        result = clip_halfplane(result, a, b, c)
        if not result:
            break
    return result


def clip_bearing_wedge(
    polygon: Sequence[Point],
    station: Point,
    measured_bearing_deg: float,
    error_deg: float = BEARING_ERROR_DEG,
) -> list[Point]:
    """把示向度 ±error_deg 转成两个带正向约束的半平面。"""

    lower = math.radians(measured_bearing_deg - error_deg)
    upper = math.radians(measured_bearing_deg + error_deg)
    u_lower = math.cos(lower), math.sin(lower)
    u_upper = math.cos(upper), math.sin(upper)

    # cross(u_lower, X-station) >= 0
    a1, b1 = u_lower[1], -u_lower[0]
    c1 = a1 * station[0] + b1 * station[1]

    # cross(u_upper, X-station) <= 0
    a2, b2 = -u_upper[1], u_upper[0]
    c2 = a2 * station[0] + b2 * station[1]

    result = clip_halfplane(polygon, a1, b1, c1)
    return clip_halfplane(result, a2, b2, c2)


def polygon_area(polygon: Sequence[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(
        sum(cross(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))
    )


def polygon_diameter(polygon: Sequence[Point]) -> tuple[float, tuple[Point, Point]]:
    if not polygon:
        raise GeometryError("空定位区域没有直径。")
    if len(polygon) == 1:
        return 0.0, (polygon[0], polygon[0])
    best = -1.0
    pair = (polygon[0], polygon[1])
    for i in range(len(polygon) - 1):
        for j in range(i + 1, len(polygon)):
            current = distance(polygon[i], polygon[j])
            if current > best:
                best = current
                pair = polygon[i], polygon[j]
    return best, pair


def _circle_from_two(a: Point, b: Point) -> Circle:
    center = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return Circle(center, distance(a, b) / 2.0)


def _circle_from_three(a: Point, b: Point, c: Point) -> Circle:
    determinant = 2.0 * (
        a[0] * (b[1] - c[1])
        + b[0] * (c[1] - a[1])
        + c[0] * (a[1] - b[1])
    )
    if abs(determinant) <= 1.0e-12:
        pairs = ((a, b), (a, c), (b, c))
        return max((_circle_from_two(p, q) for p, q in pairs), key=lambda item: item.radius)

    a2 = dot(a, a)
    b2 = dot(b, b)
    c2 = dot(c, c)
    ux = (
        a2 * (b[1] - c[1])
        + b2 * (c[1] - a[1])
        + c2 * (a[1] - b[1])
    ) / determinant
    uy = (
        a2 * (c[0] - b[0])
        + b2 * (a[0] - c[0])
        + c2 * (b[0] - a[0])
    ) / determinant
    center = ux, uy
    return Circle(center, distance(center, a))


def _circle_contains(circle: Circle, point: Point) -> bool:
    tolerance = 1.0e-8 * max(1.0, circle.radius)
    return distance(circle.center, point) <= circle.radius + tolerance


def minimum_enclosing_circle(points: Sequence[Point]) -> Circle:
    """确定性随机顺序的增量最小覆盖圆算法。"""

    if not points:
        raise GeometryError("空集合没有最小覆盖圆。")
    ordered = list(points)
    random.Random(20260912).shuffle(ordered)
    circle: Circle | None = None

    for i, p in enumerate(ordered):
        if circle is not None and _circle_contains(circle, p):
            continue
        circle = Circle(p, 0.0)
        for j in range(i):
            q = ordered[j]
            if _circle_contains(circle, q):
                continue
            circle = _circle_from_two(p, q)
            for k in range(j):
                r = ordered[k]
                if _circle_contains(circle, r):
                    continue
                circle = _circle_from_three(p, q, r)

    assert circle is not None
    return circle


def point_in_polygon(point: Point, polygon: Sequence[Point], tolerance: float = 1.0e-8) -> bool:
    """适用于凸多边形，也容许点位于边界。"""

    positive = False
    negative = False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        value = cross(subtract(second, first), subtract(point, first))
        positive = positive or value > tolerance
        negative = negative or value < -tolerance
        if positive and negative:
            return False
    return True


def _on_segment(a: Point, b: Point, p: Point, tolerance: float = 1.0e-8) -> bool:
    return (
        abs(cross(subtract(b, a), subtract(p, a))) <= tolerance
        and min(a[0], b[0]) - tolerance <= p[0] <= max(a[0], b[0]) + tolerance
        and min(a[1], b[1]) - tolerance <= p[1] <= max(a[1], b[1]) + tolerance
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    ab_c = cross(subtract(b, a), subtract(c, a))
    ab_d = cross(subtract(b, a), subtract(d, a))
    cd_a = cross(subtract(d, c), subtract(a, c))
    cd_b = cross(subtract(d, c), subtract(b, c))
    tolerance = 1.0e-8
    if ((ab_c > tolerance and ab_d < -tolerance) or (ab_c < -tolerance and ab_d > tolerance)) and (
        (cd_a > tolerance and cd_b < -tolerance) or (cd_a < -tolerance and cd_b > tolerance)
    ):
        return True
    return any(
        (
            abs(ab_c) <= tolerance and _on_segment(a, b, c),
            abs(ab_d) <= tolerance and _on_segment(a, b, d),
            abs(cd_a) <= tolerance and _on_segment(c, d, a),
            abs(cd_b) <= tolerance and _on_segment(c, d, b),
        )
    )


def polygons_intersect(first: Sequence[Point], second: Sequence[Point]) -> bool:
    """判断两个凸多边形是否相交或接触。"""

    if not first or not second:
        return False
    if any(point_in_polygon(point, second) for point in first):
        return True
    if any(point_in_polygon(point, first) for point in second):
        return True
    first_edges = list(zip(first, first[1:] + first[:1]))
    second_edges = list(zip(second, second[1:] + second[:1]))
    return any(_segments_intersect(a, b, c, d) for a, b in first_edges for c, d in second_edges)


def point_to_segment_distance(point: Point, first: Point, second: Point) -> float:
    direction = subtract(second, first)
    denominator = dot(direction, direction)
    if denominator <= 1.0e-18:
        return distance(point, first)
    parameter = max(0.0, min(1.0, dot(subtract(point, first), direction) / denominator))
    projection = add(first, scale(direction, parameter))
    return distance(point, projection)


def point_to_polygon_distance(point: Point, polygon: Sequence[Point]) -> float:
    if not polygon:
        return math.inf
    if point_in_polygon(point, polygon):
        return 0.0
    return min(
        point_to_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def survey_hexagon() -> list[Point]:
    return outer_survey_points()


def polygon_is_strictly_outside_survey_hexagon(polygon: Sequence[Point]) -> bool:
    """只有与外围正六边形不相交、不接触时才记为外侧区域。"""

    return bool(polygon) and not polygons_intersect(polygon, survey_hexagon())


def small_clear_plans(polygon: Sequence[Point]) -> list[ClearPlan]:
    """生成经证明的一点或两点清除方案。

    两点方案只沿当前凸多边形的直径方向切分。每个子区域均由自己的
    19.9 m 最小覆盖圆覆盖，所以找到的方案是充分且可核验的；找不到不表示
    数学上不存在其他两圆覆盖。
    """

    if not polygon:
        return []
    safe_radius = CLEAR_RADIUS_M - CLEAR_NUMERIC_MARGIN_M
    whole = minimum_enclosing_circle(polygon)
    if whole.radius <= safe_radius:
        return [ClearPlan((whole.center,), "minimum_enclosing_circle")]

    diameter_value, (first, second) = polygon_diameter(polygon)
    if diameter_value <= 1.0e-9:
        return []
    axis = unit(subtract(second, first))
    projections = sorted({round(dot(axis, point), 9) for point in polygon})
    if len(projections) < 2:
        return []

    candidates: list[ClearPlan] = []
    seen: set[tuple[float, float, float, float]] = set()
    for lower, upper in zip(projections, projections[1:]):
        if upper - lower <= 1.0e-8:
            continue
        threshold = (lower + upper) / 2.0
        left = clip_halfplane(polygon, axis[0], axis[1], threshold)
        right = clip_halfplane(polygon, -axis[0], -axis[1], -threshold)
        if polygon_area(left) <= 1.0e-9 or polygon_area(right) <= 1.0e-9:
            continue
        left_circle = minimum_enclosing_circle(left)
        right_circle = minimum_enclosing_circle(right)
        if left_circle.radius > safe_radius or right_circle.radius > safe_radius:
            continue
        if any(distance(left_circle.center, point) > safe_radius + 1.0e-7 for point in left):
            continue
        if any(distance(right_circle.center, point) > safe_radius + 1.0e-7 for point in right):
            continue
        key = (
            round(left_circle.center[0], 6),
            round(left_circle.center[1], 6),
            round(right_circle.center[0], 6),
            round(right_circle.center[1], 6),
        )
        reverse_key = key[2], key[3], key[0], key[1]
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        candidates.append(
            ClearPlan(
                (left_circle.center, right_circle.center),
                f"diameter_halfplane_split@{threshold:.6f}",
            )
        )
    return candidates


def clear_sequence_worst_time_s(
    start: Point,
    points: Sequence[Point],
    end: Point | None,
) -> float:
    """清除序列前 k-1 次失败、第 k 次成功后前往 end 的最坏虚拟时间。"""

    if not points:
        raise GeometryError("空清除序列无法计价。")
    traveled = 0.0
    previous = start
    branches: list[float] = []
    for index, point in enumerate(points):
        traveled += distance(previous, point)
        final_leg = 0.0 if end is None else distance(point, end)
        branches.append((traveled + final_leg) / 5.0 + 3.0 * index + 5.0)
        previous = point
    return max(branches)


def best_clear_order(
    start: Point,
    plan: ClearPlan,
    end: Point | None,
) -> tuple[tuple[Point, ...], float]:
    choices = []
    for order in itertools.permutations(plan.points):
        choices.append((clear_sequence_worst_time_s(start, order, end), tuple(order)))
    time_value, order = min(choices, key=lambda item: (item[0], item[1]))
    return order, time_value


def should_insert_outside_macro(
    now_time_s: float,
    wait_time_s: float,
    buffer_s: float = BPLUS_INSERTION_BUFFER_S,
) -> bool:
    """现在方案比推迟方案多耗达到 buffer 时拒绝，否则利用最近点位置机会。"""

    return now_time_s - wait_time_s < buffer_s - 1.0e-9


def action_route_distance(start: Point, actions: Sequence[ActionPreview]) -> float:
    """按两点宏任务最坏分支计算一条开放路线的移动距离。"""

    if not actions:
        return 0.0
    states: list[tuple[float, Point]] = [(0.0, start)]
    for action in actions:
        next_states: list[tuple[float, Point]] = []
        for order in itertools.permutations(action.points):
            internal = sum(distance(order[i], order[i + 1]) for i in range(len(order) - 1))
            best_cost = min(cost + distance(endpoint, order[0]) + internal for cost, endpoint in states)
            next_states.append((best_cost, order[-1]))
        states = next_states
    return min(cost for cost, _ in states)


def open_action_order(start: Point, actions: Sequence[ActionPreview]) -> list[ActionPreview]:
    """最近插入加一轮确定性 2-opt，返回不回到起点的宏任务顺序。"""

    remaining = sorted(actions, key=lambda item: (item.priority, item.channel, item.points))
    route: list[ActionPreview] = []
    while remaining:
        best: tuple[float, int, int, tuple[Point, ...], ActionPreview, list[ActionPreview]] | None = None
        base = action_route_distance(start, route)
        for action in remaining:
            for index in range(len(route) + 1):
                candidate_route = [*route[:index], action, *route[index:]]
                increase = action_route_distance(start, candidate_route) - base
                key = (increase, action.priority, action.channel, action.points, action, candidate_route)
                if best is None or key[:4] < best[:4]:
                    best = key
        assert best is not None
        selected = best[4]
        route = best[5]
        remaining.remove(selected)

    if len(route) >= 2:
        best_route = route
        best_cost = action_route_distance(start, route)
        for first_index in range(len(route) - 1):
            for last_index in range(first_index + 1, len(route)):
                candidate = [
                    *route[:first_index],
                    *reversed(route[first_index : last_index + 1]),
                    *route[last_index + 1 :],
                ]
                candidate_cost = action_route_distance(start, candidate)
                if candidate_cost < best_cost - 1.0e-8:
                    best_route = candidate
                    best_cost = candidate_cost
        route = best_route
    return route


def _segment_disk_interval(
    start: Point,
    end: Point,
    center: Point,
    radius: float,
) -> tuple[float, float] | None:
    """返回线段参数 t∈[0,1] 中位于给定闭圆盘内的区间。"""

    direction = subtract(end, start)
    quadratic = dot(direction, direction)
    if quadratic <= 1.0e-18:
        return (0.0, 1.0) if distance(start, center) <= radius else None
    offset = subtract(start, center)
    linear = 2.0 * dot(offset, direction)
    constant = dot(offset, offset) - radius * radius
    discriminant = linear * linear - 4.0 * quadratic * constant
    if discriminant < -1.0e-8:
        return None
    root = math.sqrt(max(0.0, discriminant))
    lower = max(0.0, (-linear - root) / (2.0 * quadratic))
    upper = min(1.0, (-linear + root) / (2.0 * quadratic))
    if lower > upper + 1.0e-12:
        return None
    return lower, upper


def segment_cover_center_interval(
    start: Point,
    end: Point,
    polygon: Sequence[Point],
    radius: float,
    interior_only: bool = True,
) -> tuple[float, float] | None:
    """线段上能以 radius 圆覆盖整个凸多边形的参数区间。

    到凸多边形的最大距离在顶点取得，因此逐顶点求圆盘区间再取交即可。
    """

    if not polygon:
        return None
    lower, upper = 0.0, 1.0
    for vertex in polygon:
        interval = _segment_disk_interval(start, end, vertex, radius)
        if interval is None:
            return None
        lower = max(lower, interval[0])
        upper = min(upper, interval[1])
        if lower > upper + 1.0e-12:
            return None

    if interior_only:
        length = distance(start, end)
        if length <= 2.0 * SEGMENT_ENDPOINT_GAP_M:
            return None
        gap = SEGMENT_ENDPOINT_GAP_M / length
        lower = max(lower, gap)
        upper = min(upper, 1.0 - gap)
    if lower > upper + 1.0e-12:
        return None
    return lower, upper


def interpolate_segment(start: Point, end: Point, parameter: float) -> Point:
    return (
        start[0] + parameter * (end[0] - start[0]),
        start[1] + parameter * (end[1] - start[1]),
    )


def square_intersects_polygon(
    center: Point,
    side: float,
    polygon: Sequence[Point],
) -> bool:
    half = side / 2.0
    min_x, max_x = center[0] - half, center[0] + half
    min_y, max_y = center[1] - half, center[1] + half
    corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    if any(min_x - 1.0e-8 <= p[0] <= max_x + 1.0e-8 and min_y - 1.0e-8 <= p[1] <= max_y + 1.0e-8 for p in polygon):
        return True
    if any(point_in_polygon(corner, polygon) for corner in corners):
        return True

    square_edges = list(zip(corners, corners[1:] + corners[:1]))
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        if any(_segments_intersect(first, second, edge_a, edge_b) for edge_a, edge_b in square_edges):
            return True
    return False


def grid_cover_points(
    polygon: Sequence[Point],
    start: Point,
    exclusion_disks: Sequence[Circle] = (),
) -> list[Point]:
    """返回覆盖剩余可能域的 28 m 方格中心，按逐行蛇形访问。

    只有整个方格都落入某个可靠排除圆时才删除该格；边界格一律保留。
    """

    if not polygon:
        raise GeometryError("不能为一个空区域生成清除网格。")
    side = GRID_CELL_M
    min_x = min(p[0] for p in polygon)
    max_x = max(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon)
    max_y = max(p[1] for p in polygon)
    i_min = math.floor(min_x / side)
    i_max = math.floor(max_x / side)
    j_min = math.floor(min_y / side)
    j_max = math.floor(max_y / side)

    rows: dict[int, list[tuple[int, Point]]] = {}
    for j in range(j_min, j_max + 1):
        for i in range(i_min, i_max + 1):
            center = ((i + 0.5) * side, (j + 0.5) * side)
            half = side / 2.0
            corners = [
                (center[0] - half, center[1] - half),
                (center[0] + half, center[1] - half),
                (center[0] + half, center[1] + half),
                (center[0] - half, center[1] + half),
            ]
            fully_excluded = any(
                max(distance(corner, disk.center) for corner in corners)
                <= disk.radius - EXCLUSION_NUMERIC_MARGIN_M
                for disk in exclusion_disks
            )
            if not fully_excluded and square_intersects_polygon(center, side, polygon):
                rows.setdefault(j, []).append((i, center))

    ordered: list[Point] = []
    for row_number, j in enumerate(sorted(rows)):
        row = sorted(rows[j], key=lambda item: item[0], reverse=bool(row_number % 2))
        ordered.extend(point for _, point in row)

    if ordered and distance(start, ordered[-1]) < distance(start, ordered[0]):
        ordered.reverse()
    return ordered


# ============================ 定位域与补测候选 ============================


def initial_target_polygon() -> list[Point]:
    return regular_outer_polygon((0.0, 0.0), TARGET_RADIUS_M)


def update_direction_region(
    current_polygon: Sequence[Point] | None,
    station: Point,
    measured_bearing: float,
) -> list[Point]:
    polygon = list(current_polygon) if current_polygon is not None else initial_target_polygon()
    polygon = clip_bearing_wedge(polygon, station, measured_bearing)
    # 对每次成功接收均加入 1500 m 上界；安全补测时该约束通常是冗余的。
    polygon = clip_outer_disk(polygon, station, MAX_RECEIVE_RADIUS_M)
    if not polygon or polygon_area(polygon) <= 1.0e-9:
        raise GeometryError("测向约束交集为空或退化；请检查角度、坐标和数值外包。")
    return polygon


def safe_measure_margin(polygon: Sequence[Point], point: Point) -> float:
    return MIN_RECEIVE_RADIUS_M - max(distance(point, vertex) for vertex in polygon)


def is_safe_measure_point(polygon: Sequence[Point], point: Point) -> bool:
    return safe_measure_margin(polygon, point) >= SAFE_MEASURE_NUMERIC_MARGIN_M


def _max_safe_step(center: Point, direction_vector: Point, polygon: Sequence[Point]) -> float:
    """从 center 沿单位方向移动时，仍距全部顶点不超过安全半径的最大步长。"""

    direction_unit = unit(direction_vector)
    radius = MIN_RECEIVE_RADIUS_M - SAFE_MEASURE_NUMERIC_MARGIN_M
    upper = math.inf
    for vertex in polygon:
        offset = subtract(center, vertex)
        projection = dot(direction_unit, offset)
        radicand = projection * projection + radius * radius - dot(offset, offset)
        if radicand < -1.0e-7:
            return 0.0
        root = -projection + math.sqrt(max(0.0, radicand))
        upper = min(upper, root)
    return max(0.0, upper if math.isfinite(upper) else 0.0)


def _not_repeated(point: Point, observations: Sequence[Observation]) -> bool:
    return all(distance(point, item.position) >= 1.0 for item in observations)


def generate_measure_candidates(record: ChannelRecord, current_position: Point) -> list[Point]:
    if not record.polygon:
        return []
    polygon = record.polygon
    circle = minimum_enclosing_circle(polygon)
    if not is_safe_measure_point(polygon, circle.center):
        return []

    diameter_value, (first, second) = polygon_diameter(polygon)
    long_axis = unit(subtract(second, first)) if diameter_value > 1.0e-9 else (1.0, 0.0)
    normal_axis = -long_axis[1], long_axis[0]
    toward_current = unit(subtract(current_position, circle.center))

    candidates: list[Point] = [circle.center]
    directions = [normal_axis, scale(normal_axis, -1.0), long_axis, scale(long_axis, -1.0), toward_current]
    for direction_vector in directions:
        maximum = _max_safe_step(circle.center, direction_vector, polygon)
        if maximum <= 1.0:
            continue
        fractions = (0.55, 0.85) if abs(dot(unit(direction_vector), normal_axis)) > 0.99 else (0.85,)
        for fraction in fractions:
            candidates.append(add(circle.center, scale(unit(direction_vector), fraction * maximum)))

    unique: list[Point] = []
    for point in candidates:
        if (
            abs(point[0]) <= 2_000_000
            and abs(point[1]) <= 2_000_000
            and is_safe_measure_point(polygon, point)
            and _not_repeated(point, record.observations)
            and all(distance(point, existing) > 0.1 for existing in unique)
        ):
            unique.append(point)
    return unique


def _polygon_samples(polygon: Sequence[Point], maximum: int = 24) -> list[Point]:
    candidates: list[Point] = []
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        candidates.append(first)
        candidates.append(((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0))
    if len(candidates) <= maximum:
        return candidates
    return [candidates[round(i * (len(candidates) - 1) / (maximum - 1))] for i in range(maximum)]


def evaluate_candidate(
    record: ChannelRecord,
    candidate: Point,
    current_position: Point,
    current_channel: int,
) -> CandidateScore:
    if not record.polygon:
        raise GeometryError("频道尚无定位区域。")
    margin = safe_measure_margin(record.polygon, candidate)
    if margin < SAFE_MEASURE_NUMERIC_MARGIN_M:
        return CandidateScore(candidate, math.inf, None, margin)

    worst_radius = 0.0
    worst_clear_move = 0.0
    all_scenarios_clearable = True
    errors = (-1.0, -0.5, 0.0, 0.5, 1.0)
    for possible_source in _polygon_samples(record.polygon):
        if distance(candidate, possible_source) <= 5.0:
            continue
        true_bearing = bearing_deg(candidate, possible_source)
        for error in errors:
            simulated = clip_bearing_wedge(record.polygon, candidate, true_bearing + error)
            if not simulated:
                continue
            circle = minimum_enclosing_circle(simulated)
            worst_radius = max(worst_radius, circle.radius)
            if circle.radius <= CLEAR_RADIUS_M - CLEAR_NUMERIC_MARGIN_M:
                worst_clear_move = max(worst_clear_move, distance(candidate, circle.center))
            else:
                all_scenarios_clearable = False

    switch_time = 0.0 if current_channel == record.channel else 1.0
    predicted_finish: float | None = None
    if all_scenarios_clearable:
        predicted_finish = (
            distance(current_position, candidate) / 5.0
            + switch_time
            + 5.0
            + worst_clear_move / 5.0
            + 5.0
        )
    return CandidateScore(candidate, worst_radius, predicted_finish, margin)


def select_measure_candidate(
    record: ChannelRecord,
    current_position: Point,
    current_channel: int,
) -> CandidateScore | None:
    scores = [
        evaluate_candidate(record, point, current_position, current_channel)
        for point in generate_measure_candidates(record, current_position)
    ]
    finite = [item for item in scores if math.isfinite(item.worst_radius)]
    if not finite:
        return None

    def key(item: CandidateScore) -> tuple[float, float, float]:
        if item.predicted_finish_time_s is not None:
            return 0.0, item.predicted_finish_time_s, item.worst_radius
        immediate = distance(current_position, item.point) / 5.0
        if current_channel != record.channel:
            immediate += 1.0
        return 1.0, item.worst_radius, immediate

    return min(finite, key=key)


# ============================ 日志与 HTTP ============================


class RunLog:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self.actions_path = self.directory / "actions.jsonl"
        self.summary_path = self.directory / "summary.json"

    def action(self, item: dict) -> None:
        safe_item = dict(item)
        request_data = dict(safe_item.get("request", {}))
        if "robot_id" in request_data:
            request_data["robot_id"] = "<redacted>"
        safe_item["request"] = request_data
        with self.actions_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(safe_item, ensure_ascii=False, separators=(",", ":")) + "\n")

    def decision(self, event: str, **fields: object) -> None:
        """只记录会影响路径的精简规划决策，不展开大规模候选明细。"""

        self.action(
            {
                "kind": "planner",
                "event": event,
                "utc": datetime.now(timezone.utc).isoformat(),
                **fields,
            }
        )

    def summary(self, item: dict) -> None:
        self.summary_path.write_text(
            json.dumps(item, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class SimulatorClient:
    def __init__(
        self,
        base_url: str,
        robot_id: str,
        mode: str,
        logger: RunLog,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.logger = logger
        self.counter = 0
        self.prefix = f"q3-{mode[0]}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        self.interface_unavailable = False

    def _new_request_id(self, action_name: str) -> str:
        self.counter += 1
        return f"{self.prefix}-{action_name}-{self.counter}"

    def post(self, path: str, fields: dict | None = None) -> dict:
        action_name = path.removeprefix("/")
        request_id = self._new_request_id(action_name)
        payload = {
            "arena_id": ARENA_ID,
            "robot_id": self.robot_id,
            "request_id": request_id,
        }
        if fields:
            payload.update(fields)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        url = self.base_url + path
        started = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(1, HTTP_RETRIES + 1):
            request = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
                    status = int(response.status)
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if status < 200 or status >= 300:
                    raise ActionRejected(f"HTTP {status}: {parsed}")
                if not isinstance(parsed, dict) or parsed.get("accepted") is not True:
                    self.logger.action(
                        {
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "path": path,
                            "request": payload,
                            "attempts": attempt,
                            "http_status": status,
                            "response": parsed,
                            "error": "accepted_not_true",
                            "wall_elapsed_s": round(time.monotonic() - started, 6),
                        }
                    )
                    raise ActionRejected(f"动作未被接受: HTTP {status}, response={parsed}")
                self.logger.action(
                    {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "path": path,
                        "request": payload,
                        "attempts": attempt,
                        "http_status": status,
                        "response": parsed,
                        "wall_elapsed_s": round(time.monotonic() - started, 6),
                    }
                )
                return parsed
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                self.logger.action(
                    {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "path": path,
                        "request": payload,
                        "attempts": attempt,
                        "http_status": int(exc.code),
                        "error": raw,
                    }
                )
                raise ActionRejected(f"HTTP {exc.code}: {raw}") from exc
            except ActionRejected:
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < HTTP_RETRIES:
                    # 网络结果不确定时，用完全相同的 request_id、路径和请求体重试。
                    time.sleep(HTTP_RETRY_DELAYS_S[attempt - 1])

        self.logger.action(
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "path": path,
                "request": payload,
                "attempts": HTTP_RETRIES,
                "error": repr(last_error),
            }
        )
        self.interface_unavailable = True
        raise Q3Error(f"请求 {path} 在保持相同 request_id 的重试后仍失败: {last_error}")


# ============================ 主控制器 ============================


class Q3Controller:
    def __init__(
        self,
        client: SimulatorClient,
        logger: RunLog,
        mode: str,
        case_code: str,
        strategy: str,
    ) -> None:
        self.client = client
        self.logger = logger
        self.mode = mode
        self.case_code = case_code
        self.strategy = strategy
        self.records = {channel: ChannelRecord(channel) for channel in range(1, 21)}
        self.position: Point = (0.0, 0.0)
        self.current_channel = 1
        self.stats = ActionStats()
        self.last_virtual_time_s = 0.0
        self.real_limit_s = 0.0
        self.entered_at: float | None = None
        self.exited = False
        self.exit_response: dict | None = None
        self.exit_reason: str | None = None
        self.outer_route_start: int | None = None
        self.outer_route_direction: int | None = None
        self.end_reason = "not_started"

    def remaining_real_s(self) -> float:
        if self.entered_at is None:
            return math.inf
        return self.real_limit_s - (time.monotonic() - self.entered_at)

    def _before_action(self) -> None:
        if self.stats.action_count >= MAX_TOTAL_ACTIONS:
            raise SafetyStop(f"动作数达到保护上限 {MAX_TOTAL_ACTIONS}，停止继续发送动作。")
        if self.remaining_real_s() <= REAL_TIME_EXIT_RESERVE_S:
            raise SafetyStop("剩余现实时间已进入退出保护区。")

    def enter(self) -> None:
        response = self.client.post("/enter")
        self.entered_at = time.monotonic()
        self.real_limit_s = float(response.get("remaining_real_duration_s", 0.0))
        self.last_virtual_time_s = float(response.get("virtual_time_s", 0.0))
        if self.real_limit_s <= REAL_TIME_EXIT_RESERVE_S:
            raise SafetyStop(f"/enter 后只剩 {self.real_limit_s:.2f}s，不足以安全运行。")
        self.end_reason = "running"
        print(f"已进入：实际可用时间 {self.real_limit_s:.2f}s，模式={self.mode}，案例={self.case_code}")

    def measure(self, point: Point, channel: int) -> dict:
        self._before_action()
        response = self.client.post(
            "/measure",
            {"position": {"x": float(point[0]), "y": float(point[1])}, "channel": int(channel)},
        )
        self.stats.move_distance_m += distance(self.position, point)
        if self.current_channel != channel:
            self.stats.switch_count += 1
        self.stats.measure_count += 1
        self.position = point
        self.current_channel = channel
        self.last_virtual_time_s = float(response["virtual_time_s"])
        return response

    def clear(self, point: Point, channel: int) -> dict:
        self._before_action()
        response = self.client.post(
            "/clear",
            {"position": {"x": float(point[0]), "y": float(point[1])}, "channel": int(channel)},
        )
        self.stats.move_distance_m += distance(self.position, point)
        self.position = point
        self.last_virtual_time_s = float(response["virtual_time_s"])
        if response.get("clear_result") == "success":
            self.stats.clear_success_count += 1
        elif response.get("clear_result") == "no_target_in_range":
            self.stats.clear_failure_count += 1
        else:
            raise Q3Error(f"未知 clear_result: {response}")
        return response

    def exit(self) -> None:
        if self.entered_at is None or self.exited:
            return
        if self.client.interface_unavailable:
            # 附件要求：测试已结束、接口关闭后不要再调用 /exit 查询原因。
            self.exit_reason = "exit_skipped_after_interface_failure"
            return
        try:
            self.exit_response = self.client.post("/exit")
            self.exited = True
            self.exit_reason = str(self.exit_response.get("exit_reason", "user_exit"))
        except Q3Error as exc:
            self.exit_reason = f"exit_failed: {exc}"

    def cleared_count(self) -> int:
        return sum(item.status == ChannelStatus.CLEARED for item in self.records.values())

    def active_records(self) -> list[ChannelRecord]:
        return [item for item in self.records.values() if item.status == ChannelStatus.ACTIVE]

    @property
    def bplus(self) -> bool:
        return self.strategy == "bplus"

    def _handle_direction(self, record: ChannelRecord, point: Point, value: float) -> None:
        old_radius = None
        if record.polygon:
            old_radius = minimum_enclosing_circle(record.polygon).radius
        updated = update_direction_region(record.polygon, point, float(value))
        record.polygon = updated
        record.small_clear_plan_cache = None
        record.observations.append(Observation(point, float(value)))
        record.status = ChannelStatus.ACTIVE
        if old_radius is not None:
            new_radius = minimum_enclosing_circle(updated).radius
            if new_radius >= LOW_GAIN_RATIO * old_radius:
                record.low_gain_streak += 1
            else:
                record.low_gain_streak = 0

    def _handle_near(self, record: ChannelRecord, point: Point) -> None:
        record.status = ChannelStatus.ACTIVE
        response = self.clear(point, record.channel)
        if response.get("clear_result") != "success":
            record.clear_fail_points.append(point)
            raise GeometryError("measure 返回 near 后在同一点清除仍失败，与题面规则矛盾。")
        record.status = ChannelStatus.CLEARED

    def _handle_measure_response(self, record: ChannelRecord, point: Point, response: dict) -> None:
        result = response.get("measure_result")
        if result == "direction":
            if "svd_deg" not in response:
                raise Q3Error("direction 响应缺少 svd_deg。")
            self._handle_direction(record, point, float(response["svd_deg"]))
        elif result == "near":
            self._handle_near(record, point)
        elif result == "no_signal":
            record.no_signal_points.append(point)
            if record.status == ChannelStatus.ACTIVE:
                raise GeometryError("保证可测的已发现频道返回 no_signal，请检查几何或接口状态。")
        else:
            raise Q3Error(f"未知 measure_result: {response}")

    @staticmethod
    def _small_clear_plans_for(record: ChannelRecord) -> tuple[ClearPlan, ...]:
        if not record.polygon:
            return ()
        if record.small_clear_plan_cache is None:
            record.small_clear_plan_cache = tuple(small_clear_plans(record.polygon))
        return record.small_clear_plan_cache

    def _survey_channel_order(self) -> list[int]:
        pending = sorted(
            record.channel
            for record in self.records.values()
            if record.status == ChannelStatus.UNKNOWN
        )
        if self.current_channel in pending:
            pending.remove(self.current_channel)
            pending.insert(0, self.current_channel)
        return pending

    def _clear_guaranteed_at_point(self, point: Point, label: str) -> int:
        """清除所有被当前点的 19.9 m 圆完整覆盖的活动频道。"""

        cleared_here = 0
        candidates: list[ChannelRecord] = []
        for record in self.active_records():
            if not record.polygon:
                continue
            if max(distance(point, vertex) for vertex in record.polygon) > CLEAR_RADIUS_M - CLEAR_NUMERIC_MARGIN_M:
                continue
            candidates.append(record)

        for record in sorted(candidates, key=lambda item: item.channel):
            print(f"  {label}：当前位置保证清除频道 {record.channel}")
            response = self.clear(point, record.channel)
            if response.get("clear_result") != "success":
                record.clear_fail_points.append(point)
                raise GeometryError(
                    f"频道 {record.channel} 的完整定位域已被19.9m圆覆盖，但清除失败。"
                )
            record.status = ChannelStatus.CLEARED
            cleared_here += 1
            if self.bplus:
                self.stats.bplus_opportunity_clear_count += 1
            if self.cleared_count() >= 16:
                break
        return cleared_here

    def _opportunistic_options_at_point(
        self,
        point: Point,
        excluded_channels: set[int] | None = None,
    ) -> list[tuple[int, float, int, ChannelRecord, CandidateScore]]:
        """只生成保证接收且代表情景下有明显收缩的附加检测候选。"""

        excluded = excluded_channels or set()
        options: list[tuple[int, float, int, ChannelRecord, CandidateScore]] = []
        for record in self.active_records():
            if record.channel in excluded or not record.polygon:
                continue
            current_radius = minimum_enclosing_circle(record.polygon).radius
            if current_radius <= CLEAR_RADIUS_M - CLEAR_NUMERIC_MARGIN_M:
                continue
            if not _not_repeated(point, record.observations):
                continue
            if not is_safe_measure_point(record.polygon, point):
                continue
            score = evaluate_candidate(record, point, self.position, self.current_channel)
            ratio = score.worst_radius / max(current_radius, 1.0e-9)
            predicted_one_more = score.predicted_finish_time_s is not None
            if not predicted_one_more and ratio > BPLUS_REQUIRED_RADIUS_RATIO:
                continue
            options.append(
                (
                    0 if predicted_one_more else 1,
                    ratio,
                    0 if self.current_channel == record.channel else 1,
                    record,
                    score,
                )
            )
        return sorted(options, key=lambda item: (item[0], item[1], item[2], item[3].channel))

    def _run_opportunistic_at_point(
        self,
        point: Point,
        label: str,
        limit: int = BPLUS_MAX_EXTRA_CHANNELS_PER_STOP,
        excluded_channels: set[int] | None = None,
    ) -> int:
        options = self._opportunistic_options_at_point(point, excluded_channels)
        measured = 0
        for _, ratio, _, record, _ in options[:limit]:
            if record.status != ChannelStatus.ACTIVE:
                continue
            print(f"  {label}：补测频道 {record.channel}，代表情景半径比 {ratio:.3f}")
            response = self.measure(point, record.channel)
            self._handle_measure_response(record, point, response)
            measured += 1
            self.stats.bplus_extra_measure_count += 1
            if self.cleared_count() >= 16:
                break
        if self.cleared_count() < 16:
            self._clear_guaranteed_at_point(point, label)
        return measured

    def _route_point_value(self, point: Point) -> tuple[int, int, float]:
        clearable = 0
        receivable = 0
        geometry = 0.0
        for record in self.active_records():
            if not record.polygon:
                continue
            if max(distance(point, vertex) for vertex in record.polygon) <= CLEAR_RADIUS_M - CLEAR_NUMERIC_MARGIN_M:
                clearable += 1
                continue
            if not is_safe_measure_point(record.polygon, point) or not _not_repeated(point, record.observations):
                continue
            receivable += 1
            if record.observations:
                previous = record.observations[-1]
                old_direction = (
                    math.cos(math.radians(previous.bearing_deg)),
                    math.sin(math.radians(previous.bearing_deg)),
                )
                center = minimum_enclosing_circle(record.polygon).center
                new_direction = unit(subtract(center, point))
                geometry += abs(cross(old_direction, new_direction))
        return clearable, receivable, geometry

    def _choose_outer_route(self) -> list[Point]:
        if not self.bplus:
            return outer_survey_points()
        ranked: list[tuple[tuple[float, ...], int, int, list[Point]]] = []
        for start, direction, route in outer_survey_routes():
            values = [self._route_point_value(point) for point in route]
            early_clear = sum((6 - index) * value[0] for index, value in enumerate(values))
            early_receive = sum((6 - index) * value[1] for index, value in enumerate(values))
            early_geometry = sum((6 - index) * value[2] for index, value in enumerate(values))
            key = (
                values[0][0],
                values[0][1],
                values[1][0],
                values[1][1],
                early_clear,
                early_receive,
                early_geometry,
                -start,
                1 if direction == 1 else 0,
            )
            ranked.append((key, start, direction, route))
        _, start, direction, route = max(ranked, key=lambda item: item[0])
        self.outer_route_start = start + 1
        self.outer_route_direction = direction
        direction_text = "逆时针" if direction == 1 else "顺时针"
        print(f"B+选择外围起点 P{start + 1}，随后{direction_text}访问；七点骨架长度不变。")
        return route

    def _run_edge_opportunity(self, edge_start: Point, edge_end: Point) -> None:
        """每段必经线最多增加一个内部停靠点，且不改变该段移动距离。"""

        if not self.bplus or not self.active_records():
            return

        clear_intervals: dict[int, tuple[float, float]] = {}
        for record in self.active_records():
            if not record.polygon:
                continue
            interval = segment_cover_center_interval(
                edge_start,
                edge_end,
                record.polygon,
                CLEAR_RADIUS_M - CLEAR_NUMERIC_MARGIN_M,
            )
            if interval is not None:
                clear_intervals[record.channel] = interval

        if clear_intervals:
            parameters = sorted(
                {
                    value
                    for interval in clear_intervals.values()
                    for value in (interval[0], (interval[0] + interval[1]) / 2.0, interval[1])
                }
            )

            def clear_key(parameter: float) -> tuple[int, float]:
                count = sum(
                    interval[0] - 1.0e-12 <= parameter <= interval[1] + 1.0e-12
                    for interval in clear_intervals.values()
                )
                return count, -parameter

            parameter = max(parameters, key=clear_key)
            point = interpolate_segment(edge_start, edge_end, parameter)
            self.stats.bplus_edge_stop_count += 1
            print(f"边上零绕路停靠：({point[0]:.3f}, {point[1]:.3f})，优先保证清除。")
            self._clear_guaranteed_at_point(point, "沿途")
            if self.cleared_count() < 16:
                self._run_opportunistic_at_point(point, "沿途")
            return

        measure_stops: list[tuple[int, float, int, Point]] = []
        for record in self.active_records():
            if not record.polygon:
                continue
            interval = segment_cover_center_interval(
                edge_start,
                edge_end,
                record.polygon,
                MIN_RECEIVE_RADIUS_M - SAFE_MEASURE_NUMERIC_MARGIN_M,
            )
            if interval is None:
                continue
            parameter = (interval[0] + interval[1]) / 2.0
            point = interpolate_segment(edge_start, edge_end, parameter)
            if not _not_repeated(point, record.observations):
                continue
            current_radius = minimum_enclosing_circle(record.polygon).radius
            if current_radius <= CLEAR_RADIUS_M - CLEAR_NUMERIC_MARGIN_M:
                continue
            score = evaluate_candidate(record, point, self.position, self.current_channel)
            ratio = score.worst_radius / max(current_radius, 1.0e-9)
            predicted_one_more = score.predicted_finish_time_s is not None
            if predicted_one_more or ratio <= BPLUS_REQUIRED_RADIUS_RATIO:
                measure_stops.append((0 if predicted_one_more else 1, ratio, record.channel, point))

        if not measure_stops:
            return
        _, _, channel, point = min(measure_stops, key=lambda item: (item[0], item[1], item[2]))
        self.stats.bplus_edge_stop_count += 1
        print(
            f"边上零绕路停靠：({point[0]:.3f}, {point[1]:.3f})，"
            f"由频道 {channel} 的强补测机会触发。"
        )
        self._run_opportunistic_at_point(point, "沿途")

    def _execute_small_clear_plan(
        self,
        record: ChannelRecord,
        plan: ClearPlan,
        label: str,
        end: Point | None = None,
    ) -> None:
        order, worst_time = best_clear_order(self.position, plan, end)
        if len(order) == 1:
            self.stats.single_clear_macro_count += 1
        elif len(order) == 2:
            self.stats.two_clear_macro_count += 1
        else:
            raise GeometryError("小清除宏任务只允许一点或两点。")

        print(
            f"  {label}：频道 {record.channel} 执行 {len(order)} 点保证清除，"
            f"保守完成时间 {worst_time:.3f}s"
        )
        for index, point in enumerate(order):
            response = self.clear(point, record.channel)
            if response.get("clear_result") == "success":
                record.status = ChannelStatus.CLEARED
                self.stats.bplus_opportunity_clear_count += 1
                if len(order) == 2 and index == 0:
                    self.stats.two_clear_first_success_count += 1
                print(f"  频道 {record.channel} 在第 {index + 1} 个保证清除点成功。")
                return
            record.clear_fail_points.append(point)
            if len(order) == 1:
                print("  单点覆盖已核验但清除失败，转入28m有限网格并保留异常记录。")
                self._run_grid_fallback(record)
                return
            if index == 0:
                print("  第一个清除点失败；由两圆联合覆盖证明，第二点现为保证清除点。")
                continue
            raise GeometryError(
                f"频道 {record.channel} 的两点联合覆盖已核验，但第二点仍清除失败。"
            )

    def _nearest_outer_point_for_region(
        self,
        polygon: Sequence[Point],
        outer_route: Sequence[Point],
    ) -> Point:
        distances = {
            point: point_to_polygon_distance(point, polygon)
            for point in outer_survey_points()
        }
        minimum = min(distances.values())
        tied = {
            point
            for point, value in distances.items()
            if value <= minimum + 1.0e-6
        }
        return min(tied, key=lambda point: outer_route.index(point))

    def _try_outside_opportunity(
        self,
        current_point: Point,
        next_point: Point,
        following_point: Point | None,
        outer_route: Sequence[Point],
    ) -> bool:
        """在外侧源的最近外围点比较“现在清除”与“推迟一段”。"""

        if not self.bplus:
            return False
        accepted: list[
            tuple[float, int, int, float, float, ChannelRecord, ClearPlan]
        ] = []
        for record in sorted(self.active_records(), key=lambda item: item.channel):
            if not record.polygon or not polygon_is_strictly_outside_survey_hexagon(record.polygon):
                continue
            nearest = self._nearest_outer_point_for_region(record.polygon, outer_route)
            if distance(current_point, nearest) > 1.0e-6:
                continue
            self.stats.outside_opportunity_count += 1
            plans = self._small_clear_plans_for(record)
            if not plans:
                self.logger.decision(
                    "outside_opportunity",
                    channel=record.channel,
                    current_point=current_point,
                    decision="defer",
                    reason="no_verified_one_or_two_point_cover",
                )
                continue

            best_for_record: tuple[float, int, float, float, ClearPlan] | None = None
            for plan in plans:
                now_order, now_time = best_clear_order(current_point, plan, next_point)
                if following_point is not None:
                    now_time += distance(next_point, following_point) / 5.0
                _, wait_macro_time = best_clear_order(next_point, plan, following_point)
                wait_time = distance(current_point, next_point) / 5.0 + wait_macro_time
                advantage = wait_time - now_time
                ordered_plan = ClearPlan(now_order, plan.proof)
                key = (advantage, -len(plan.points), -now_time)
                if best_for_record is None or key > best_for_record[:3]:
                    best_for_record = (advantage, -len(plan.points), -now_time, wait_time, ordered_plan)

            assert best_for_record is not None
            advantage, negative_count, negative_now, wait_time, ordered_plan = best_for_record
            now_time = -negative_now
            accepted_now = should_insert_outside_macro(now_time, wait_time)
            self.logger.decision(
                "outside_opportunity",
                channel=record.channel,
                current_point=current_point,
                next_point=next_point,
                clear_point_count=len(ordered_plan.points),
                now_time_s=round(now_time, 6),
                wait_time_s=round(wait_time, 6),
                advantage_s=round(advantage, 6),
                decision="insert" if accepted_now else "defer",
                reason="nearest_outer_position_window" if accepted_now else "waiting_saves_at_least_buffer",
            )
            if accepted_now:
                accepted.append(
                    (
                        advantage,
                        -len(ordered_plan.points),
                        -record.channel,
                        -now_time,
                        wait_time,
                        record,
                        ordered_plan,
                    )
                )

        if not accepted:
            return False
        _, _, _, _, wait_time, record, plan = max(accepted, key=lambda item: item[:4])
        move_before = self.stats.move_distance_m
        self.logger.decision(
            "outside_macro_selected",
            channel=record.channel,
            current_point=current_point,
            next_point=next_point,
            clear_points=plan.points,
            clear_point_count=len(plan.points),
            deferred_time_s=round(wait_time, 6),
        )
        self._execute_small_clear_plan(record, plan, "外侧位置机会", next_point)
        actual_macro_distance = self.stats.move_distance_m - move_before
        detour = actual_macro_distance + distance(self.position, next_point) - distance(current_point, next_point)
        self.stats.opportunity_detour_distance_m += max(0.0, detour)
        return True

    def _scan_unknown_channels_at_point(self, point: Point, point_label: str) -> None:
        channels = self._survey_channel_order()
        print(f"搜索点 {point_label}: ({point[0]:.3f}, {point[1]:.3f})，待查频道 {len(channels)} 个")
        for channel in channels:
            record = self.records[channel]
            response = self.measure(point, channel)
            self._handle_measure_response(record, point, response)
            if self.cleared_count() >= 16:
                return

    def run_survey(self) -> None:
        print("开始七点逐频道保证搜索。")
        center = (0.0, 0.0)
        self._scan_unknown_channels_at_point(center, "P0")
        if self.cleared_count() >= 16:
            return

        outer_route = self._choose_outer_route()
        skip_edge_opportunity = False
        for route_index, point in enumerate(outer_route, start=1):
            if self.cleared_count() >= 16:
                return
            if not any(record.status == ChannelStatus.UNKNOWN for record in self.records.values()):
                print("所有频道均已发现或已解决，不再机械访问剩余托底点。")
                break
            if not skip_edge_opportunity:
                self._run_edge_opportunity(self.position, point)
            skip_edge_opportunity = False
            if self.cleared_count() >= 16:
                return
            self._scan_unknown_channels_at_point(point, f"R{route_index}")
            if self.cleared_count() >= 16:
                return
            if self.bplus:
                self._clear_guaranteed_at_point(point, "外围点")
                if self.cleared_count() < 16:
                    self._run_opportunistic_at_point(point, "外围点")
            next_index = route_index
            if (
                self.bplus
                and next_index < len(outer_route)
                and any(record.status == ChannelStatus.UNKNOWN for record in self.records.values())
            ):
                next_point = outer_route[next_index]
                following_point = outer_route[next_index + 1] if next_index + 1 < len(outer_route) else None
                skip_edge_opportunity = self._try_outside_opportunity(
                    point,
                    next_point,
                    following_point,
                    outer_route,
                )
                if self.cleared_count() >= 16:
                    return

        for record in self.records.values():
            if record.status == ChannelStatus.UNKNOWN:
                if len(record.no_signal_points) != len(survey_points()):
                    raise GeometryError(
                        f"频道 {record.channel} 未完成七点排查，不能标记为已排除。"
                    )
                record.status = ChannelStatus.EXCLUDED
        print("七点搜索完成，所有未发现频道均已有逐频道全域排除证据。")

    def _preview_next_action(self, record: ChannelRecord) -> ActionPreview:
        if not record.polygon:
            raise GeometryError(f"频道 {record.channel} 已发现但没有定位区域。")
        plans = self._small_clear_plans_for(record) if self.bplus else ()
        if plans:
            ordered = [
                (*best_clear_order(self.position, plan, None), plan.proof)
                for plan in plans
            ]
            points, _, proof = min(ordered, key=lambda item: (item[1], item[0], item[2]))
            kind = "clear_one" if len(points) == 1 else "clear_two"
            priority = 0 if len(points) == 1 else 1
            return ActionPreview(record.channel, kind, points, priority)

        circle = minimum_enclosing_circle(record.polygon)
        if not self.bplus and circle.radius <= CLEAR_RADIUS_M - CLEAR_NUMERIC_MARGIN_M:
            return ActionPreview(record.channel, "clear_one", (circle.center,), 0)
        if self._should_try_grid(record):
            points = grid_cover_points(
                record.polygon,
                self.position,
                self._record_exclusion_disks(record) if self.bplus else (),
            )
            if not points:
                raise GeometryError("网格保底没有生成清除点。")
            return ActionPreview(record.channel, "grid", (points[0],), 3)
        candidate = select_measure_candidate(record, self.position, self.current_channel)
        if candidate is None:
            points = grid_cover_points(
                record.polygon,
                self.position,
                self._record_exclusion_disks(record) if self.bplus else (),
            )
            if not points:
                raise GeometryError("网格保底没有生成清除点。")
            return ActionPreview(record.channel, "grid", (points[0],), 3)
        return ActionPreview(record.channel, "measure", (candidate.point,), 2)

    def _choose_next_action(self) -> ActionPreview:
        previews = [self._preview_next_action(record) for record in self.active_records()]
        if not previews:
            raise GeometryError("没有可处理的活动频道。")
        route = open_action_order(self.position, previews)
        selected = route[0]
        self.logger.decision(
            "rolling_action_selected",
            current_point=self.position,
            selected_channel=selected.channel,
            selected_kind=selected.kind,
            selected_points=selected.points,
            provisional_route_channels=[item.channel for item in route],
            provisional_route_distance_m=round(action_route_distance(self.position, route), 6),
        )
        return selected

    @staticmethod
    def _record_exclusion_disks(record: ChannelRecord) -> list[Circle]:
        return [
            *(Circle(point, MIN_RECEIVE_RADIUS_M) for point in record.no_signal_points),
            *(Circle(point, CLEAR_RADIUS_M) for point in record.clear_fail_points),
        ]

    def _run_grid_fallback(self, record: ChannelRecord) -> None:
        if not record.polygon:
            raise GeometryError("无定位区域，不能启动网格保底。")
        points = grid_cover_points(
            record.polygon,
            self.position,
            self._record_exclusion_disks(record) if self.bplus else (),
        )
        if not points:
            raise GeometryError("网格保底没有生成清除点。")
        if len(points) > GRID_POINT_WARNING:
            print(
                f"警告：频道 {record.channel} 的保底网格有 {len(points)} 点，"
                f"超过演练关注值 {GRID_POINT_WARNING}，但不会因此放弃该频道。"
            )
        print(f"频道 {record.channel} 启动保底网格，共 {len(points)} 个清除点。")
        for point in points:
            response = self.clear(point, record.channel)
            if response.get("clear_result") == "success":
                record.status = ChannelStatus.CLEARED
                return
            record.clear_fail_points.append(point)
        raise GeometryError(f"频道 {record.channel} 的保守定位域已被网格覆盖，但仍未清除成功。")

    def _should_try_grid(self, record: ChannelRecord) -> bool:
        count = len(record.observations)
        return count >= GRID_AFTER_DIRECTION_COUNT or record.low_gain_streak >= LOW_GAIN_STREAK_TRIGGER

    def process_record(self, record: ChannelRecord) -> None:
        print(f"开始定位清除频道 {record.channel}。")
        while record.status == ChannelStatus.ACTIVE:
            if not record.polygon:
                raise GeometryError(f"频道 {record.channel} 没有定位区域。")
            diameter_value, _ = polygon_diameter(record.polygon)
            circle = minimum_enclosing_circle(record.polygon)
            print(
                f"  频道 {record.channel}: 观测 {len(record.observations)} 次，"
                f"直径 {diameter_value:.3f}m，最小覆盖圆半径 {circle.radius:.3f}m"
            )

            if circle.radius <= CLEAR_RADIUS_M - CLEAR_NUMERIC_MARGIN_M:
                response = self.clear(circle.center, record.channel)
                if response.get("clear_result") == "success":
                    record.status = ChannelStatus.CLEARED
                    print(f"  频道 {record.channel} 清除成功。")
                    if self.bplus and self.cleared_count() < 16:
                        self._run_opportunistic_at_point(
                            circle.center,
                            "保证清除点",
                            excluded_channels={record.channel},
                        )
                    return
                record.clear_fail_points.append(circle.center)
                print("  保证清除点返回失败，转入保守网格并保留异常记录。")
                self._run_grid_fallback(record)
                return

            if self._should_try_grid(record):
                self._run_grid_fallback(record)
                return

            candidate = select_measure_candidate(record, self.position, self.current_channel)
            if candidate is None:
                self._run_grid_fallback(record)
                return
            print(
                f"  补测点 ({candidate.point[0]:.3f}, {candidate.point[1]:.3f})，"
                f"预测最坏半径 {candidate.worst_radius:.3f}m"
            )
            response = self.measure(candidate.point, record.channel)
            self._handle_measure_response(record, candidate.point, response)
            if self.bplus:
                self._clear_guaranteed_at_point(candidate.point, "专项补测点")
                if self.cleared_count() < 16:
                    self._run_opportunistic_at_point(
                        candidate.point,
                        "专项补测点",
                        excluded_channels={record.channel},
                    )

    def process_action(self, preview: ActionPreview) -> None:
        record = self.records[preview.channel]
        if record.status != ChannelStatus.ACTIVE:
            return
        if preview.kind in {"clear_one", "clear_two"}:
            self._execute_small_clear_plan(
                record,
                ClearPlan(preview.points, "rolling_verified_small_cover"),
                "滚动路径",
            )
            return
        if preview.kind == "grid":
            self._run_grid_fallback(record)
            return
        if preview.kind != "measure":
            raise GeometryError(f"未知滚动宏任务类型：{preview.kind}")

        point = preview.points[0]
        candidate = evaluate_candidate(record, point, self.position, self.current_channel)
        print(
            f"  滚动补测频道 {record.channel}：({point[0]:.3f}, {point[1]:.3f})，"
            f"预测最坏半径 {candidate.worst_radius:.3f}m"
        )
        response = self.measure(point, record.channel)
        self._handle_measure_response(record, point, response)
        if self.cleared_count() < 16:
            # 不再连续为其他频道补测；只利用当前位置已经成立的零移动保证清除机会。
            self._clear_guaranteed_at_point(point, "滚动补测点")

    def run_clearance(self) -> None:
        if not self.bplus:
            print("开始基础版逐源定位清除。")
            while self.active_records() and self.cleared_count() < 16:
                choices: list[tuple[float, int, ChannelRecord]] = []
                for record in self.active_records():
                    preview = self._preview_next_action(record)
                    immediate = distance(self.position, preview.points[0]) / 5.0
                    if preview.kind == "measure" and self.current_channel != record.channel:
                        immediate += 1.0
                    choices.append((immediate, record.channel, record))
                self.process_record(min(choices, key=lambda item: (item[0], item[1]))[2])
            return

        print("开始B+逐宏任务滚动定位清除。")
        while self.active_records() and self.cleared_count() < 16:
            self.stats.rolling_replan_count += 1
            preview = self._choose_next_action()
            self.process_action(preview)

    def verify_completion(self) -> None:
        if self.cleared_count() >= 16:
            return
        unresolved = [
            record.channel
            for record in self.records.values()
            if record.status not in (ChannelStatus.CLEARED, ChannelStatus.EXCLUDED)
        ]
        if unresolved:
            raise SafetyStop(f"仍有未解决频道 {unresolved}，禁止宣称任务完成。")

    def summary_data(self, error: str | None = None) -> dict:
        real_duration = None
        if self.entered_at is not None:
            real_duration = time.monotonic() - self.entered_at
        channels: dict[str, dict] = {}
        for channel, record in self.records.items():
            radius = None
            diameter_value = None
            if record.polygon:
                radius = minimum_enclosing_circle(record.polygon).radius
                diameter_value = polygon_diameter(record.polygon)[0]
            channels[str(channel)] = {
                "status": record.status.value,
                "direction_observations": len(record.observations),
                "no_signal_count": len(record.no_signal_points),
                "clear_failure_count": len(record.clear_fail_points),
                "final_diameter_m": diameter_value,
                "final_mec_radius_m": radius,
            }

        cleared = self.cleared_count()
        return {
            "schema_version": 2,
            "mode": self.mode,
            "case_code": self.case_code,
            "algorithm": self.strategy,
            "outer_route_start": self.outer_route_start,
            "outer_route_direction": self.outer_route_direction,
            "formal_unlocked_at_start": FORMAL_UNLOCKED,
            "end_reason": self.end_reason,
            "exit_reason": self.exit_reason,
            "error": error,
            "cleared_count": cleared,
            "actual_source_count": None,
            "clear_fraction": None,
            "virtual_time_s": self.last_virtual_time_s,
            "average_location_clear_time_s": self.last_virtual_time_s / cleared if cleared else None,
            "program_run_duration_s": real_duration,
            "remaining_real_s_at_summary": self.remaining_real_s() if self.entered_at is not None else None,
            "action_stats": asdict(self.stats),
            "status_counts": {
                status.value: sum(record.status == status for record in self.records.values())
                for status in ChannelStatus
            },
            "channels": channels,
        }

    def run(self) -> None:
        error: str | None = None
        try:
            self.enter()
            self.run_survey()
            if self.cleared_count() < 16:
                self.run_clearance()
            self.verify_completion()
            self.end_reason = "completed_by_program"
        except (Q3Error, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.end_reason = "stopped_with_error"
            print(f"程序停止：{error}", file=sys.stderr)
        except Exception as exc:
            # 响应字段异常或未预见的程序错误也必须安全退出并留下摘要。
            error = f"Unexpected {type(exc).__name__}: {exc}"
            self.end_reason = "stopped_with_error"
            print(f"程序异常停止：{error}", file=sys.stderr)
        finally:
            self.exit()
            self.logger.summary(self.summary_data(error))
            print(f"运行记录：{self.logger.directory.resolve()}")
        if error:
            raise SystemExit(2)


# ============================ 命令行与防误操作 ============================


def _validate_robot_id(value: str) -> str:
    encoded = value.encode("utf-8")
    if not 1 <= len(encoded) <= 64:
        raise ValueError("robot_id 的 UTF-8 长度必须为 1 至 64 字节。")
    if any(unicodedata.category(character) in {"Cc", "Cf"} for character in value):
        raise ValueError("robot_id 不能包含控制字符或不可见格式字符。")
    return value


def _print_offline_plan(args: argparse.Namespace) -> None:
    print("当前为离线预览：不会连接模拟器，也不会调用 /enter。")
    print(f"接口地址：{args.base_url}")
    print(f"策略：{'增强版B+' if args.strategy == 'bplus' else '基础版B'}")
    print("七点坐标：")
    for index, point in enumerate(survey_points()):
        print(f"  P{index}=({point[0]:.6f}, {point[1]:.6f})")
    print("真正演练必须显式加入：--connect --mode practice")


def _safety_confirmation(args: argparse.Namespace) -> None:
    if args.mode == "formal":
        if not FORMAL_UNLOCKED:
            raise SafetyStop(
                "正式模式仍被源码中的 FORMAL_UNLOCKED=False 硬锁。"
                "演练阶段不要修改该值。"
            )
        if args.confirm_formal != FORMAL_CONFIRM_TEXT:
            raise SafetyStop("正式模式缺少正确的 --confirm-formal 确认词。")
        print("警告：你正在准备连接问题3正式测试，正式机会一旦在模拟器中启动即会消耗。")
        typed = input(f"确认模拟器当前确为问题3正式测试后，输入 {FORMAL_CONFIRM_TEXT}：").strip()
        if typed != FORMAL_CONFIRM_TEXT:
            raise SafetyStop("正式模式人工确认失败，未连接模拟器。")
    else:
        print("仅允许连接『问题3演练测试』。程序无法从接口自动识别模拟器当前页面。")
        print("请肉眼确认：不是问题3正式测试，也不是问题4测试，并且界面已显示接口就绪。")
        typed = input(f"确认后输入 {PRACTICE_CONFIRM_TEXT}：").strip()
        if typed != PRACTICE_CONFIRM_TEXT:
            raise SafetyStop("演练模式人工确认失败，未连接模拟器。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三自动搜索定位清除程序（默认离线）")
    parser.add_argument("--connect", action="store_true", help="显式允许连接本机模拟器")
    parser.add_argument("--mode", choices=("practice", "formal"), help="连接时必须明确测试模式")
    parser.add_argument("--robot-id", default=os.environ.get("JAMMERS_ROBOT_ID"), help="参赛队号；也可用 JAMMERS_ROBOT_ID 环境变量")
    parser.add_argument("--case-code", help="模拟器界面显示的案例编码，仅用于本地日志关联")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="模拟器接口地址")
    parser.add_argument(
        "--strategy",
        choices=("baseline", "bplus"),
        default="bplus",
        help="baseline=原基础版B；bplus=当前增强版B+（默认）",
    )
    parser.add_argument("--confirm-formal", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.connect:
        _print_offline_plan(args)
        return
    if args.mode is None:
        raise SystemExit("连接模拟器时必须明确写 --mode practice 或 --mode formal。")

    robot_id = args.robot_id or input("请输入当前登录模拟器的参赛队号 robot_id：").strip()
    case_code = args.case_code or input("请输入模拟器界面显示的本局案例编码：").strip()
    try:
        robot_id = _validate_robot_id(robot_id)
        if not case_code:
            raise ValueError("案例编码不能为空。")
        _safety_confirmation(args)
    except (ValueError, SafetyStop) as exc:
        raise SystemExit(f"安全检查未通过：{exc}") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_case = "".join(character if character.isalnum() or character in "-_" else "_" for character in case_code)
    log_directory = (
        Path(__file__).resolve().parent
        / "runs"
        / f"{timestamp}_{args.mode}_{args.strategy}_{safe_case}"
    )
    logger = RunLog(log_directory)
    client = SimulatorClient(args.base_url, robot_id, args.mode, logger)
    controller = Q3Controller(client, logger, args.mode, case_code, args.strategy)
    controller.run()


if __name__ == "__main__":
    main()
