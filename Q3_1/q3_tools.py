#!/usr/bin/env python3
"""问题三离线自检与演练结果汇总工具；本文件绝不连接模拟器。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import q3_runner as q3


def _assert_close(actual: float, expected: float, tolerance: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{name}: actual={actual}, expected={expected}")


def test_seven_point_coverage() -> None:
    points = q3.survey_points()
    worst_distance = 0.0
    worst_point = (0.0, 0.0)
    for radius_index in range(181):
        radius = 10.0 * radius_index
        for angle_index in range(720):
            angle = 2.0 * math.pi * angle_index / 720.0
            target = radius * math.cos(angle), radius * math.sin(angle)
            nearest = min(q3.distance(target, station) for station in points)
            if nearest > worst_distance:
                worst_distance = nearest
                worst_point = target
    if worst_distance > q3.MIN_RECEIVE_RADIUS_M + 1.0e-8:
        raise AssertionError(f"七点覆盖失败：最坏距离 {worst_distance}, point={worst_point}")

    theoretical = math.sqrt(
        q3.TARGET_RADIUS_M**2
        + q3.SURVEY_RING_RADIUS_M**2
        - 2.0
        * q3.TARGET_RADIUS_M
        * q3.SURVEY_RING_RADIUS_M
        * math.cos(math.pi / 6.0)
    )
    _assert_close(theoretical, 988.511420436013, 1.0e-6, "七点边界最坏距离")


def test_equal_length_outer_routes() -> None:
    routes = q3.outer_survey_routes()
    if len(routes) != 12:
        raise AssertionError(f"外围等长路线应为12条，实际为 {len(routes)} 条。")
    expected = {
        (round(point[0], 8), round(point[1], 8))
        for point in q3.outer_survey_points()
    }
    for _, _, route in routes:
        actual = {(round(point[0], 8), round(point[1], 8)) for point in route}
        if actual != expected:
            raise AssertionError("动态外围路线没有恰好访问六个不同外围点。")
        length = q3.distance((0.0, 0.0), route[0]) + sum(
            q3.distance(route[index], route[index + 1])
            for index in range(5)
        )
        _assert_close(length, 6900.0, 1.0e-7, "动态外围路线长度")


def test_outer_polygon_and_wedge() -> None:
    outer = q3.initial_target_polygon()
    for angle_index in range(1440):
        angle = 2.0 * math.pi * angle_index / 1440.0
        boundary = q3.TARGET_RADIUS_M * math.cos(angle), q3.TARGET_RADIUS_M * math.sin(angle)
        if not q3.point_in_polygon(boundary, outer):
            raise AssertionError(f"目标圆外包遗漏边界点：{boundary}")

    wedge = q3.clip_bearing_wedge(outer, (0.0, 0.0), 0.0)
    if not q3.point_in_polygon((1000.0, 0.0), wedge):
        raise AssertionError("0° 测向角域没有包含正东方向。")
    if not q3.point_in_polygon((1000.0 * math.cos(math.radians(0.9)), 1000.0 * math.sin(math.radians(0.9))), wedge):
        raise AssertionError("测向角域没有包含 +0.9° 边界内点。")
    if q3.point_in_polygon((-1000.0, 0.0), wedge):
        raise AssertionError("测向角域错误地包含了反方向。")

    wrapped = q3.clip_bearing_wedge(outer, (0.0, 0.0), 359.5)
    target = 1000.0 * math.cos(math.radians(0.2)), 1000.0 * math.sin(math.radians(0.2))
    if not q3.point_in_polygon(target, wrapped):
        raise AssertionError("跨 0° 的测向角域处理失败。")


def test_minimum_enclosing_circle() -> None:
    square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    circle = q3.minimum_enclosing_circle(square)
    _assert_close(circle.center[0], 0.0, 1.0e-9, "正方形圆心 x")
    _assert_close(circle.center[1], 0.0, 1.0e-9, "正方形圆心 y")
    _assert_close(circle.radius, math.sqrt(2.0), 1.0e-9, "正方形最小覆盖圆半径")

    root3 = math.sqrt(3.0)
    triangle = [(-20.0, -20.0 / root3), (20.0, -20.0 / root3), (0.0, 40.0 / root3)]
    diameter_value, _ = q3.polygon_diameter(triangle)
    triangle_circle = q3.minimum_enclosing_circle(triangle)
    _assert_close(diameter_value, 40.0, 1.0e-8, "等边三角形直径")
    _assert_close(triangle_circle.radius, 40.0 / root3, 1.0e-8, "等边三角形覆盖半径")
    if triangle_circle.radius <= q3.CLEAR_RADIUS_M:
        raise AssertionError("直径 40m 的等边三角形不应被判为一次可保证清除。")


def test_segment_cover_interval() -> None:
    polygon = [(490.0, -5.0), (510.0, -5.0), (510.0, 5.0), (490.0, 5.0)]
    interval = q3.segment_cover_center_interval(
        (0.0, 0.0),
        (1000.0, 0.0),
        polygon,
        q3.CLEAR_RADIUS_M - q3.CLEAR_NUMERIC_MARGIN_M,
    )
    if interval is None:
        raise AssertionError("路线穿过保证清除中心域，但没有生成沿途区间。")
    point = q3.interpolate_segment((0.0, 0.0), (1000.0, 0.0), sum(interval) / 2.0)
    if max(q3.distance(point, vertex) for vertex in polygon) > q3.CLEAR_RADIUS_M - q3.CLEAR_NUMERIC_MARGIN_M + 1.0e-8:
        raise AssertionError("沿途区间中点不能覆盖完整定位区域。")
    missed = q3.segment_cover_center_interval(
        (0.0, 100.0),
        (1000.0, 100.0),
        polygon,
        q3.CLEAR_RADIUS_M - q3.CLEAR_NUMERIC_MARGIN_M,
    )
    if missed is not None:
        raise AssertionError("远离定位区域的路线不应产生保证清除区间。")


def test_region_update_and_candidates() -> None:
    true_source = (1200.0, 100.0)
    first_station = (0.0, 0.0)
    first_bearing = q3.bearing_deg(first_station, true_source)
    polygon = q3.update_direction_region(None, first_station, first_bearing)
    if not q3.point_in_polygon(true_source, polygon):
        raise AssertionError("第一次定位外包没有包含真实目标。")

    record = q3.ChannelRecord(
        channel=3,
        status=q3.ChannelStatus.ACTIVE,
        observations=[q3.Observation(first_station, first_bearing)],
        polygon=polygon,
    )
    candidates = q3.generate_measure_candidates(record, first_station)
    if not candidates:
        raise AssertionError("第一次测向后没有生成保证可测候选点。")
    if not all(q3.is_safe_measure_point(polygon, point) for point in candidates):
        raise AssertionError("候选点中混入了不能保证接收的点。")

    second_station = candidates[0]
    second_bearing = q3.bearing_deg(second_station, true_source)
    updated = q3.update_direction_region(polygon, second_station, second_bearing)
    if not q3.point_in_polygon(true_source, updated):
        raise AssertionError("第二次定位外包没有包含真实目标。")
    if q3.minimum_enclosing_circle(updated).radius > q3.minimum_enclosing_circle(polygon).radius + 1.0e-7:
        raise AssertionError("加入新约束后保守定位域不应变大。")


def test_grid_cover() -> None:
    polygon = [(-70.0, -15.0), (55.0, -25.0), (90.0, 30.0), (-35.0, 65.0)]
    centers = q3.grid_cover_points(polygon, (0.0, 0.0))
    if not centers:
        raise AssertionError("网格覆盖点为空。")
    bound = q3.GRID_CELL_M * math.sqrt(2.0) / 2.0 + 1.0e-7

    # 对边界和由顶点凸组合得到的内部点作交叉检查。
    samples: list[q3.Point] = []
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        for step in range(21):
            ratio = step / 20.0
            samples.append(
                (
                    first[0] + ratio * (second[0] - first[0]),
                    first[1] + ratio * (second[1] - first[1]),
                )
            )
    center = (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )
    for vertex in polygon:
        for step in range(21):
            ratio = step / 20.0
            samples.append(
                (
                    center[0] + ratio * (vertex[0] - center[0]),
                    center[1] + ratio * (vertex[1] - center[1]),
                )
            )
    for sample in samples:
        nearest = min(q3.distance(sample, grid_center) for grid_center in centers)
        if nearest > bound:
            raise AssertionError(f"网格覆盖交叉检查失败：point={sample}, distance={nearest}")


def test_exclusion_aware_grid() -> None:
    polygon = [(-84.0, -84.0), (84.0, -84.0), (84.0, 84.0), (-84.0, 84.0)]
    plain = q3.grid_cover_points(polygon, (0.0, 0.0))
    exclusion = q3.Circle((0.0, 0.0), 60.0)
    reduced = q3.grid_cover_points(polygon, (0.0, 0.0), [exclusion])
    if len(reduced) >= len(plain):
        raise AssertionError("可靠排除圆没有减少任何保底方格。")

    bound = q3.GRID_CELL_M * math.sqrt(2.0) / 2.0 + 1.0e-7
    for x in range(-84, 85, 7):
        for y in range(-84, 85, 7):
            sample = float(x), float(y)
            if q3.distance(sample, exclusion.center) <= exclusion.radius:
                continue
            nearest = min(q3.distance(sample, center) for center in reduced)
            if nearest > bound:
                raise AssertionError(
                    f"排除圆外的可能点被网格遗漏：point={sample}, distance={nearest}"
                )


def test_outside_hexagon_classification() -> None:
    inside = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
    outside = [(1500.0, -10.0), (1520.0, -10.0), (1520.0, 10.0), (1500.0, 10.0)]
    crossing = [(1130.0, -10.0), (1170.0, -10.0), (1170.0, 10.0), (1130.0, 10.0)]
    touching = [(1150.0, -5.0), (1170.0, -5.0), (1170.0, 5.0), (1150.0, 5.0)]
    if q3.polygon_is_strictly_outside_survey_hexagon(inside):
        raise AssertionError("六边形内区域被误标为外侧。")
    if not q3.polygon_is_strictly_outside_survey_hexagon(outside):
        raise AssertionError("与六边形分离的区域未标为外侧。")
    if q3.polygon_is_strictly_outside_survey_hexagon(crossing):
        raise AssertionError("穿越六边形边界的区域被误标为外侧。")
    if q3.polygon_is_strictly_outside_survey_hexagon(touching):
        raise AssertionError("接触六边形边界的区域被误标为严格外侧。")


def test_small_clear_plans() -> None:
    one = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
    one_plans = q3.small_clear_plans(one)
    if len(one_plans) != 1 or len(one_plans[0].points) != 1:
        raise AssertionError("单圆可覆盖区域没有生成单点方案。")

    two = [(-30.0, -5.0), (30.0, -5.0), (30.0, 5.0), (-30.0, 5.0)]
    two_plans = q3.small_clear_plans(two)
    if not two_plans or any(len(plan.points) != 2 for plan in two_plans):
        raise AssertionError("约60m窄区域没有生成经分割证明的两点方案。")

    too_long = [(-50.0, -5.0), (50.0, -5.0), (50.0, 5.0), (-50.0, 5.0)]
    if q3.small_clear_plans(too_long):
        raise AssertionError("两个19.9m圆不能按当前充分算法覆盖的长区域被误判。")


def test_clear_sequence_time_and_order() -> None:
    start = (0.0, 0.0)
    end = (100.0, 0.0)
    first = (20.0, 0.0)
    second = (60.0, 0.0)
    forward = q3.clear_sequence_worst_time_s(start, (first, second), end)
    reverse = q3.clear_sequence_worst_time_s(start, (second, first), end)
    _assert_close(forward, 28.0, 1.0e-9, "两点清除顺序最坏时间")
    if forward >= reverse:
        raise AssertionError("面向终点的两点清除顺序没有被正确优选。")
    order, value = q3.best_clear_order(start, q3.ClearPlan((first, second), "test"), end)
    if order != (first, second):
        raise AssertionError("两点保证清除未选择最小最坏时间顺序。")
    _assert_close(value, forward, 1.0e-9, "两点清除最优顺序时间")


def test_outside_insertion_buffer() -> None:
    if not q3.should_insert_outside_macro(99.0, 100.0, 6.0):
        raise AssertionError("现在处理更快时应插入外侧宏任务。")
    if not q3.should_insert_outside_macro(105.999, 100.0, 6.0):
        raise AssertionError("多耗不足6s的最近点位置机会应进入灰色插入。")
    if q3.should_insert_outside_macro(106.0, 100.0, 6.0):
        raise AssertionError("现在处理至少多耗6s时应推迟。")


def test_two_clear_execution_branches() -> None:
    class FakeLogger:
        def decision(self, event: str, **fields: object) -> None:
            del event, fields

    class FakeClient:
        interface_unavailable = False

        def __init__(self, results: list[str]) -> None:
            self.results = list(results)
            self.calls = 0

        def post(self, path: str, fields: dict | None = None) -> dict:
            del fields
            if path != "/clear":
                raise AssertionError(f"两点清除分支出现非清除动作：{path}")
            self.calls += 1
            return {
                "accepted": True,
                "clear_result": self.results.pop(0),
                "virtual_time_s": float(self.calls * 5),
            }

    polygon = [(-30.0, -5.0), (30.0, -5.0), (30.0, 5.0), (-30.0, 5.0)]
    plan = q3.small_clear_plans(polygon)[0]

    first_client = FakeClient(["success"])
    first_controller = q3.Q3Controller(first_client, FakeLogger(), "practice", "offline", "bplus")
    first_record = q3.ChannelRecord(channel=1, status=q3.ChannelStatus.ACTIVE, polygon=list(polygon))
    first_controller.records[1] = first_record
    first_controller._execute_small_clear_plan(first_record, plan, "offline-test")
    if first_client.calls != 1 or first_controller.stats.two_clear_first_success_count != 1:
        raise AssertionError("两点方案首点成功后没有取消第二点。")

    second_client = FakeClient(["no_target_in_range", "success"])
    second_controller = q3.Q3Controller(second_client, FakeLogger(), "practice", "offline", "bplus")
    second_record = q3.ChannelRecord(channel=1, status=q3.ChannelStatus.ACTIVE, polygon=list(polygon))
    second_controller.records[1] = second_record
    second_controller._execute_small_clear_plan(second_record, plan, "offline-test")
    if second_client.calls != 2 or second_record.status != q3.ChannelStatus.CLEARED:
        raise AssertionError("两点方案首点失败后没有在第二保证点完成清除。")


def test_outside_opportunity_execution() -> None:
    class FakeLogger:
        def __init__(self) -> None:
            self.events: list[str] = []

        def decision(self, event: str, **fields: object) -> None:
            del fields
            self.events.append(event)

    class FakeClient:
        interface_unavailable = False

        def post(self, path: str, fields: dict | None = None) -> dict:
            del fields
            if path != "/clear":
                raise AssertionError(f"外侧源机会测试出现非清除动作：{path}")
            return {"accepted": True, "clear_result": "success", "virtual_time_s": 5.0}

    route = q3.outer_survey_points()
    current, next_point, following = route[0], route[1], route[2]
    polygon = [(1470.0, -5.0), (1530.0, -5.0), (1530.0, 5.0), (1470.0, 5.0)]
    logger = FakeLogger()
    controller = q3.Q3Controller(FakeClient(), logger, "practice", "offline", "bplus")
    controller.position = current
    record = q3.ChannelRecord(channel=1, status=q3.ChannelStatus.ACTIVE, polygon=polygon)
    controller.records[1] = record
    executed = controller._try_outside_opportunity(current, next_point, following, route)
    if not executed or record.status != q3.ChannelStatus.CLEARED:
        raise AssertionError("外侧源在最近保底点的低成本清除机会未执行。")
    if "outside_macro_selected" not in logger.events:
        raise AssertionError("外侧源宏任务没有保存精简决策日志。")


def test_open_action_route() -> None:
    actions = [
        q3.ActionPreview(3, "measure", ((100.0, 100.0),), 2),
        q3.ActionPreview(1, "clear_one", ((20.0, 0.0),), 0),
        q3.ActionPreview(2, "clear_two", ((50.0, 0.0), (70.0, 0.0)), 1),
    ]
    first = q3.open_action_order((0.0, 0.0), actions)
    second = q3.open_action_order((0.0, 0.0), actions)
    if first != second:
        raise AssertionError("最近插入和2-opt路线不可复现。")
    if q3.action_route_distance((0.0, 0.0), first) > q3.action_route_distance((0.0, 0.0), actions) + 1.0e-8:
        raise AssertionError("滚动开放路线比输入顺序更长。")


def test_single_measure_action_returns() -> None:
    class FakeLogger:
        def decision(self, event: str, **fields: object) -> None:
            del event, fields

    class FakeClient:
        interface_unavailable = False

        def __init__(self) -> None:
            self.calls: list[str] = []

        def post(self, path: str, fields: dict | None = None) -> dict:
            self.calls.append(path)
            if path != "/measure":
                raise AssertionError(f"单次补测测试出现了额外接口动作：{path}")
            assert fields is not None
            point = float(fields["position"]["x"]), float(fields["position"]["y"])
            return {
                "accepted": True,
                "measure_result": "direction",
                "svd_deg": q3.bearing_deg(point, (30.0, 0.0)),
                "virtual_time_s": 5.0,
            }

    client = FakeClient()
    controller = q3.Q3Controller(client, FakeLogger(), "practice", "offline-test", "bplus")
    polygon = [(-50.0, -5.0), (50.0, -5.0), (50.0, 5.0), (-50.0, 5.0)]
    controller.records[1] = q3.ChannelRecord(
        channel=1,
        status=q3.ChannelStatus.ACTIVE,
        polygon=polygon,
    )
    point = (0.0, -100.0)
    controller.process_action(q3.ActionPreview(1, "measure", (point,), 2))
    if client.calls != ["/measure"]:
        raise AssertionError("普通补测宏任务没有在一次测量后返回。")


def run_self_tests() -> None:
    tests = [
        test_seven_point_coverage,
        test_equal_length_outer_routes,
        test_outer_polygon_and_wedge,
        test_minimum_enclosing_circle,
        test_segment_cover_interval,
        test_region_update_and_candidates,
        test_grid_cover,
        test_exclusion_aware_grid,
        test_outside_hexagon_classification,
        test_small_clear_plans,
        test_clear_sequence_time_and_order,
        test_outside_insertion_buffer,
        test_two_clear_execution_branches,
        test_outside_opportunity_execution,
        test_open_action_route,
        test_single_measure_action_returns,
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print("全部离线自检通过。本命令没有连接模拟器。")


def _summary_path(raw: Path) -> Path:
    return raw if raw.name == "summary.json" else raw / "summary.json"


def add_truth(run_path: Path, source_count: int) -> None:
    path = _summary_path(run_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not 10 <= source_count <= 16:
        raise ValueError("演练公布的真实干扰源总数必须在 10 至 16 之间。")
    cleared = int(data.get("cleared_count", 0))
    if cleared > source_count:
        raise ValueError("清除数大于真实源数，请检查案例编码或运行记录。")
    data["actual_source_count"] = source_count
    data["clear_fraction"] = cleared / source_count
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已更新 {path}")
    print(f"清除比例：{cleared}/{source_count}={cleared/source_count:.6f}")


def summarize(runs_directory: Path, output: Path) -> None:
    summaries = sorted(runs_directory.glob("*/summary.json"))
    fields = [
        "run_directory",
        "mode",
        "case_code",
        "algorithm",
        "outer_route_start",
        "outer_route_direction",
        "end_reason",
        "exit_reason",
        "error",
        "actual_source_count",
        "cleared_count",
        "clear_fraction",
        "virtual_time_s",
        "average_location_clear_time_s",
        "program_run_duration_s",
        "move_distance_m",
        "measure_count",
        "switch_count",
        "clear_success_count",
        "clear_failure_count",
        "bplus_edge_stop_count",
        "bplus_extra_measure_count",
        "bplus_opportunity_clear_count",
        "outside_opportunity_count",
        "single_clear_macro_count",
        "two_clear_macro_count",
        "two_clear_first_success_count",
        "opportunity_detour_distance_m",
        "rolling_replan_count",
    ]
    rows: list[dict] = []
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        stats = data.get("action_stats", {})
        rows.append(
            {
                "run_directory": path.parent.name,
                "mode": data.get("mode"),
                "case_code": data.get("case_code"),
                "algorithm": data.get("algorithm"),
                "outer_route_start": data.get("outer_route_start"),
                "outer_route_direction": data.get("outer_route_direction"),
                "end_reason": data.get("end_reason"),
                "exit_reason": data.get("exit_reason"),
                "error": data.get("error"),
                "actual_source_count": data.get("actual_source_count"),
                "cleared_count": data.get("cleared_count"),
                "clear_fraction": data.get("clear_fraction"),
                "virtual_time_s": data.get("virtual_time_s"),
                "average_location_clear_time_s": data.get("average_location_clear_time_s"),
                "program_run_duration_s": data.get("program_run_duration_s"),
                "move_distance_m": stats.get("move_distance_m"),
                "measure_count": stats.get("measure_count"),
                "switch_count": stats.get("switch_count"),
                "clear_success_count": stats.get("clear_success_count"),
                "clear_failure_count": stats.get("clear_failure_count"),
                "bplus_edge_stop_count": stats.get("bplus_edge_stop_count"),
                "bplus_extra_measure_count": stats.get("bplus_extra_measure_count"),
                "bplus_opportunity_clear_count": stats.get("bplus_opportunity_clear_count"),
                "outside_opportunity_count": stats.get("outside_opportunity_count"),
                "single_clear_macro_count": stats.get("single_clear_macro_count"),
                "two_clear_macro_count": stats.get("two_clear_macro_count"),
                "two_clear_first_success_count": stats.get("two_clear_first_success_count"),
                "opportunity_detour_distance_m": stats.get("opportunity_detour_distance_m"),
                "rolling_replan_count": stats.get("rolling_replan_count"),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已汇总 {len(rows)} 次运行：{output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三离线自检与演练汇总工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="运行离线几何自检，绝不连接模拟器")

    truth = subparsers.add_parser("add-truth", help="把演练结束后公布的真实源数写入摘要")
    truth.add_argument("run_path", type=Path, help="某次运行目录或其中的 summary.json")
    truth.add_argument("source_count", type=int, help="模拟器演练结束后公布的真实源数")

    summary = subparsers.add_parser("summarize", help="汇总 runs 下的全部摘要")
    summary.add_argument("--runs", type=Path, default=Path(__file__).resolve().parent / "runs")
    summary.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-test":
        run_self_tests()
    elif args.command == "add-truth":
        add_truth(args.run_path, args.source_count)
    elif args.command == "summarize":
        summarize(args.runs, args.output)


if __name__ == "__main__":
    main()
