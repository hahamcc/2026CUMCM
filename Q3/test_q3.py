from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from Q3.client import SimulatorClient
from Q3.config import Q3Config
from Q3.geometry import (
    ConservativeGrid,
    certified_responsibility_zones,
    minimum_enclosing_circle,
    responsibility_zone_certificate,
    search_coverage_certificate,
    search_points,
)
from Q3.local_simulator import LocalSimulatorClient, Source
from Q3.models import Action, ActionKind, ChannelPhase, ChannelState, DirectionEvidence, RobotState, StrategyMode
from Q3.run import run_strategy
from Q3.strategy import Q3Strategy, new_channel_states
from Q3.planner import ordered_channels, sequence_switch_count, weighted_open_search_order


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Q3Config()
        cls.grid = ConservativeGrid(cls.config)

    def test_search_coverage_certificate(self) -> None:
        certificate = search_coverage_certificate(self.config)
        self.assertTrue(certificate["covered"])
        self.assertAlmostEqual(certificate["outer_endpoint_distance_m"], 988.5, delta=0.1)
        self.assertAlmostEqual(certificate["safety_margin_m"], 11.5, delta=0.1)

    def test_dense_search_coverage(self) -> None:
        stations = search_points(self.config)
        worst = 0.0
        for radius in np.linspace(0.0, self.config.target_radius_m, 181):
            for angle in np.linspace(-math.pi, math.pi, 721):
                point = (radius * math.cos(angle), radius * math.sin(angle))
                worst = max(worst, min(math.dist(point, station) for station in stations))
        self.assertLessEqual(worst, self.config.guaranteed_receive_radius_m)

    def test_fixed_points_certify_exactly_their_responsibility_zones(self) -> None:
        for index, point in enumerate(search_points(self.config)):
            self.assertEqual(certified_responsibility_zones(point, self.config), {index})
            certificate = responsibility_zone_certificate(index, point, self.config)
            self.assertTrue(certificate["covered"])
        outer = responsibility_zone_certificate(1, search_points(self.config)[1], self.config)
        self.assertAlmostEqual(float(outer["maximum_distance_m"]), 988.5114, places=3)

    def test_sparse_interior_checks_cannot_replace_zone_boundary_certificate(self) -> None:
        station = (1110.0, 0.0)
        for sample in ((1200.0, 0.0), (1500.0, 0.0)):
            self.assertLess(math.dist(station, sample), self.config.guaranteed_receive_radius_m)
        certificate = responsibility_zone_certificate(1, station, self.config)
        self.assertFalse(certificate["covered"])
        self.assertGreater(float(certificate["maximum_distance_m"]), 1000.0)

    def test_one_disk_cannot_certify_two_complete_zones(self) -> None:
        for x in np.linspace(-1800.0, 1800.0, 25):
            for y in np.linspace(-1800.0, 1800.0, 25):
                self.assertLessEqual(len(certified_responsibility_zones((float(x), float(y)), self.config)), 1)

    def test_dependency_free_minimum_enclosing_circle(self) -> None:
        points = np.asarray([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)] * 40)
        circle = minimum_enclosing_circle(points)
        self.assertAlmostEqual(circle.center[0], 1.0, places=9)
        self.assertAlmostEqual(circle.center[1], 1.0, places=9)
        self.assertAlmostEqual(circle.radius, math.sqrt(2.0), places=9)

    def _cell_index(self, point: tuple[float, float]) -> tuple[int, int]:
        col = int(math.floor((point[0] + self.config.target_radius_m) / self.config.grid_step_m))
        row = int(math.floor((point[1] + self.config.target_radius_m) / self.config.grid_step_m))
        return row, col

    def test_direction_never_deletes_compatible_truth_cell(self) -> None:
        source = (1400.0, 10.0)
        station = (0.0, 0.0)
        true_bearing = math.degrees(math.atan2(source[1], source[0]))
        for error in (-1.0, 0.0, 1.0):
            channel = ChannelState(1)
            update = self.grid.update_direction(channel, DirectionEvidence(station, true_bearing + error))
            self.assertTrue(update.accepted)
            self.assertTrue(channel.region_mask[self._cell_index(source)])

    def test_no_signal_and_clear_failure_keep_compatible_truth(self) -> None:
        source = (1100.0, 0.0)
        channel = ChannelState(1, region_mask=self.grid.fresh_region())
        self.grid.update_no_signal(channel, (0.0, 0.0))
        self.assertTrue(channel.region_mask[self._cell_index(source)])
        self.grid.update_clear_failure(channel, (1000.0, 0.0))
        self.assertTrue(channel.region_mask[self._cell_index(source)])

    def test_measurement_distance_boundaries(self) -> None:
        for distance in (5.000001, 1000.0, 1500.0):
            source = (distance, 0.0)
            channel = ChannelState(1)
            update = self.grid.update_direction(channel, DirectionEvidence((0.0, 0.0), 0.0))
            self.assertTrue(update.accepted)
            self.assertTrue(channel.region_mask[self._cell_index(source)])

    def test_empty_update_is_rejected(self) -> None:
        mask = np.zeros(self.grid.shape, dtype=bool)
        row, col = self._cell_index((100.0, 0.0))
        mask[row, col] = True
        channel = ChannelState(1, region_mask=mask)
        before = channel.region_mask.copy()
        update = self.grid.update_direction(channel, DirectionEvidence((0.0, 0.0), 180.0))
        self.assertFalse(update.accepted)
        self.assertTrue(np.array_equal(channel.region_mask, before))

    def test_clear_plan_certificate(self) -> None:
        mask = np.zeros(self.grid.shape, dtype=bool)
        center_row, center_col = self._cell_index((100.0, 100.0))
        mask[center_row - 5 : center_row + 6, center_col - 5 : center_col + 6] = True
        plan = self.grid.build_clear_plan(mask, (0.0, 0.0))
        self.assertTrue(plan.certificate["all_retained_cells_fully_covered"])
        self.assertGreaterEqual(len(plan.points), 1)


class ProtocolTimeTests(unittest.TestCase):
    def test_attachment_time_example(self) -> None:
        config = Q3Config()
        client = LocalSimulatorClient([], config)
        actions = [
            Action(ActionKind.ENTER, "enter"),
            Action(ActionKind.MEASURE, "m1", (300.0, 400.0), 1),
            Action(ActionKind.MEASURE, "m2", (300.0, 400.0), 2),
            Action(ActionKind.CLEAR, "c1", (300.0, 0.0), 3),
            Action(ActionKind.MEASURE, "m3", (300.0, 0.0), 2),
            Action(ActionKind.EXIT, "exit"),
        ]
        responses = [client.execute(action) for action in actions]
        self.assertEqual(responses[-1].body["virtual_time_s"], 199.0)
        self.assertEqual(client.current_channel, 2)

    def test_idempotent_retry_does_not_repeat(self) -> None:
        client = LocalSimulatorClient([])
        client.execute(Action(ActionKind.ENTER, "enter"))
        action = Action(ActionKind.MEASURE, "same", (300.0, 400.0), 1)
        first = client.execute(action)
        second = client.execute(action)
        self.assertEqual(first.body, second.body)
        self.assertEqual(client.virtual_time_s, 105.0)

    def test_uncertain_http_measurement_is_not_blindly_retried(self) -> None:
        client = SimulatorClient("http://127.0.0.1:2026", "test", Q3Config())
        action = Action(ActionKind.MEASURE, "uncertain", (0.0, 0.0), 1)
        with patch("Q3.client.urlopen", side_effect=TimeoutError("timeout")) as mocked:
            response = client.execute(action)
        self.assertIsNone(response.body)
        self.assertIn("uncertain remote state", response.network_error)
        self.assertEqual(mocked.call_count, 1)

    def test_clear_twenty_meter_boundary_and_repeat(self) -> None:
        client = LocalSimulatorClient([Source(1, (20.0, 0.0), 1000.0)])
        client.execute(Action(ActionKind.ENTER, "enter"))
        first = client.execute(Action(ActionKind.CLEAR, "c1", (0.0, 0.0), 1))
        second = client.execute(Action(ActionKind.CLEAR, "c2", (0.0, 0.0), 1))
        self.assertEqual(first.body["clear_result"], "success")
        self.assertEqual(second.body["clear_result"], "no_target_in_range")
        self.assertEqual(second.body["virtual_time_s"], 8.0)

    def test_local_simulator_reported_error_stays_within_one_degree(self) -> None:
        source = Source(1, (1000.0, 333.0), 1500.0)
        client = LocalSimulatorClient([source])
        client.execute(Action(ActionKind.ENTER, "enter"))
        for index, point in enumerate(((0.0, 0.0), (100.0, -200.0), (-300.0, 100.0))):
            response = client.execute(Action(ActionKind.MEASURE, f"m{index}", point, 1))
            measured = float(response.body["svd_deg"])
            truth = math.degrees(math.atan2(source.position[1] - point[1], source.position[0] - point[0]))
            difference = abs((measured - truth + 180.0) % 360.0 - 180.0)
            self.assertLessEqual(difference, 1.0 + 1.0e-9)

    def test_payload_has_no_unknown_fields(self) -> None:
        client = SimulatorClient("http://127.0.0.1:2026", "test-id", Q3Config())
        path, payload = client.payload_for(Action(ActionKind.CLEAR, "c", (1.0, 2.0), 3))
        self.assertEqual(path, "/clear")
        self.assertEqual(set(payload), {"arena_id", "robot_id", "request_id", "position", "channel"})

    def test_switch_estimate_matches_simulator_sequence(self) -> None:
        config = Q3Config()
        client = LocalSimulatorClient([], config)
        client.execute(Action(ActionKind.ENTER, "enter"))
        sequence = ordered_channels([2, 3, 4], 1, None)
        for index, channel in enumerate(sequence):
            client.execute(Action(ActionKind.MEASURE, f"m-{index}", (0.0, 0.0), channel))
        estimated = (
            len(sequence) * config.measure_time_s
            + sequence_switch_count(1, sequence) * config.switch_time_s
        )
        self.assertEqual(client.virtual_time_s, estimated)


class LatestPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Q3Config()
        self.strategy = Q3Strategy(self.config, mode=StrategyMode.B_PLUS)

    def _single_cell_mask(self, point: tuple[float, float]) -> np.ndarray:
        mask = np.zeros(self.strategy.grid.shape, dtype=bool)
        col = int(np.argmin(np.abs(self.strategy.grid.x_centers - point[0])))
        row = int(np.argmin(np.abs(self.strategy.grid.y_centers - point[1])))
        mask[row, col] = True
        return mask

    def test_variable_radius_history_only_expands_candidate_guarantee(self) -> None:
        state = ChannelState(1, phase=ChannelPhase.ACTIVE)
        state.region_mask = self._single_cell_mask((1400.0, 0.0))
        state.direction_observations.append(DirectionEvidence((0.0, 0.0), 0.0))
        candidate = (2600.0, 0.0)
        self.assertFalse(self.strategy.grid.point_guarantees_reception(state.region_mask, candidate))
        self.assertTrue(
            self.strategy.grid.point_guarantees_reception_with_history(state, candidate)
        )

    def test_outside_hexagon_committed_channel_uses_nearest_opportunity(self) -> None:
        robot = RobotState(
            position=search_points(self.config)[1],
            entered=True,
            visited_search_points={0, 1},
            last_outer_search_index=1,
        )
        channels = new_channel_states(self.config)
        state = channels[1]
        state.phase = ChannelPhase.CLEAR_COMMITTED
        state.clear_committed = True
        state.region_mask = self._single_cell_mask((1600.0, 0.0))
        action = self.strategy._best_local_insertion(
            robot,
            channels,
            search_points(self.config)[2],
            list(search_points(self.config)[2:]),
            2,
        )
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, ActionKind.CLEAR)
        self.assertTrue(action.metadata["outside_hexagon"])
        self.assertTrue(action.metadata["at_nearest_opportunity"])

    def test_safe_and_emergency_modes_disable_optimization(self) -> None:
        channels = new_channel_states(self.config)
        safe_robot = RobotState(entered=True, entered_real_monotonic=time.monotonic() - 901.0)
        self.strategy.next_action(safe_robot, channels)
        self.assertEqual(safe_robot.safety_mode.value, "SAFE")
        self.assertTrue(self.strategy._force_fallback)

        emergency = Q3Strategy(self.config, mode=StrategyMode.B_PLUS)
        emergency_robot = RobotState(
            entered=True, entered_real_monotonic=time.monotonic() - 1081.0
        )
        emergency.next_action(emergency_robot, new_channel_states(self.config))
        self.assertEqual(emergency_robot.safety_mode.value, "EMERGENCY")

    def test_planning_snapshot_separates_H_and_safe_ledgers(self) -> None:
        robot = RobotState(entered=True)
        channels = new_channel_states(self.config)
        self.strategy.next_action(robot, channels)
        snapshot = self.strategy._last_planning_snapshot
        self.assertIn("H_hat_s", snapshot)
        self.assertIn("U_survey_s", snapshot)
        self.assertIn("U_clear_s", snapshot)
        self.assertIn("U_future_s", snapshot)
        self.assertAlmostEqual(
            snapshot["U_safe_s"],
            snapshot["U_survey_s"] + snapshot["U_clear_s"] + snapshot["U_future_s"],
        )

    def test_first_outer_default_and_then_adjacent(self) -> None:
        robot = RobotState(entered=True, visited_search_points={0})
        channels = {1: ChannelState(1)}
        remaining = {index: self.strategy.fixed_points[index] for index in range(1, 7)}
        self.assertEqual(self.strategy._select_next_fixed_index(robot, channels, remaining), 1)
        robot.last_outer_search_index = 2
        for index in (1, 2, 3):
            remaining.pop(index)
        self.assertEqual(self.strategy._select_next_fixed_index(robot, channels, remaining), 4)

    def test_origin_direction_uses_perpendicular_tie_break(self) -> None:
        robot = RobotState(entered=True, visited_search_points={0})
        mask = np.zeros_like(self.strategy.grid.target_mask)
        row = int(np.argmin(np.abs(self.strategy.grid.y_centers)))
        col = int(np.argmin(np.abs(self.strategy.grid.x_centers)))
        mask[row, col] = True
        state = ChannelState(
            1,
            phase=ChannelPhase.ACTIVE,
            direction_observations=[DirectionEvidence((0.0, 0.0), 0.0)],
            region_mask=mask,
        )
        remaining = {index: self.strategy.fixed_points[index] for index in range(1, 7)}
        self.assertEqual(self.strategy._select_next_fixed_index(robot, {1: state}, remaining), 2)

    def test_optional_measurement_cap_is_two_per_stop(self) -> None:
        robot = RobotState(
            entered=True,
            visited_search_points={0},
            optional_measurements_at_stop=2,
            optional_measurement_position=(0.0, 0.0),
        )
        self.assertIsNone(
            self.strategy._zero_move_opportunity(robot, {1: ChannelState(1)}, None, [], None)
        )

    def test_dedicated_limit_defers_commit_for_rescue_comparison(self) -> None:
        mask = np.zeros_like(self.strategy.grid.target_mask)
        row = int(np.argmin(np.abs(self.strategy.grid.y_centers)))
        col = int(np.argmin(np.abs(self.strategy.grid.x_centers)))
        mask[row, col] = True
        state = ChannelState(1, phase=ChannelPhase.ACTIVE, region_mask=mask)
        action = Action(
            ActionKind.MEASURE,
            "supplement",
            (0.0, 0.0),
            1,
            metadata={
                "supplementary": True,
                "before_clear_count": 1,
                "before_clear_time_s": 0.0,
                "reconnect_point": None,
            },
        )
        state.dedicated_measurement_count = self.config.max_dedicated_measurements_per_channel
        self.strategy._record_supplement_gain(
            state, action, "direction", RobotState(position=(0.0, 0.0))
        )
        self.assertFalse(state.clear_committed)

    def test_failed_committed_clear_rebuilds_plan(self) -> None:
        grid = self.strategy.grid
        mask = np.zeros_like(grid.target_mask)
        row = int(np.argmin(np.abs(grid.y_centers)))
        col_a = int(np.argmin(np.abs(grid.x_centers)))
        col_b = int(np.argmin(np.abs(grid.x_centers - 100.0)))
        mask[row, col_a] = True
        mask[row, col_b] = True
        state = ChannelState(
            1,
            phase=ChannelPhase.ACTIVE,
            region_mask=mask,
            clear_committed=True,
        )
        robot = RobotState(entered=True)
        first = self.strategy.next_action(robot, {1: state})
        self.assertEqual(first.kind, ActionKind.CLEAR)
        self.strategy.apply_response(
            first,
            {"accepted": True, "virtual_time_s": 3.0, "clear_result": "no_target_in_range"},
            robot,
            {1: state},
        )
        second = self.strategy.next_action(robot, {1: state})
        self.assertEqual(second.kind, ActionKind.CLEAR)
        self.assertNotEqual(first.position, second.position)
        self.assertFalse(second.metadata["clear_certificate_reused"])

    def test_unchanged_no_signal_is_retained_in_candidate_worst_branch(self) -> None:
        state = ChannelState(1, phase=ChannelPhase.ACTIVE)
        state.region_mask = self._single_cell_mask((1000.0, 0.0))
        robot = RobotState(position=(0.0, 1000.0), current_channel=1, entered=True)
        self.strategy.planner.rank_measure_candidates(state, robot, [], None)
        evaluations = self.strategy.planner.last_candidate_evaluations[(1, state.region_version)]
        current = next(item for item in evaluations if tuple(item["point"]) == robot.position)
        self.assertEqual(current["branch_remaining_cells"]["no_signal"], 1)
        self.assertTrue(current["decision_band"].startswith("REJECTED_"))

    def test_side_geometry_candidates_survive_route_preselection(self) -> None:
        state = ChannelState(
            1,
            phase=ChannelPhase.ACTIVE,
            direction_observations=[DirectionEvidence((0.0, 0.0), 0.0)],
        )
        state.region_mask = self._single_cell_mask((1000.0, 0.0))
        candidates = self.strategy.planner.generate_measure_candidates(
            state, RobotState(entered=True), [], None
        )
        self.assertTrue(any(source == "side_geometry" for _, source, _ in candidates))

    def test_committed_clear_reenters_task_pool_after_failure(self) -> None:
        grid = self.strategy.grid
        mask = np.zeros_like(grid.target_mask)
        row = int(np.argmin(np.abs(grid.y_centers)))
        for x in (0.0, 100.0):
            mask[row, int(np.argmin(np.abs(grid.x_centers - x)))] = True
        state = ChannelState(1, phase=ChannelPhase.CLEAR_COMMITTED, region_mask=mask, clear_committed=True)
        robot = RobotState(entered=True)
        first = self.strategy.next_action(robot, {1: state})
        self.strategy.apply_response(
            first,
            {"accepted": True, "virtual_time_s": 3.0, "clear_result": "no_target_in_range"},
            robot,
            {1: state},
        )
        second = self.strategy.next_action(robot, {1: state})
        self.assertTrue(second.metadata["clear_task_reentered_pool"])
        self.assertEqual(second.reason, "global certified clear task")


class EndToEndTests(unittest.TestCase):
    def _run(self, sources: list[Source], optimized: bool = False) -> tuple[dict[str, object], LocalSimulatorClient]:
        client = LocalSimulatorClient(sources)
        strategy = Q3Strategy(mode=StrategyMode.B_PLUS if optimized else StrategyMode.B)
        with tempfile.TemporaryDirectory(prefix="q3-test-") as temp:
            summary = run_strategy(client, strategy, Path(temp), max_actions=20_000)
        return summary, client

    def test_run_writes_intent_response_and_checkpoint(self) -> None:
        client = LocalSimulatorClient([])
        strategy = Q3Strategy(mode=StrategyMode.B)
        with tempfile.TemporaryDirectory(prefix="q3-log-") as temp:
            path = Path(temp)
            summary = run_strategy(client, strategy, path, max_actions=1)
            records = [json.loads(line) for line in (path / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
            checkpoint = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual([record["stage"] for record in records], ["intent", "response"])
        self.assertEqual(checkpoint["mode"], "B")
        self.assertEqual(summary["spec_version"], "q3-b-bplus-v7-directional-rescue")

    def test_origin_scan_is_119_without_near(self) -> None:
        client = LocalSimulatorClient([])
        strategy = Q3Strategy(mode=StrategyMode.B)
        from Q3.models import RobotState
        from Q3.strategy import new_channel_states

        robot = RobotState()
        channels = new_channel_states(strategy.config)
        action = strategy.next_action(robot, channels)
        response = client.execute(action)
        strategy.apply_response(action, response.body, robot, channels)
        measure_count = 0
        while measure_count < 20:
            action = strategy.next_action(robot, channels)
            response = client.execute(action)
            strategy.apply_response(action, response.body, robot, channels)
            if action.kind == ActionKind.MEASURE:
                measure_count += 1
        self.assertEqual(robot.virtual_time_s, 119.0)

    def test_one_source_full_completion(self) -> None:
        sources = [Source(7, (900.0, 200.0), 1200.0)]
        summary, client = self._run(sources)
        self.assertTrue(summary["completion_certificate"]["valid"])
        self.assertEqual(summary["cleared_count"], 1)
        self.assertTrue(client.sources[7].cleared)
        self.assertIsNone(summary["terminal_error"])
        self.assertEqual(summary["mode"], "B")
        self.assertAlmostEqual(
            sum(summary["time_breakdown"].values()),
            summary["virtual_total_time_s"],
            places=6,
        )

    def test_empty_world_search_backbone_is_2213(self) -> None:
        summary, _ = self._run([])
        self.assertTrue(summary["completion_certificate"]["valid"])
        self.assertAlmostEqual(summary["virtual_total_time_s"], 2213.0, places=6)

    def test_zone_evidence_is_channel_specific_and_skips_redundant_measurement(self) -> None:
        config = Q3Config()
        strategy = Q3Strategy(config, mode=StrategyMode.B)
        from Q3.strategy import new_channel_states

        channels = new_channel_states(config)
        channels[1].certified_zone_indices.add(1)
        channels[1].zone_certificate_points[1] = search_points(config)[1]
        self.assertNotIn(1, strategy._scan_channels_for_index(1, channels))
        self.assertIn(2, strategy._scan_channels_for_index(1, channels))
        for state in channels.values():
            state.certified_zone_indices.add(1)
            state.zone_certificate_points[1] = search_points(config)[1]
        from Q3.models import RobotState

        robot = RobotState(visited_search_points={0})
        self.assertNotIn(1, strategy._remaining_fixed_indices(robot, channels))

    def test_same_point_can_certify_same_zone_for_multiple_channels(self) -> None:
        config = Q3Config()
        client = LocalSimulatorClient([], config)
        strategy = Q3Strategy(config, mode=StrategyMode.B)
        from Q3.models import RobotState
        from Q3.strategy import new_channel_states

        robot = RobotState()
        channels = new_channel_states(config)
        enter = strategy.next_action(robot, channels)
        strategy.apply_response(enter, client.execute(enter).body, robot, channels)
        point = search_points(config)[0]
        for channel in (1, 2):
            action = Action(ActionKind.MEASURE, f"z-{channel}", point, channel)
            strategy.apply_response(action, client.execute(action).body, robot, channels)
            self.assertIn(0, channels[channel].certified_zone_indices)

    def test_nonfixed_certificate_removes_a_redundant_fixed_task_and_time(self) -> None:
        config = Q3Config()
        point = (1140.0, 0.0)
        self.assertEqual(certified_responsibility_zones(point, config), {1})
        from Q3.models import RobotState
        from Q3.strategy import new_channel_states

        robot = RobotState(entered=True, visited_search_points={0})
        zone_strategy = Q3Strategy(config, mode=StrategyMode.B_PLUS)
        legacy_strategy = Q3Strategy(config, mode=StrategyMode.B)
        zone_channels = new_channel_states(config)
        legacy_channels = new_channel_states(config)
        for number in range(2, 21):
            zone_channels[number].phase = ChannelPhase.CLEARED
            legacy_channels[number].phase = ChannelPhase.CLEARED
        zone_channels[1].certified_zone_indices.update({0, 1})
        zone_channels[1].zone_certificate_points.update(
            {0: search_points(config)[0], 1: point}
        )
        zone_remaining = zone_strategy._remaining_fixed_indices(robot, zone_channels)
        legacy_remaining = legacy_strategy._remaining_fixed_indices(robot, legacy_channels)
        self.assertNotIn(1, zone_remaining)
        self.assertIn(1, legacy_remaining)
        zone_order = weighted_open_search_order(
            robot.position,
            robot.current_channel,
            zone_remaining,
            {index: [1] for index in zone_remaining},
            config,
        )
        legacy_order = weighted_open_search_order(
            robot.position,
            robot.current_channel,
            legacy_remaining,
            {index: [1] for index in legacy_remaining},
            config,
        )
        zone_distance = sum(
            math.dist(
                robot.position if offset == 0 else zone_remaining[zone_order[offset - 1]],
                zone_remaining[index],
            )
            for offset, index in enumerate(zone_order)
        )
        legacy_distance = sum(
            math.dist(
                robot.position if offset == 0 else legacy_remaining[legacy_order[offset - 1]],
                legacy_remaining[index],
            )
            for offset, index in enumerate(legacy_order)
        )
        self.assertLess(len(zone_order), len(legacy_order))
        self.assertLess(
            zone_distance / config.speed_mps + len(zone_order) * config.measure_time_s,
            legacy_distance / config.speed_mps + len(legacy_order) * config.measure_time_s,
        )

    def test_zone_completion_certificate_rechecks_every_analytic_certificate(self) -> None:
        config = Q3Config()
        strategy = Q3Strategy(config, mode=StrategyMode.B)
        from Q3.strategy import new_channel_states

        channels = new_channel_states(config)
        for number, state in channels.items():
            state.phase = ChannelPhase.EXCLUDED
            state.certified_zone_indices.update(range(7))
            state.zone_certificate_points.update(
                {index: point for index, point in enumerate(search_points(config))}
            )
        certificate = strategy.completion_certificate(channels)
        self.assertTrue(certificate["valid"])
        self.assertTrue(
            all(
                item["seven_point_coverage_valid"]
                for item in certificate["channels"].values()
            )
        )

    def test_direction_cancels_remaining_zone_search_obligations(self) -> None:
        config = Q3Config()
        strategy = Q3Strategy(config, mode=StrategyMode.B)
        from Q3.models import RobotState
        from Q3.strategy import new_channel_states

        channels = new_channel_states(config)
        state = channels[1]
        state.phase = ChannelPhase.ACTIVE
        state.region_mask = strategy.grid.fresh_region()
        robot = RobotState(visited_search_points={0})
        remaining = strategy._remaining_fixed_indices(robot, channels)
        for index in remaining:
            self.assertNotIn(1, strategy._scan_channels_for_index(index, channels))

    def test_zero_move_zone_measure_does_not_consume_nonzero_insertion(self) -> None:
        config = Q3Config()
        strategy = Q3Strategy(config, mode=StrategyMode.B_PLUS)
        from Q3.models import RobotState
        from Q3.strategy import new_channel_states

        robot = RobotState(
            position=search_points(config)[1],
            visited_search_points={0},
            entered=True,
        )
        channels = new_channel_states(config)
        action = strategy.next_action(robot, channels)
        self.assertTrue(action.metadata.get("zero_move"))
        self.assertEqual(robot.local_insertions_since_search, 0)

    def test_B_does_not_insert_optional_zero_move_task(self) -> None:
        config = Q3Config()
        strategy = Q3Strategy(config, mode=StrategyMode.B)
        robot = RobotState(
            position=search_points(config)[1],
            visited_search_points={0},
            entered=True,
        )
        action = strategy.next_action(robot, new_channel_states(config))
        self.assertFalse(action.metadata.get("optional_extra", False))
        self.assertEqual(action.metadata.get("fixed_search_index"), 1)

    def test_weighted_search_order_counts_full_channel_switch_sequence(self) -> None:
        config = Q3Config()
        remaining = {1: (10.0, 0.0), 2: (20.0, 0.0)}
        scans = {1: [2, 3], 2: [3, 4]}
        order = weighted_open_search_order((0.0, 0.0), 2, remaining, scans, config)
        self.assertEqual(order, [1, 2])
        sequence = ordered_channels([2, 3, 4], 2, 4)
        self.assertEqual(sequence, [2, 3, 4])
        self.assertEqual(sequence_switch_count(2, sequence, 4), 2)

    def test_sixteen_near_sources_trigger_early_completion(self) -> None:
        sources = [Source(channel, (0.0, 0.0), 1000.0) for channel in range(1, 17)]
        summary, _ = self._run(sources)
        self.assertTrue(summary["completion_certificate"]["early_stop_at_upper_bound"])
        self.assertEqual(summary["cleared_count"], 16)
        self.assertLess(summary["action_count"], 1 + 20 + 16 + 1)

    def test_ten_near_sources_do_not_trigger_early_exit(self) -> None:
        sources = [Source(channel, (0.0, 0.0), 1000.0) for channel in range(1, 11)]
        summary, _ = self._run(sources)
        self.assertTrue(summary["completion_certificate"]["valid"])
        self.assertEqual(summary["cleared_count"], 10)
        excluded = sum(value == ChannelPhase.EXCLUDED.value for value in summary["channel_states"].values())
        self.assertEqual(excluded, 10)

    def test_seed_10008_never_discards_a_live_source_cell(self) -> None:
        from Q3.local_simulator import random_case
        from Q3.models import RobotState
        from Q3.strategy import new_channel_states

        config = Q3Config()
        sources = random_case(10008, config)
        client = LocalSimulatorClient(sources, config)
        strategy = Q3Strategy(config, mode=StrategyMode.B_PLUS)
        robot = RobotState()
        channels = new_channel_states(config)
        for step in range(20_000):
            action = strategy.next_action(robot, channels)
            if action is None:
                break
            response = client.execute(action)
            strategy.apply_response(action, response.body, robot, channels)
            for source in sources:
                state = channels[source.channel]
                if source.cleared or state.region_mask is None:
                    continue
                row = int(math.floor((source.position[1] + config.target_radius_m) / config.grid_step_m))
                col = int(math.floor((source.position[0] + config.target_radius_m) / config.grid_step_m))
                self.assertTrue(
                    state.region_mask[row, col],
                    msg=f"truth cell lost at step={step}, action={action}, response={response.body}, source={source}",
                )
            if robot.exited:
                break
        self.assertEqual(strategy.anomalies, [])
        self.assertTrue(all(not state.anomaly_log for state in channels.values()))


if __name__ == "__main__":
    unittest.main()
