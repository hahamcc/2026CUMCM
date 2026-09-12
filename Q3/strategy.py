from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

from .config import Q3Config
from .geometry import (
    ConservativeGrid,
    certified_responsibility_zones,
    responsibility_zone_certificate,
    search_coverage_certificate,
    search_points,
)
from .models import (
    Action,
    ActionKind,
    ClearRouteSession,
    ChannelPhase,
    ChannelState,
    DirectionEvidence,
    MeasurementAppointment,
    Point,
    RobotState,
    RunSummary,
    SafetyMode,
    StrategyMode,
)
from .planner import RollingPlanner, ordered_channels, shortest_open_search_order


ACTIVE_PHASES = {ChannelPhase.ACTIVE, ChannelPhase.CLEAR_COMMITTED}


class Q3Strategy:
    def __init__(
        self,
        config: Q3Config | None = None,
        *,
        mode: StrategyMode | str = StrategyMode.B_PLUS,
    ):
        self.config = config or Q3Config()
        self.mode = StrategyMode(mode)
        self.grid = ConservativeGrid(self.config)
        self.planner = RollingPlanner(self.config, self.grid)
        self.fixed_points = search_points(self.config)
        self.enable_optimization = self.mode == StrategyMode.B_PLUS
        self.use_zone_scheduler = True
        self._request_counter = 0
        self._scan_point_index: int | None = None
        self._scan_queue: deque[int] = deque()
        self._urgent_clear: deque[int] = deque()
        self._force_fallback = False
        self._deep_scoring_disabled = False
        self._clear_session: ClearRouteSession | None = None
        self._tail_cluster_order: list[int] = []
        self._tail_cluster_history: list[dict[str, Any]] = []
        self._tail_clusters_dirty = True
        self._commit_counter = 0
        self._active_ring_order: list[int] = []
        self._last_planning_snapshot: dict[str, Any] = {}
        self.decision_trace: list[dict[str, Any]] = []
        self.anomalies: list[str] = []
        self.action_count = 0

    def _request_id(self, label: str) -> str:
        self._request_counter += 1
        return f"q3-{self._request_counter:06d}-{label}"

    def next_action(
        self, robot: RobotState, channels: dict[int, ChannelState]
    ) -> Action | None:
        if not robot.entered:
            return Action(ActionKind.ENTER, self._request_id("enter"), reason="start run")
        if robot.exited:
            return None
        if self.is_complete(channels):
            return Action(ActionKind.EXIT, self._request_id("exit"), reason="completion certificate satisfied")

        self._update_safety_mode(robot)

        while self._urgent_clear:
            channel = self._urgent_clear.popleft()
            state = channels[channel]
            if state.near_pending and state.phase in ACTIVE_PHASES:
                return Action(
                    ActionKind.CLEAR,
                    self._request_id(f"clear-near-{channel}"),
                    robot.position,
                    channel,
                    reason="near response: immediate clear",
                    metadata={"near": True},
                )

        continuation = self._continue_clear_session(robot, channels)
        if continuation is not None:
            return continuation

        co_measurement = self._co_measurement_action(robot, channels)
        if co_measurement is not None:
            return co_measurement

        while self._scan_queue:
            channel = self._scan_queue.popleft()
            state = channels[channel]
            if state.complete:
                continue
            if self._scan_point_index is None:
                raise RuntimeError("scan queue exists without a fixed point")
            return Action(
                ActionKind.MEASURE,
                self._request_id(f"measure-s{self._scan_point_index}-c{channel}"),
                self.fixed_points[self._scan_point_index],
                channel,
                reason=(
                    "early broad full-channel discovery scan"
                    if self._is_early_broad_outer_index(self._scan_point_index)
                    else "fixed coverage scan"
                ),
                metadata={
                    "fixed_search_index": self._scan_point_index,
                    "ring_direction": robot.ring_direction,
                    "outer_segment": robot.current_outer_segment,
                    "early_broad_discovery": self._is_early_broad_outer_index(self._scan_point_index),
                },
            )

        if self._scan_point_index is not None:
            completed_segment = robot.current_outer_segment
            if completed_segment is not None:
                self._expire_completed_segment_appointments(channels, completed_segment)
            robot.visited_search_points.add(self._scan_point_index)
            robot.local_insertions_since_search = 0
            robot.appointment_measurements_since_search = 0
            robot.directional_rescues_since_search = 0
            if self._scan_point_index > 0:
                robot.last_outer_search_index = self._scan_point_index
                robot.completed_outer_segments += 1
                self._update_waiting_segments(robot, channels)
            self._scan_point_index = None

        remaining_indices = self._remaining_fixed_indices(robot, channels)
        if remaining_indices:
            scan_by_index = {
                index: self._scan_channels_for_index(index, channels)
                for index in remaining_indices
            }
            self._refresh_planning_snapshot(robot, channels, remaining_indices, scan_by_index)
            next_index = self._select_next_fixed_index(robot, channels, remaining_indices)
            next_point = self.fixed_points[next_index]
            next_sequence = ordered_channels(scan_by_index[next_index], robot.current_channel)
            next_channel = next_sequence[0] if next_sequence else None
            if self.mode == StrategyMode.B_PLUS and robot.visited_search_points and not self._deep_scoring_disabled:
                self._refresh_measurement_appointments(robot, channels, next_point, next_channel)
                appointment = self._due_appointment_action(robot, channels, next_point, next_channel)
                if appointment is not None:
                    return appointment
                zero_move = self._zero_move_opportunity(
                    robot, channels, next_point, list(remaining_indices.values()), next_channel
                )
                if zero_move is not None:
                    return zero_move
            if (
                self.enable_optimization
                and not self._force_fallback
                and not self._deep_scoring_disabled
                and robot.local_insertions_since_search
                < self.config.max_macro_insertions_per_segment
            ):
                insertion = self._best_local_insertion(
                    robot,
                    channels,
                    next_point,
                    list(remaining_indices.values()),
                    next_channel,
                )
                if insertion is not None:
                    robot.local_insertions_since_search += 1
                    self._record_macro_decision(insertion, next_index)
                    return insertion

            self._record_macro_decision(None, next_index)
            self._scan_point_index = next_index
            scan_channels = self._scan_channels_for_index(next_index, channels)
            self._scan_queue.extend(ordered_channels(scan_channels, robot.current_channel))
            if not self._scan_queue:
                robot.visited_search_points.add(next_index)
                self._scan_point_index = None
                return self.next_action(robot, channels)
            return self.next_action(robot, channels)

        if self.mode == StrategyMode.B_PLUS and robot.visited_search_points and not self._deep_scoring_disabled:
            zero_move = self._zero_move_opportunity(robot, channels, None, [], None)
            if zero_move is not None:
                return zero_move
        self._finalize_exclusions(channels)
        self._refresh_planning_snapshot(robot, channels, {}, {})
        unresolved = [state for state in channels.values() if state.phase in ACTIVE_PHASES]
        if not unresolved:
            if self.is_complete(channels):
                return Action(ActionKind.EXIT, self._request_id("exit"), reason="all channels completed")
            raise RuntimeError("search finished but at least one channel lacks completion evidence")
        return self._post_search_action(robot, unresolved)

    def _select_next_fixed_index(
        self,
        robot: RobotState,
        channels: dict[int, ChannelState],
        remaining: dict[int, Point],
    ) -> int:
        """Choose a once-fixed B+ ring direction, then retain it."""
        if 0 in remaining:
            return 0
        outer = sorted(index for index in remaining if index > 0)
        if not outer:
            return min(remaining)
        if self.mode == StrategyMode.B:
            return 1 if 1 in remaining else outer[0]

        if not robot.ring_order:
            robot.ring_order, robot.ring_direction = self._choose_ring_order(robot, channels)
        self._active_ring_order = list(robot.ring_order)
        for index in robot.ring_order:
            if index in remaining:
                previous = robot.last_outer_search_index or 0
                robot.current_outer_segment = (previous, index)
                return index
        return outer[0]

    def _choose_ring_order(
        self, robot: RobotState, channels: dict[int, ChannelState]
    ) -> tuple[list[int], str]:
        """Compare only the two possible ring directions at the origin.

        Ring length is identical in both directions. The comparison therefore
        uses coarse, worst-branch savings for exterior opportunity channels on
        the respective first segment; ties deliberately retain clockwise order.
        """
        orders = (([1, 2, 3, 4, 5, 6], "clockwise"), ([6, 5, 4, 3, 2, 1], "counterclockwise"))
        scored: list[tuple[float, int, list[int], str]] = []
        for order, label in orders:
            next_point = self.fixed_points[order[0]]
            saving = 0.0
            for state in channels.values():
                if (
                    not self._is_outer_opportunity(state)
                    and self._appointment_classification(state, robot)
                    not in {"LARGE_UNCERTAIN", "DIRECTIONAL_RESCUE"}
                ):
                    continue
                ranked = self.planner.rank_measure_candidates(
                    state,
                    robot,
                    [self.fixed_points[index] for index in order],
                    next_point,
                    guaranteed_only=True,
                    directional_rescue=self._appointment_classification(state, robot) == "DIRECTIONAL_RESCUE",
                )
                if ranked:
                    best = ranked[0]
                    saving += max(0.0, best.direct_clear_baseline_s - best.estimated_total_s)
            scored.append((saving, int(label == "clockwise"), order, label))
        _, _, order, label = max(scored, key=lambda item: (item[0], item[1]))
        return list(order), label

    def _is_outer_opportunity(self, state: ChannelState) -> bool:
        return bool(
            state.phase in ACTIVE_PHASES
            and state.region_mask is not None
            and self.grid.region_extends_outside_fallback_hexagon(state.region_mask)
        )

    def _is_ring_active(self, robot: RobotState) -> bool:
        return bool(robot.ring_order and any(index not in robot.visited_search_points for index in robot.ring_order))

    def _remaining_ring_indices(self, robot: RobotState) -> list[int]:
        return [index for index in robot.ring_order if index not in robot.visited_search_points]

    def _invalidate_appointments(self, state: ChannelState, reason: str) -> None:
        """Retire appointments whose geometry no longer matches the safe region."""
        for appointment in state.appointments:
            appointment.status = "INVALIDATED"
            appointment.status_reason = reason
            state.appointment_history.append(
                {
                    "event": "invalidated",
                    "appointment_id": appointment.appointment_id,
                    "region_version": appointment.region_version,
                    "reason": reason,
                }
            )
        state.appointments.clear()

    @staticmethod
    def _expire_completed_segment_appointments(
        channels: dict[int, ChannelState], segment: tuple[int, int]
    ) -> None:
        """A reservation expires only after its complete directed window closes."""
        for state in channels.values():
            for appointment in list(state.appointments):
                if appointment.status != "PENDING" or appointment.anchor_segment != segment:
                    continue
                appointment.status = "EXPIRED"
                appointment.status_reason = "forward_window_missed"
                state.appointment_history.append(
                    {
                        "event": "expired",
                        "appointment_id": appointment.appointment_id,
                        "reason": appointment.status_reason,
                        "window_segment": segment,
                    }
                )
                state.appointments.remove(appointment)

    def _appointment_classification(self, state: ChannelState, robot: RobotState) -> str:
        if state.region_mask is None:
            return "TAIL_ONLY"
        clear_time = self.planner.fast_clear_time_proxy(state.region_mask, robot.position)
        clear_count = self.planner.fast_clear_count_proxy(state.region_mask)
        if (
            state.direction_count < 2
            and (
                clear_time >= self.config.appointment_large_clear_time_s
                or clear_count >= self.config.appointment_large_clear_count
            )
        ):
            return "DIRECTIONAL_RESCUE"
        if (
            clear_time >= self.config.appointment_large_clear_time_s
            or clear_count >= self.config.appointment_large_clear_count
        ):
            return "LARGE_UNCERTAIN"
        if state.appointments:
            segment = robot.current_outer_segment
            if segment is not None and any(item.anchor_segment == segment for item in state.appointments):
                return "CURRENT_OPPORTUNITY"
            return "FUTURE_OPPORTUNITY"
        return "TAIL_ONLY"

    def _forward_anchor(
        self, point: Point, robot: RobotState
    ) -> tuple[tuple[int, int], int] | None:
        """Attach a candidate to its least-detour unvisited directed ring segment."""
        remaining = self._remaining_ring_indices(robot)
        if not remaining:
            return None
        start_index = robot.last_outer_search_index or 0
        start_point = robot.position
        route: list[tuple[int, Point]] = [(start_index, start_point)]
        route.extend((index, self.fixed_points[index]) for index in remaining)
        options: list[tuple[float, tuple[int, int], int]] = []
        for (left_index, left), (right_index, right) in zip(route, route[1:]):
            detour = math.dist(left, point) + math.dist(point, right) - math.dist(left, right)
            options.append((detour, (left_index, right_index), right_index))
        if not options:
            return None
        _, segment, _ = min(options, key=lambda item: (item[0], item[2]))
        # A segment opens at its left endpoint; use that as the trigger anchor.
        return segment, segment[0]

    @staticmethod
    def _appointment_reduction(state: ChannelState, candidate: Any, cover_capacity: int) -> int:
        before = max(1, int(cover_capacity))
        before_count = max(1, math.ceil(int(state.region_mask.sum()) / before)) if state.region_mask is not None else 0
        remaining = max(candidate.branch_remaining_cells.values(), default=0)
        after_count = max(0, math.ceil(remaining / before))
        return max(0, before_count - after_count)

    def _refresh_measurement_appointments(
        self,
        robot: RobotState,
        channels: dict[int, ChannelState],
        next_point: Point,
        next_channel: int | None,
    ) -> None:
        """Maintain at most two high-value measurement reservations per channel."""
        if not self.enable_optimization or not self._is_ring_active(robot):
            return
        remaining_points = [self.fixed_points[index] for index in self._remaining_ring_indices(robot)]
        cover_capacity = max(1, len(self.grid._cover_offsets()))
        eligible_states = [
            state
            for state in channels.values()
            if state.phase in ACTIVE_PHASES
            and state.region_mask is not None
            and not state.clear_committed
            and state.direction_count < self.config.max_direction_observations_per_channel
        ]
        eligible_states.sort(
            key=lambda state: (
                -int(self._appointment_classification(state, robot) == "DIRECTIONAL_RESCUE"),
                -int(self._appointment_classification(state, robot) == "LARGE_UNCERTAIN"),
                -self.planner.fast_clear_count_proxy(state.region_mask),
                -self.planner.fast_clear_time_proxy(state.region_mask, robot.position),
                -int(self._is_outer_opportunity(state)),
                state.channel,
            )
        )
        for state in eligible_states:
            if (
                state.region_mask is None
            ):
                continue
            stale = [item for item in state.appointments if item.region_version != state.region_version]
            if stale:
                self._invalidate_appointments(state, "region_version_changed")
            valid = [item for item in state.appointments if item.status == "PENDING"]
            if len(valid) >= self.config.max_measurement_appointments_per_channel:
                continue
            rescue = self._appointment_classification(state, robot) == "DIRECTIONAL_RESCUE"
            ranked = self.planner.rank_measure_candidates(
                state,
                robot,
                remaining_points,
                next_point,
                None,
                next_channel,
                guaranteed_only=True,
                directional_rescue=rescue,
            )
            safe_candidate_count = sum(
                int(candidate.guaranteed_reception) for candidate in ranked
            )
            if rescue and not ranked:
                state.directional_rescue_history.append({
                    "event": "no_safe_candidate_for_forward_appointment",
                    "region_version": state.region_version,
                    "direction_count": state.direction_count,
                    "clear_time_proxy_s": self.planner.fast_clear_time_proxy(state.region_mask, robot.position),
                    "clear_count_proxy": self.planner.fast_clear_count_proxy(state.region_mask),
                })
            existing_anchors = {item.anchor_segment for item in valid}
            candidates: list[tuple[float, MeasurementAppointment]] = []
            for rank, candidate in enumerate(ranked):
                if not candidate.guaranteed_reception:
                    continue
                anchor = self._forward_anchor(candidate.point, robot)
                if anchor is None:
                    continue
                segment, anchor_index = anchor
                if segment in existing_anchors:
                    continue
                saving = candidate.direct_clear_baseline_s - candidate.estimated_total_s
                reduction = self._appointment_reduction(state, candidate, cover_capacity)
                if saving < self.config.gain_buffer_s - 1.0e-9 and reduction <= 0:
                    continue
                classification = "DIRECTIONAL_RESCUE" if rescue else ("LARGE_UNCERTAIN" if self._appointment_classification(state, robot) == "LARGE_UNCERTAIN" else (
                    "CURRENT_OPPORTUNITY" if segment == robot.current_outer_segment else "FUTURE_OPPORTUNITY"
                ))
                appointment = MeasurementAppointment(
                    appointment_id=f"c{state.channel}-v{state.region_version}-a{len(state.appointment_history) + rank + 1}",
                    point=candidate.point,
                    source=candidate.source,
                    anchor_segment=segment,
                    anchor_index=anchor_index,
                    region_version=state.region_version,
                    guaranteed_reception=candidate.guaranteed_reception,
                    estimated_saving_s=saving,
                    estimated_clear_count_reduction=reduction,
                    classification=classification,
                )
                candidates.append((-(saving + 6.0 * reduction), appointment))
            for _, appointment in sorted(candidates, key=lambda item: (item[0], item[1].anchor_index, item[1].source)):
                if len(valid) >= self.config.max_measurement_appointments_per_channel:
                    break
                if appointment.anchor_segment in existing_anchors:
                    continue
                state.appointments.append(appointment)
                state.appointment_history.append(
                    {
                        "event": "created",
                        "appointment_id": appointment.appointment_id,
                        "point": appointment.point,
                        "source": appointment.source,
                        "anchor_segment": appointment.anchor_segment,
                        "anchor_index": appointment.anchor_index,
                        "region_version": appointment.region_version,
                        "guaranteed_reception": appointment.guaranteed_reception,
                        "estimated_saving_s": appointment.estimated_saving_s,
                        "estimated_clear_count_reduction": appointment.estimated_clear_count_reduction,
                        "classification": appointment.classification,
                        "safe_candidate_count": safe_candidate_count,
                        "risk_priority": {
                            "large_uncertain": appointment.classification == "LARGE_UNCERTAIN",
                            "clear_count_reduction": appointment.estimated_clear_count_reduction,
                            "estimated_saving_s": appointment.estimated_saving_s,
                            "outer_opportunity": self._is_outer_opportunity(state),
                        },
                        "window_segment": appointment.anchor_segment,
                    }
                )
                valid.append(appointment)
                existing_anchors.add(appointment.anchor_segment)

    def _due_appointment_action(
        self,
        robot: RobotState,
        channels: dict[int, ChannelState],
        next_point: Point,
        next_channel: int | None,
    ) -> Action | None:
        """Re-evaluate appointments only when their directed segment is reached."""
        if robot.current_outer_segment is None:
            return None
        if (
            robot.appointment_measurements_since_search
            >= self.config.max_optional_measurements_per_stop
        ):
            return None
        remaining_points = [self.fixed_points[index] for index in self._remaining_ring_indices(robot)]
        due: list[tuple[tuple[float, ...], ChannelState, MeasurementAppointment, Any, float]] = []
        for state in channels.values():
            for appointment in list(state.appointments):
                if appointment.status != "PENDING" or appointment.anchor_segment != robot.current_outer_segment:
                    continue
                if (
                    state.phase not in ACTIVE_PHASES
                    or state.clear_committed
                    or state.region_mask is None
                    or appointment.region_version != state.region_version
                    or state.direction_count >= self.config.max_direction_observations_per_channel
                ):
                    appointment.status = "INVALIDATED"
                    appointment.status_reason = "not_actionable_at_anchor"
                    state.appointment_history.append({"event": "invalidated", "appointment_id": appointment.appointment_id, "reason": appointment.status_reason})
                    state.appointments.remove(appointment)
                    continue
                ranked = self.planner.rank_measure_candidates(
                    state,
                    robot,
                    remaining_points,
                    next_point,
                    None,
                    next_channel,
                    retained_points=[(appointment.point, "appointment_retained")],
                    guaranteed_only=True,
                    directional_rescue=appointment.classification == "DIRECTIONAL_RESCUE",
                )
                candidate = next((item for item in ranked if math.dist(item.point, appointment.point) <= 1.0e-6), None)
                if candidate is None:
                    retained_log = next(
                        (
                            item for item in self.planner.last_candidate_evaluations.get(
                                (state.channel, state.region_version), []
                            )
                            if isinstance(item, dict)
                            and item.get("point") is not None
                            and math.dist(tuple(item["point"]), appointment.point) <= 1.0e-6
                        ),
                        None,
                    )
                    appointment.status = "EXPIRED"
                    appointment.status_reason = (
                        f"retained_point_rejected_{retained_log.get('decision_band', 'no_compatible_branch')}"
                        if retained_log is not None
                        else "retained_point_no_compatible_branch"
                    )
                    state.appointment_history.append({
                        "event": "expired",
                        "appointment_id": appointment.appointment_id,
                        "reason": appointment.status_reason,
                        "retained_point_revalidated": True,
                    })
                    state.appointments.remove(appointment)
                    continue
                saving = candidate.direct_clear_baseline_s - candidate.estimated_total_s
                if saving < self.config.gain_buffer_s - 1.0e-9:
                    appointment.status = "EXPIRED"
                    appointment.status_reason = "revalidation_below_six_second_gain"
                    state.appointment_history.append({"event": "expired", "appointment_id": appointment.appointment_id, "reason": appointment.status_reason, "revalidated_saving_s": saving})
                    state.appointments.remove(appointment)
                    continue
                if (
                    appointment.classification == "DIRECTIONAL_RESCUE"
                    and robot.directional_rescues_since_search
                    >= self.config.max_directional_rescues_per_segment
                ):
                    continue
                risk_priority = (
                    -float(appointment.classification == "DIRECTIONAL_RESCUE"),
                    -float(self._appointment_classification(state, robot) == "LARGE_UNCERTAIN"),
                    -float(appointment.estimated_clear_count_reduction),
                    -saving,
                    float(candidate.estimated_best_s),
                    -float(self._is_outer_opportunity(state)),
                    float(candidate.detour_distance_m),
                    float(state.channel),
                )
                due.append((risk_priority, state, appointment, candidate, saving))
        if not due:
            return None
        selected = min(due, key=lambda item: item[0])
        priority, state, appointment, candidate, saving = selected
        appointment.status = "EXECUTED"
        appointment.status_reason = "revalidated_at_forward_anchor"
        state.appointment_history.append(
            {
                "event": "triggered",
                "appointment_id": appointment.appointment_id,
                "window_segment": robot.current_outer_segment,
                "revalidated_saving_s": saving,
                "risk_priority": priority,
                "detour_distance_m": candidate.detour_distance_m,
            }
        )
        robot.appointment_measurements_since_search += 1
        if appointment.classification == "DIRECTIONAL_RESCUE":
            robot.directional_rescues_since_search += 1
            state.directional_rescue_history.append({
                "event": "directional_rescue_triggered",
                "region_version": state.region_version,
                "direction_count_before": state.direction_count,
                "safe_candidate_count": sum(
                    int(item.get("guaranteed_reception") is True)
                    for item in self.planner.last_candidate_evaluations.get(
                        (state.channel, state.region_version), []
                    )
                    if isinstance(item, dict)
                ),
                "estimated_saving_s": saving,
                "detour_distance_m": candidate.detour_distance_m,
            })
        return Action(
            ActionKind.MEASURE,
            self._request_id(f"appointment-measure-c{state.channel}"),
            appointment.point,
            state.channel,
            reason="forward measurement appointment revalidated at anchor",
            metadata={
                "local_insertion": True,
                "supplementary": True,
                "dedicated_measurement": math.dist(robot.position, appointment.point) > 1.0e-6,
                "appointment_id": appointment.appointment_id,
                "appointment_anchor_segment": appointment.anchor_segment,
                "appointment_classification": appointment.classification,
                "appointment_estimated_saving_s": appointment.estimated_saving_s,
                "appointment_estimated_clear_count_reduction": appointment.estimated_clear_count_reduction,
                "appointment_revalidated_saving_s": saving,
                "appointment_risk_priority": priority,
                "appointment_detour_distance_m": candidate.detour_distance_m,
                "reconnect_point": next_point,
                "candidate_source": candidate.source,
                "candidate_guaranteed_reception": candidate.guaranteed_reception,
                "candidate_worst_branch": candidate.worst_branch,
                "candidate_branch_estimates_s": candidate.branch_estimates_s,
                "ring_direction": robot.ring_direction,
                "outer_segment": robot.current_outer_segment,
                "active_measurement_count_before": state.active_measurement_count,
                "direction_count_before": state.direction_count,
                "directional_rescue": appointment.classification == "DIRECTIONAL_RESCUE",
                "enable_co_measurements": True,
            },
        )

    def _has_current_segment_appointment(self, state: ChannelState, robot: RobotState) -> bool:
        return bool(
            robot.current_outer_segment is not None
            and any(
                item.status == "PENDING" and item.anchor_segment == robot.current_outer_segment
                for item in state.appointments
            )
        )

    @staticmethod
    def _prioritize_current_channel(channels: list[int], current: int) -> list[int]:
        if current not in channels:
            return sorted(channels)
        return [current] + sorted(channel for channel in channels if channel != current)

    def _remaining_fixed_indices(
        self, robot: RobotState, channels: dict[int, ChannelState]
    ) -> dict[int, Point]:
        return {
            index: point
            for index, point in enumerate(self.fixed_points)
            if index not in robot.visited_search_points
            and any(
                state.phase == ChannelPhase.UNKNOWN
                and index not in state.certified_zone_indices
                for state in channels.values()
            )
        }

    def _is_early_broad_outer_index(self, index: int) -> bool:
        """Whether this is one of the first two directional outer stations."""
        if self.mode != StrategyMode.B_PLUS or index <= 0 or not self._ring_order_known():
            return False
        return self._ring_order_rank(index) < self.config.early_full_outer_scan_count

    def _ring_order_known(self) -> bool:
        # The helper is intentionally kept separate so B never silently gains
        # B+ discovery scans before a ring direction exists.
        return bool(getattr(self, "_active_ring_order", ()))

    def _ring_order_rank(self, index: int) -> int:
        return list(self._active_ring_order).index(index)

    def _scan_channels_for_index(
        self, index: int, channels: dict[int, ChannelState]
    ) -> list[int]:
        if self._is_early_broad_outer_index(index):
            point = self.fixed_points[index]
            return [
                channel for channel in self.config.channels
                if not channels[channel].complete
                and not channels[channel].has_measured_at(point)
            ]
        return [
            channel
            for channel in self.config.channels
            if channels[channel].phase == ChannelPhase.UNKNOWN
            and index not in channels[channel].certified_zone_indices
        ]

    def _co_measurement_action(
        self, robot: RobotState, channels: dict[int, ChannelState]
    ) -> Action | None:
        """Reuse a completed dedicated-measurement stop for up to two channels.

        This is deliberately armed only by a supplementary primary measurement.
        It cannot turn every ring position into an uncontrolled all-channel scan.
        """
        if robot.co_measurement_position is None:
            return None
        if math.dist(robot.position, robot.co_measurement_position) > 1.0e-6:
            robot.co_measurement_position = None
            robot.co_measurement_reconnect_point = None
            robot.co_measurements_at_stop = 0
            return None
        if robot.co_measurements_at_stop >= self.config.max_co_measured_active_channels_per_stop:
            robot.co_measurement_position = None
            robot.co_measurement_reconnect_point = None
            return None
        reconnect = robot.co_measurement_reconnect_point
        remaining_points = [self.fixed_points[index] for index in self._remaining_ring_indices(robot)]
        options: list[tuple[tuple[float, ...], ChannelState, Any, float]] = []
        for state in channels.values():
            if (
                state.phase not in ACTIVE_PHASES
                or state.clear_committed
                or state.region_mask is None
                or state.direction_count >= self.config.max_direction_observations_per_channel
                or self._appointment_classification(state, robot) != "DIRECTIONAL_RESCUE"
                or state.has_measured_at(robot.position)
            ):
                continue
            ranked = self.planner.rank_measure_candidates(
                state,
                robot,
                remaining_points,
                reconnect,
                None,
                None,
                retained_points=[(robot.position, "co_measure_reuse")],
                guaranteed_only=True,
                directional_rescue=True,
            )
            candidate = next(
                (item for item in ranked if math.dist(item.point, robot.position) <= 1.0e-6),
                None,
            )
            if candidate is None:
                continue
            saving = candidate.direct_clear_baseline_s - candidate.estimated_total_s
            reduction = self._appointment_reduction(
                state, candidate, max(1, len(self.grid._cover_offsets()))
            )
            priority = (
                -float(state.direction_count < 2),
                -float(self._appointment_classification(state, robot) == "LARGE_UNCERTAIN"),
                -float(reduction),
                -saving,
                float(candidate.estimated_best_s),
                float(state.channel),
            )
            options.append((priority, state, candidate, saving))
        if not options:
            robot.co_measurement_position = None
            robot.co_measurement_reconnect_point = None
            return None
        priority, state, candidate, saving = min(options, key=lambda item: item[0])
        return Action(
            ActionKind.MEASURE,
            self._request_id(f"co-measure-c{state.channel}"),
            robot.position,
            state.channel,
            reason="co-measure high-value active channel at dedicated stop",
            metadata={
                "supplementary": True,
                "co_measurement": True,
                "optional_extra": True,
                "reconnect_point": reconnect,
                "co_measurement_saving_s": saving,
                "co_measurement_clear_count_reduction": self._appointment_reduction(
                    state, candidate, max(1, len(self.grid._cover_offsets()))
                ),
                "candidate_source": candidate.source,
                "candidate_guaranteed_reception": candidate.guaranteed_reception,
                "candidate_worst_branch": candidate.worst_branch,
                "candidate_branch_estimates_s": candidate.branch_estimates_s,
                "direction_count_before": state.direction_count,
                "directional_rescue": True,
                "ring_direction": robot.ring_direction,
                "outer_segment": robot.current_outer_segment,
            },
        )

    def _zero_move_opportunity(
        self,
        robot: RobotState,
        channels: dict[int, ChannelState],
        next_point: Point | None,
        remaining_points: list[Point],
        next_channel: int | None,
    ) -> Action | None:
        if robot.optional_measurements_at_stop >= self.config.max_optional_measurements_per_stop:
            return None
        zones_here = certified_responsibility_zones(robot.position, self.config)
        unknown = [
            state.channel
            for state in channels.values()
            if state.phase == ChannelPhase.UNKNOWN
            and not state.has_measured_at(robot.position)
            and bool(zones_here - state.certified_zone_indices)
        ]
        ordered_unknown = ordered_channels(unknown, robot.current_channel, next_channel)
        if ordered_unknown:
            channel = ordered_unknown[0]
            return Action(
                ActionKind.MEASURE,
                self._request_id(f"zero-zone-c{channel}"),
                robot.position,
                channel,
                reason="zero-move measurement can certify an unfinished responsibility zone",
                metadata={
                    "zero_move": True,
                    "optional_extra": True,
                    "candidate_zone_indices": sorted(zones_here),
                },
            )

        # Active-channel measurements during the ring are dispatched exclusively
        # through revalidated forward appointments.  This prevents a zero-move
        # convenience action from bypassing the agreed one-direction route rule.
        return None

    def _best_local_insertion(
        self,
        robot: RobotState,
        channels: dict[int, ChannelState],
        next_point: Point,
        remaining_points: list[Point],
        next_channel: int | None,
    ) -> Action | None:
        options: list[tuple[float, Action]] = []
        active_states = [
            state
            for state in channels.values()
            if state.phase in ACTIVE_PHASES
            and state.region_mask is not None
            and (
                not self._is_ring_active(robot)
                or self._has_current_segment_appointment(state, robot)
            )
        ]
        active_states.sort(
            key=lambda state: (
                -int(state.waiting_outer_segments >= self.config.forced_wait_segments),
                -int(state.clear_committed),
                -state.waiting_outer_segments,
                state.committed_order if state.committed_order is not None else math.inf,
                self.grid.distance_to_region_segment(state.region_mask, robot.position, next_point),
                state.channel,
            )
        )
        for state in active_states[: self.config.max_scored_channels_per_insertion_decision]:
            ranked = []
            if (
                not state.clear_committed
                and not self._is_ring_active(robot)
                and state.direction_count
                < self.config.max_direction_observations_per_channel
                and state.consecutive_low_gain_count
                < self.config.max_consecutive_low_gain_measurements
            ):
                ranked = self.planner.rank_measure_candidates(
                    state,
                    robot,
                    remaining_points,
                    next_point,
                    None,
                    next_channel,
                    guaranteed_only=True,
                    directional_rescue=self._appointment_classification(state, robot) == "DIRECTIONAL_RESCUE",
                )
                if self._last_planning_snapshot:
                    audit = self._last_planning_snapshot.setdefault("candidate_evaluations", {})
                    audit[str(state.channel)] = self.planner.last_candidate_evaluations.get(
                        (state.channel, state.region_version), []
                    )
            if ranked:
                candidate = ranked[0]
                delta = candidate.estimated_total_s - candidate.direct_clear_baseline_s
                if (
                    candidate.decision_band == "STRONG_ACCEPT"
                    and delta <= -self.config.gain_buffer_s + 1.0e-9
                ):
                    before_count = self.planner.fast_clear_count_proxy(state.region_mask)
                    before_time = self.planner.fast_clear_time_proxy(state.region_mask, candidate.point)
                    options.append(
                        (
                            delta,
                            Action(
                                ActionKind.MEASURE,
                                self._request_id(f"local-measure-c{state.channel}"),
                                candidate.point,
                                state.channel,
                                reason="dedicated supplement clears the six-second benefit buffer",
                                metadata={
                                    "local_insertion": True,
                                    "supplementary": True,
                                    "dedicated_measurement": True,
                                    "before_clear_count": before_count,
                                    "before_clear_time_s": before_time,
                                    "reconnect_point": next_point,
                                    "estimated_delta_s": delta,
                                    "decision_band": candidate.decision_band,
                                    "candidate_source": candidate.source,
                                    "candidate_guaranteed_reception": candidate.guaranteed_reception,
                                    "candidate_worst_branch": candidate.worst_branch,
                                    "candidate_branch_estimates_s": candidate.branch_estimates_s,
                                    "candidate_branch_remaining_cells": candidate.branch_remaining_cells,
                                    "candidate_forward_preferred": candidate.forward_preferred,
                                    "candidate_forward_gap_s": candidate.forward_gap_s,
                                    "ring_direction": robot.ring_direction,
                                    "outer_segment": robot.current_outer_segment,
                                    "outer_opportunity": self._is_outer_opportunity(state),
                                    "active_measurement_count_before": state.active_measurement_count,
                                    "direction_count_before": state.direction_count,
                                    "directional_rescue": self._appointment_classification(state, robot) == "DIRECTIONAL_RESCUE",
                                },
                            ),
                        )
                    )
            clear_option = self.planner.best_clear_insertion_delta(state, robot, next_point)
            mec_trial = self.planner.mec_trial_clear_delta(state, robot, next_point)
            if mec_trial is not None and (
                clear_option is None or mec_trial[0] < clear_option[0]
            ):
                clear_option = mec_trial
            if clear_option is not None:
                delta, point = clear_option
                outside = self.grid.region_extends_outside_fallback_hexagon(state.region_mask)
                nearest_outer = (
                    self.grid.nearest_outer_point_index(state.region_mask) if outside else None
                )
                at_nearest_opportunity = (
                    nearest_outer is not None
                    and math.dist(robot.position, self.fixed_points[nearest_outer]) <= 1.0e-6
                )
                opportunity_value = -delta
                forced_review = state.waiting_outer_segments >= self.config.forced_wait_segments
                opportunity_trigger = outside and at_nearest_opportunity and (
                    opportunity_value >= self.config.gain_buffer_s - 1.0e-9
                )
                accept_clear = (
                    delta <= -self.config.gain_buffer_s + 1.0e-9
                    or opportunity_trigger
                )
                if not accept_clear:
                    continue
                options.append(
                    (
                        delta,
                        Action(
                            ActionKind.CLEAR,
                            self._request_id(f"local-clear-c{state.channel}"),
                            point,
                            state.channel,
                            reason=(
                                "outside-hexagon nearest opportunity"
                                if opportunity_trigger
                                else "opportunistic clear clears the six-second benefit buffer"
                            ),
                            metadata={
                                "local_insertion": True,
                                "estimated_delta_s": delta,
                                "commit_clear": True,
                                "reconnect_point": next_point,
                                "outside_hexagon": outside,
                                "nearest_outer_index": nearest_outer,
                                "at_nearest_opportunity": at_nearest_opportunity,
                                "opportunity_value_s": opportunity_value,
                                "waiting_outer_segments": state.waiting_outer_segments,
                                "forced_review": forced_review,
                                "ring_direction": robot.ring_direction,
                                "outer_segment": robot.current_outer_segment,
                            },
                        ),
                    )
                )
        if self._last_planning_snapshot:
            self._last_planning_snapshot["task_pool"] = [
                {
                    "kind": action.kind.value,
                    "channel": action.channel,
                    "incremental_delta_s": delta,
                    "reason": action.reason,
                }
                for delta, action in sorted(options, key=lambda item: item[0])
            ]
        return min(options, key=lambda item: item[0])[1] if options else None

    def _refresh_tail_cluster_order(
        self, robot: RobotState, feasible: list[ChannelState]
    ) -> None:
        """Build a small look-ahead route over certified-clear channel clusters."""
        pending = {state.channel: state for state in feasible}
        cursor = robot.position
        order: list[int] = []
        records: list[dict[str, Any]] = []
        while pending:
            options: list[tuple[float, float, int, ChannelState, Any, Point]] = []
            for state in pending.values():
                plan = self.planner.clear_plan(state, cursor)
                exit_point = plan.points[-1]
                next_access = 0.0
                if len(pending) > 1:
                    next_access = min(
                        math.dist(
                            exit_point,
                            self.planner.clear_plan(other, exit_point).points[0],
                        )
                        / self.config.speed_mps
                        for other in pending.values()
                        if other.channel != state.channel
                    )
                options.append((plan.upper_time_s + next_access, plan.upper_time_s, state.channel, state, plan, exit_point))
            _, _, _, chosen, plan, exit_point = min(options, key=lambda item: (item[0], item[1], item[2]))
            order.append(chosen.channel)
            records.append(
                {
                    "order_position": len(order),
                    "channel": chosen.channel,
                    "entry_point": plan.points[0],
                    "exit_point": exit_point,
                    "access_s": math.dist(cursor, plan.points[0]) / self.config.speed_mps,
                    "local_upper_time_s": plan.upper_time_s,
                    "clear_point_count": len(plan.points),
                    "tail_reason": "no unexpired forward appointment remained at ring completion",
                }
            )
            cursor = exit_point
            del pending[chosen.channel]
        self._tail_cluster_order = order
        self._tail_cluster_history.append({"rebuilt_at_s": robot.virtual_time_s, "clusters": records})
        self._tail_clusters_dirty = False
        self._last_planning_snapshot["tail_clusters"] = records

    def _post_search_action(self, robot: RobotState, unresolved: list[ChannelState]) -> Action:
        feasible = [state for state in unresolved if state.region_mask is not None]
        if len(feasible) != len(unresolved):
            missing = [state.channel for state in unresolved if state.region_mask is None]
            raise RuntimeError(f"active channels without safe regions: {missing}")
        feasible_by_channel = {state.channel: state for state in feasible}
        if self._tail_clusters_dirty or set(self._tail_cluster_order) != set(feasible_by_channel):
            self._refresh_tail_cluster_order(robot, feasible)
        scheduled_channel = next(
            (channel for channel in self._tail_cluster_order if channel in feasible_by_channel),
            None,
        )
        if scheduled_channel is None:
            raise RuntimeError("tail clear cluster order contains no unresolved channel")
        scheduled_state = feasible_by_channel[scheduled_channel]
        deep_allowed = not self._deep_scoring_disabled and not self._force_fallback
        candidate_states = [scheduled_state]
        clear_options: list[tuple[float, ChannelState, Action]] = []
        measure_options: list[tuple[float, ChannelState, Action]] = []
        candidate_logs: dict[str, list[dict[str, Any]]] = {}

        for state in candidate_states:
            plan = self.planner.clear_plan(state, robot.position)
            clear_action = self._certified_clear_action(robot, state, plan, "global certified clear task")
            clear_options.append((plan.upper_time_s, state, clear_action))
            rescue_allowed = (
                self.enable_optimization
                and deep_allowed
                and not state.clear_committed
                and state.direction_count < self.config.max_direction_observations_per_channel
                and self._appointment_classification(state, robot) == "DIRECTIONAL_RESCUE"
                and state.rescue_measurement_count < self.config.max_rescue_measurements_per_channel
            )
            if not rescue_allowed:
                continue
            ranked = self.planner.rank_measure_candidates(
                state,
                robot,
                [],
                None,
                None,
                guaranteed_only=True,
                directional_rescue=True,
            )
            candidate_logs[str(state.channel)] = self.planner.last_candidate_evaluations.get(
                (state.channel, state.region_version), []
            )
            ranked = [
                candidate
                for candidate in ranked
                if candidate.guaranteed_reception
                and candidate.decision_band == "STRONG_ACCEPT"
                and candidate.direct_clear_baseline_s - candidate.estimated_total_s
                >= self.config.gain_buffer_s - 1.0e-9
            ]
            if not ranked:
                state.directional_rescue_history.append({
                    "event": "rescue_unavailable_fallback_certified_clear",
                    "region_version": state.region_version,
                    "direction_count": state.direction_count,
                    "safe_candidate_count": 0,
                    "certified_clear_upper_time_s": plan.upper_time_s,
                })
                continue
            candidate = ranked[0]
            measure_options.append(
                (
                    candidate.estimated_total_s,
                    state,
                    self._supplement_action(robot, state, candidate, is_rescue=True),
                )
            )
            state.directional_rescue_history.append({
                "event": "tail_directional_rescue_candidate",
                "region_version": state.region_version,
                "direction_count": state.direction_count,
                "safe_candidate_count": len(ranked),
                "certified_clear_upper_time_s": plan.upper_time_s,
                "estimated_saving_s": candidate.direct_clear_baseline_s - candidate.estimated_total_s,
            })

        if candidate_logs:
            self._last_planning_snapshot["candidate_evaluations"] = candidate_logs
        options: list[tuple[float, ChannelState, Action]] = clear_options + measure_options
        if not options:
            raise RuntimeError("post-search task pool is empty")
        _, chosen_state, action = min(options, key=lambda item: (item[0], item[1].channel))
        if action.kind == ActionKind.CLEAR:
            self._commit_clear(chosen_state)
        return action

    def _supplement_action(
        self, robot: RobotState, state: ChannelState, candidate: Any, *, is_rescue: bool) -> Action:
        before_count = self.planner.fast_clear_count_proxy(state.region_mask)
        before_time = self.planner.fast_clear_time_proxy(state.region_mask, candidate.point)
        return Action(
            ActionKind.MEASURE,
            self._request_id(f"{'rescue' if is_rescue else 'finish-measure'}-c{state.channel}"),
            candidate.point,
            state.channel,
            reason=("guaranteed-reception rescue measurement before large certified clear" if is_rescue else "supplement is faster than certified direct clear plan"),
            metadata={
                "supplementary": True,
                "dedicated_measurement": math.dist(robot.position, candidate.point) > 1.0e-6 and not is_rescue,
                "rescue_measurement": is_rescue,
                "before_clear_count": before_count,
                "before_clear_time_s": before_time,
                "reconnect_point": None,
                "estimated_total_s": candidate.estimated_total_s,
                "decision_band": candidate.decision_band,
                "candidate_source": candidate.source,
                "candidate_guaranteed_reception": candidate.guaranteed_reception,
                "candidate_worst_branch": candidate.worst_branch,
                "candidate_branch_estimates_s": candidate.branch_estimates_s,
                "candidate_branch_remaining_cells": candidate.branch_remaining_cells,
                "candidate_forward_preferred": candidate.forward_preferred,
                "candidate_forward_gap_s": candidate.forward_gap_s,
                "ring_direction": robot.ring_direction,
                "outer_segment": robot.current_outer_segment,
                "outer_opportunity": self._is_outer_opportunity(state),
                "active_measurement_count_before": state.active_measurement_count,
                "direction_count_before": state.direction_count,
                "directional_rescue": is_rescue,
            },
        )

    def _tail_cluster_metadata(self, channel: int) -> dict[str, Any]:
        if not self._tail_cluster_history:
            return {}
        clusters = self._tail_cluster_history[-1].get("clusters", [])
        record = next((item for item in clusters if item.get("channel") == channel), None)
        if record is None:
            return {}
        return {
            "tail_cluster_order_position": record["order_position"],
            "tail_cluster_entry_point": record["entry_point"],
            "tail_cluster_exit_point": record["exit_point"],
            "tail_cluster_access_s": record["access_s"],
            "tail_cluster_reason": record["tail_reason"],
        }

    def _certified_clear_action(
        self, robot: RobotState, state: ChannelState, plan: Any, reason: str,
        reconnect_point: Point | None = None, *, session_continuation: bool = False,
    ) -> Action:
        if not plan.points:
            raise RuntimeError(f"channel {state.channel}: committed clear plan is empty")
        plan_record = {
            "region_version": state.region_version,
            "point_count": len(plan.points),
            "route_length_m": plan.route_length_m,
            "upper_time_s": plan.upper_time_s,
            "certificate": plan.certificate,
        }
        if not state.clear_plan_history or state.clear_plan_history[-1] != plan_record:
            state.clear_plan_history.append(plan_record)
        return Action(
            ActionKind.CLEAR,
            self._request_id(f"certified-clear-c{state.channel}"),
            plan.points[0],
            state.channel,
            reason=reason,
            metadata={
                "clear_committed": True,
                "clear_certificate_reused": False,
                "certified_plan_point_count": len(plan.points),
                "certified_worst_time_s": plan.upper_time_s,
                "reconnect_point": reconnect_point,
                "clear_task_access_s": math.dist(robot.position, plan.points[0]) / self.config.speed_mps,
                "clear_task_route_length_m": plan.route_length_m,
                "clear_task_reentered_pool": False,
                "clear_session_continuation": session_continuation,
                "clear_session_remaining_points": len(plan.points),
                "ring_direction": robot.ring_direction,
                "outer_segment": robot.current_outer_segment,
                "outer_opportunity": self._is_outer_opportunity(state),
                **self._tail_cluster_metadata(state.channel),
            },
        )

    def _continue_clear_session(
        self, robot: RobotState, channels: dict[int, ChannelState]
    ) -> Action | None:
        """Continue the same certified-clear channel after a failed point.

        The route is regenerated against the updated region from the current
        position, so no obsolete point is mechanically reused; the channel is
        nevertheless not thrown back into a cross-channel route lottery.
        """
        session = self._clear_session
        if session is None:
            return None
        state = channels[session.channel]
        if state.complete or state.phase == ChannelPhase.INCONSISTENT or state.region_mask is None:
            self._clear_session = None
            return None
        plan = self.planner.clear_plan(state, robot.position, session.reconnect_point)
        self._clear_session = ClearRouteSession(
            channel=state.channel,
            remaining_points=list(plan.points[1:]),
            reconnect_point=session.reconnect_point,
            region_version=state.region_version,
        )
        return self._certified_clear_action(
            robot, state, plan, "certified clear session continuation",
            session.reconnect_point, session_continuation=True,
        )

    def apply_response(
        self,
        action: Action,
        response: dict[str, Any],
        robot: RobotState,
        channels: dict[int, ChannelState],
    ) -> None:
        if response.get("accepted") is not True:
            return
        self.action_count += 1
        if "virtual_time_s" in response:
            robot.virtual_time_s = float(response["virtual_time_s"])

        if action.kind == ActionKind.ENTER:
            robot.entered = True
            robot.entered_real_monotonic = time.monotonic()
            robot.remaining_real_duration_s = float(response["remaining_real_duration_s"])
            return
        if action.kind == ActionKind.EXIT:
            robot.exited = True
            return
        if action.position is None or action.channel is None:
            raise RuntimeError("accepted position action lacks local position/channel")
        previous_position = robot.position
        previous_channel = robot.current_channel
        robot.time_breakdown["movement_s"] += (
            math.dist(previous_position, action.position) / self.config.speed_mps
        )
        if action.kind == ActionKind.MEASURE:
            robot.time_breakdown["switch_s"] += (
                self.config.switch_time_s if previous_channel != action.channel else 0.0
            )
            robot.time_breakdown["measure_s"] += self.config.measure_time_s
        elif response.get("clear_result") == "success":
            robot.time_breakdown["clear_success_s"] += self.config.clear_success_time_s
        elif response.get("clear_result") == "no_target_in_range":
            robot.time_breakdown["clear_failure_s"] += self.config.clear_failure_time_s
        if (
            robot.optional_measurement_position is None
            or math.dist(robot.optional_measurement_position, action.position) > 1.0e-6
        ):
            robot.optional_measurement_position = action.position
            robot.optional_measurements_at_stop = 0
        robot.position = action.position
        state = channels[action.channel]
        was_active_before_measure = state.phase in ACTIVE_PHASES

        if action.kind == ActionKind.MEASURE:
            robot.current_channel = action.channel
            if action.metadata.get("optional_extra"):
                robot.optional_measurements_at_stop += 1
            if action.metadata.get("co_measurement"):
                robot.co_measurements_at_stop += 1
            if action.metadata.get("dedicated_measurement"):
                state.dedicated_measurement_count += 1
            if action.metadata.get("rescue_measurement"):
                state.rescue_measurement_count += 1
            state.measured_positions.append(action.position)
            result = response.get("measure_result")
            if result in {"direction", "near"} or was_active_before_measure:
                state.active_measurement_count += 1
            fixed_index = action.metadata.get("fixed_search_index")
            if result == "direction":
                bearing = float(response["svd_deg"])
                evidence = DirectionEvidence(action.position, bearing)
                state.direction_observations.append(evidence)
                update = self.grid.update_direction(state, evidence)
                self._record_region_update(state, "direction", update)
                if update.accepted:
                    self._invalidate_appointments(state, "direction_region_updated")
                    self._tail_clusters_dirty = True
                    state.phase = (
                        ChannelPhase.CLEAR_COMMITTED
                        if state.clear_committed
                        else ChannelPhase.ACTIVE
                    )
                    self._mark_progress(state, robot, fixed_index)
                else:
                    state.phase = ChannelPhase.INCONSISTENT
                    self.anomalies.append(f"channel {state.channel}: {update.reason}")
            elif result == "near":
                state.phase = ChannelPhase.ACTIVE
                state.near_pending = True
                self._mark_progress(state, robot, fixed_index)
                self._urgent_clear.appendleft(state.channel)
            elif result == "no_signal":
                state.no_signal_points.append(action.position)
                if fixed_index is not None:
                    state.fixed_no_signal_indices.add(int(fixed_index))
                if self.use_zone_scheduler and state.phase == ChannelPhase.UNKNOWN:
                    for zone_index in certified_responsibility_zones(action.position, self.config):
                        state.certified_zone_indices.add(zone_index)
                        state.zone_certificate_points.setdefault(zone_index, action.position)
                    if set(range(7)).issubset(state.certified_zone_indices):
                        state.phase = ChannelPhase.EXCLUDED
                if state.phase in ACTIVE_PHASES and state.region_mask is not None:
                    update = self.grid.update_no_signal(state, action.position)
                    self._record_region_update(state, "no_signal", update)
                    if not update.accepted:
                        state.phase = ChannelPhase.INCONSISTENT
                        self.anomalies.append(f"channel {state.channel}: {update.reason}")
                    else:
                        self._invalidate_appointments(state, "no_signal_region_updated")
                        self._tail_clusters_dirty = True
                        self._mark_progress(state, robot, fixed_index)
            else:
                raise RuntimeError(f"unknown measure_result: {result!r}")
            if action.metadata.get("supplementary"):
                self._record_supplement_gain(state, action, str(result), robot)
            if action.metadata.get("directional_rescue"):
                state.directional_rescue_history.append({
                    "event": "directional_rescue_result",
                    "region_version_after": state.region_version,
                    "direction_count_after": state.direction_count,
                    "result": str(result),
                    "guaranteed_reception": bool(action.metadata.get("candidate_guaranteed_reception")),
                })
            if action.metadata.get("supplementary") and not action.metadata.get("co_measurement"):
                # Keep the movement already paid for, but only for a bounded
                # number of independently worthwhile ACTIVE channels.
                robot.co_measurement_position = action.position
                robot.co_measurement_reconnect_point = action.metadata.get("reconnect_point")
                robot.co_measurements_at_stop = 0
            return

        result = response.get("clear_result")
        if action.metadata.get("commit_clear") or action.metadata.get("clear_committed"):
            self._commit_clear(state)
        if result == "success":
            state.phase = ChannelPhase.CLEARED
            state.near_pending = False
            state.region_mask = None
            self._invalidate_appointments(state, "channel_cleared")
            self._tail_clusters_dirty = True
            self._mark_progress(state, robot, None)
            state.skipped_clear_decisions = 0
            if self._clear_session is not None and self._clear_session.channel == state.channel:
                self._clear_session = None
        elif result == "no_target_in_range":
            state.clear_failure_points.append(action.position)
            # The latest口径 requires a fresh region/route calculation after every
            # failed clear; an old point queue must never be reused mechanically.
            if action.metadata.get("near"):
                message = f"channel {state.channel}: near followed by failed clear"
                state.phase = ChannelPhase.INCONSISTENT
                state.anomaly_log.append(message)
                self.anomalies.append(message)
            elif state.region_mask is not None:
                update = self.grid.update_clear_failure(state, action.position)
                self._record_region_update(state, "clear_failure", update)
                if not update.accepted:
                    state.phase = ChannelPhase.INCONSISTENT
                    self.anomalies.append(f"channel {state.channel}: {update.reason}")
                elif state.phase != ChannelPhase.INCONSISTENT:
                    self._invalidate_appointments(state, "clear_failure_region_updated")
                    self._tail_clusters_dirty = True
                    state.phase = (
                        ChannelPhase.CLEAR_COMMITTED
                        if state.clear_committed
                        else ChannelPhase.ACTIVE
                    )
                    self._mark_progress(state, robot, None)
                    if action.metadata.get("commit_clear") or action.metadata.get("clear_committed"):
                        reconnect_raw = action.metadata.get("reconnect_point")
                        reconnect = tuple(reconnect_raw) if reconnect_raw is not None else None
                        self._clear_session = ClearRouteSession(
                            channel=state.channel,
                            remaining_points=[],
                            reconnect_point=reconnect,
                            region_version=state.region_version,
                        )
        else:
            raise RuntimeError(f"unknown clear_result: {result!r}")

    def _record_supplement_gain(
        self,
        state: ChannelState,
        action: Action,
        result: str,
        robot: RobotState,
    ) -> None:
        if result == "near":
            state.consecutive_low_gain_count = 0
            return
        before_count = int(action.metadata.get("before_clear_count", 0))
        before_time = float(action.metadata.get("before_clear_time_s", math.inf))
        reconnect_raw = action.metadata.get("reconnect_point")
        reconnect = tuple(reconnect_raw) if reconnect_raw is not None else None
        obvious_gain = False
        if state.region_mask is not None and state.phase in ACTIVE_PHASES:
            after_count = self.planner.fast_clear_count_proxy(state.region_mask)
            after_time = self.planner.fast_clear_time_proxy(state.region_mask, robot.position)
            obvious_gain = (
                after_count == 1
                or (before_count > 0 and after_count < before_count)
                or before_time - after_time >= self.config.gain_buffer_s - 1.0e-9
            )
        if obvious_gain:
            state.consecutive_low_gain_count = 0
        else:
            state.consecutive_low_gain_count += 1
        # Reaching the ordinary supplement limit does not itself lock this
        # channel.  The post-search task pool first gets one chance to compare
        # a guaranteed-reception rescue measurement against a large certified
        # clear plan; it commits only after that comparison rejects rescue.

    def _commit_clear(self, state: ChannelState) -> None:
        if state.clear_committed:
            state.phase = ChannelPhase.CLEAR_COMMITTED
            return
        self._commit_counter += 1
        state.commit_clear(self._commit_counter)

    @staticmethod
    def _mark_progress(
        state: ChannelState, robot: RobotState, fixed_index: int | None
    ) -> None:
        segment = robot.completed_outer_segments
        if fixed_index is not None and int(fixed_index) > 0:
            segment += 1
        state.last_progress_outer_segment = max(state.last_progress_outer_segment, segment)
        state.waiting_outer_segments = 0

    def _update_waiting_segments(
        self, robot: RobotState, channels: dict[int, ChannelState]
    ) -> None:
        for state in channels.values():
            if not state.active:
                continue
            state.waiting_outer_segments = max(
                0, robot.completed_outer_segments - state.last_progress_outer_segment
            )

    def _refresh_planning_snapshot(
        self,
        robot: RobotState,
        channels: dict[int, ChannelState],
        remaining_indices: dict[int, Point],
        scan_by_index: dict[int, list[int]],
    ) -> None:
        order = sorted(remaining_indices)
        estimate = self.planner.estimate_state(
            robot,
            channels,
            order,
            self.fixed_points,
            scan_by_index,
        )
        estimate.update(
            {
                "mode": self.mode.value,
                "safety_mode": robot.safety_mode.value,
                "remaining_search_indices": order,
                "completed_outer_segments": robot.completed_outer_segments,
                "ring_direction": robot.ring_direction,
                "ring_order": list(robot.ring_order),
                "current_outer_segment": robot.current_outer_segment,
                "appointment_measurements_this_segment": robot.appointment_measurements_since_search,
                "directional_rescues_this_segment": robot.directional_rescues_since_search,
                "measurement_appointments": {
                    str(state.channel): [
                        {
                            "appointment_id": item.appointment_id,
                            "point": item.point,
                            "source": item.source,
                            "anchor_segment": item.anchor_segment,
                            "anchor_index": item.anchor_index,
                            "region_version": item.region_version,
                            "guaranteed_reception": item.guaranteed_reception,
                            "estimated_saving_s": item.estimated_saving_s,
                            "estimated_clear_count_reduction": item.estimated_clear_count_reduction,
                            "classification": item.classification,
                            "status": item.status,
                            "status_reason": item.status_reason,
                        }
                        for item in state.appointments
                    ]
                    for state in channels.values()
                    if state.appointments
                },
                "tail_cluster_history": list(self._tail_cluster_history),
                "clear_route_session": (
                    None
                    if self._clear_session is None
                    else {
                        "channel": self._clear_session.channel,
                        "remaining_points": len(self._clear_session.remaining_points),
                        "reconnect_point": self._clear_session.reconnect_point,
                        "region_version": self._clear_session.region_version,
                    }
                ),
                "deep_scoring_disabled": self._deep_scoring_disabled,
            }
        )
        if self.decision_trace and "H_after_s" not in self.decision_trace[-1]:
            self.decision_trace[-1]["H_after_s"] = estimate["H_hat_s"]
        self._last_planning_snapshot = estimate

    def _record_macro_decision(
        self, insertion: Action | None, next_index: int
    ) -> None:
        pool = list(self._last_planning_snapshot.get("task_pool", []))
        default = {
            "kind": "FALLBACK",
            "search_index": next_index,
            "incremental_delta_s": 0.0,
        }
        pool.append(default)
        if insertion is None:
            chosen: dict[str, Any] = default
        else:
            chosen = {
                "kind": insertion.kind.value,
                "channel": insertion.channel,
                "position": insertion.position,
                "reason": insertion.reason,
                "metadata": insertion.metadata,
            }
        self.decision_trace.append(
            {
                "H_before_s": self._last_planning_snapshot.get("H_hat_s"),
                "U_safe_before_s": self._last_planning_snapshot.get("U_safe_s"),
                "task_pool": pool,
                "candidate_evaluations": self._last_planning_snapshot.get("candidate_evaluations", {}),
                "chosen": chosen,
            }
        )

    def _update_safety_mode(self, robot: RobotState) -> None:
        elapsed = robot.real_elapsed_s
        if elapsed >= self.config.deep_scoring_stop_s:
            self._deep_scoring_disabled = True
        average_action = (
            robot.real_action_time_s / robot.real_action_count
            if robot.real_action_count
            else 0.0
        )
        remaining_calls = int(
            self._last_planning_snapshot.get("estimated_current_action_calls", 0)
        )
        projected_remaining = remaining_calls * max(average_action, 0.001)
        if elapsed >= self.config.emergency_elapsed_s:
            robot.safety_mode = SafetyMode.EMERGENCY
        elif (
            elapsed >= self.config.safe_elapsed_s
            or elapsed
            + self.config.safe_projection_factor * projected_remaining
            >= self.config.safe_projection_deadline_s
            or (
                robot.real_remaining_s is not None
                and robot.real_remaining_s <= self.config.real_time_guard_s
            )
        ):
            robot.safety_mode = SafetyMode.SAFE
        if robot.safety_mode != SafetyMode.NORMAL:
            self._force_fallback = True

    @staticmethod
    def _record_region_update(state: ChannelState, kind: str, update: Any) -> None:
        state.region_update_history.append(
            {
                "kind": kind,
                "accepted": bool(update.accepted),
                "before_cells": int(update.before_cells),
                "after_cells": int(update.after_cells),
                "reason": str(update.reason),
            }
        )

    def _finalize_exclusions(self, channels: dict[int, ChannelState]) -> None:
        required = set(range(len(self.fixed_points)))
        for state in channels.values():
            if state.phase != ChannelPhase.UNKNOWN:
                continue
            evidence = (
                state.certified_zone_indices
                if self.use_zone_scheduler
                else state.fixed_no_signal_indices
            )
            if required.issubset(evidence):
                state.phase = ChannelPhase.EXCLUDED

    def is_complete(self, channels: dict[int, ChannelState]) -> bool:
        cleared = sum(state.phase == ChannelPhase.CLEARED for state in channels.values())
        return cleared >= self.config.source_count_upper or all(state.complete for state in channels.values())

    def completion_certificate(self, channels: dict[int, ChannelState]) -> dict[str, Any]:
        coverage = search_coverage_certificate(self.config)
        required = set(range(len(self.fixed_points)))
        per_channel: dict[str, Any] = {}
        valid = True
        for number, state in channels.items():
            if state.phase == ChannelPhase.CLEARED:
                entry = {"status": "CLEARED", "evidence": "clear_success"}
            elif state.phase == ChannelPhase.EXCLUDED:
                indices = sorted(state.fixed_no_signal_indices)
                if self.use_zone_scheduler:
                    zone_details = {
                        str(index): responsibility_zone_certificate(
                            index, state.zone_certificate_points[index], self.config
                        )
                        for index in sorted(state.certified_zone_indices)
                        if index in state.zone_certificate_points
                    }
                    channel_valid = (
                        required.issubset(state.certified_zone_indices)
                        and all(str(index) in zone_details for index in required)
                        and all(bool(zone_details[str(index)]["covered"]) for index in required)
                    )
                else:
                    zone_details = {}
                    channel_valid = required.issubset(state.fixed_no_signal_indices) and bool(coverage["covered"])
                entry = {
                    "status": "EXCLUDED",
                    "fixed_no_signal_indices": indices,
                    "seven_point_coverage_valid": channel_valid,
                    "certified_zone_indices": sorted(state.certified_zone_indices),
                    "zone_certificates": zone_details,
                }
                valid &= channel_valid
            else:
                entry = {"status": state.phase.value, "evidence": "incomplete"}
                valid = False
            entry["region_updates"] = list(state.region_update_history)
            entry["clear_plans"] = list(state.clear_plan_history)
            entry["waiting_outer_segments"] = state.waiting_outer_segments
            entry["active_measurement_count"] = state.active_measurement_count
            entry["direction_count"] = state.direction_count
            entry["directional_rescue_history"] = list(state.directional_rescue_history)
            entry["outer_opportunity"] = self._is_outer_opportunity(state)
            entry["opportunity_classification"] = self._appointment_classification(state, RobotState())
            entry["measurement_appointments"] = [
                {
                    "appointment_id": item.appointment_id,
                    "point": item.point,
                    "source": item.source,
                    "anchor_segment": item.anchor_segment,
                    "anchor_index": item.anchor_index,
                    "region_version": item.region_version,
                    "guaranteed_reception": item.guaranteed_reception,
                    "estimated_saving_s": item.estimated_saving_s,
                    "estimated_clear_count_reduction": item.estimated_clear_count_reduction,
                    "classification": item.classification,
                    "status": item.status,
                    "status_reason": item.status_reason,
                }
                for item in state.appointments
            ]
            entry["appointment_history"] = list(state.appointment_history)
            per_channel[str(number)] = entry
        cleared = sum(state.phase == ChannelPhase.CLEARED for state in channels.values())
        valid = valid or cleared >= self.config.source_count_upper
        return {
            "valid": valid,
            "cleared_count": cleared,
            "early_stop_at_upper_bound": cleared >= self.config.source_count_upper,
            "search_coverage": coverage,
            "channels": per_channel,
        }

    def make_summary(
        self, robot: RobotState, channels: dict[int, ChannelState]
    ) -> RunSummary:
        cleared = sum(state.phase == ChannelPhase.CLEARED for state in channels.values())
        return RunSummary(
            mode=self.mode.value,
            cleared_count=cleared,
            channel_states={number: state.phase.value for number, state in channels.items()},
            completion_certificate=self.completion_certificate(channels),
            virtual_total_time_s=robot.virtual_time_s,
            average_clear_time_s=(robot.virtual_time_s / cleared if cleared else None),
            real_program_time_s=robot.real_elapsed_s,
            anomalies=self.anomalies + [item for state in channels.values() for item in state.anomaly_log],
            action_count=self.action_count,
            time_breakdown=dict(robot.time_breakdown),
            safety_mode=robot.safety_mode.value,
            planning_snapshot=dict(self._last_planning_snapshot),
            decision_trace=list(self.decision_trace),
        )


def new_channel_states(config: Q3Config) -> dict[int, ChannelState]:
    return {channel: ChannelState(channel) for channel in config.channels}
