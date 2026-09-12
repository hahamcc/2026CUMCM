from __future__ import annotations

import itertools
import math
import time
from dataclasses import replace
from typing import Any, Iterable, Sequence

import numpy as np

from .config import Q3Config
from .geometry import ConservativeGrid, minimum_enclosing_circle, route_length_from
from .models import ChannelPhase, ChannelState, ClearPlan, DirectionEvidence, MeasureCandidate, Point, RobotState


def shortest_open_search_order(current: Point, remaining: dict[int, Point]) -> list[int]:
    """Exact open-path order for at most six fixed outer search points."""
    keys = list(remaining)
    if len(keys) > 8:
        raise ValueError("fixed-search enumeration is only intended for a small point set")
    best_order: tuple[int, ...] = tuple(keys)
    best_length = math.inf
    for order in itertools.permutations(keys):
        points = [remaining[index] for index in order]
        length = route_length_from(current, points)
        if length < best_length:
            best_length = length
            best_order = order
    return list(best_order)


def ordered_channels(channels: Iterable[int], current: int, next_channel: int | None = None) -> list[int]:
    """Deterministic minimum-switch order for uniform one-second switches."""
    values = sorted(set(channels))
    result: list[int] = []
    if current in values:
        result.append(current)
        values.remove(current)
    if next_channel in values:
        values.remove(next_channel)
        tail = [int(next_channel)]
    else:
        tail = []
    result.extend(values)
    result.extend(tail)
    return result


def sequence_switch_count(current: int, sequence: Sequence[int], next_channel: int | None = None) -> int:
    count = 0
    previous = current
    for channel in sequence:
        count += int(channel != previous)
        previous = channel
    if next_channel is not None:
        count += int(previous != next_channel)
    return count


def weighted_open_search_order(
    current_position: Point,
    current_channel: int,
    remaining: dict[int, Point],
    scan_channels: dict[int, Sequence[int]],
    config: Q3Config,
) -> list[int]:
    """Exact open order for remaining fixed sites including scan/switch costs."""
    keys = list(remaining)
    if len(keys) > 8:
        raise ValueError("weighted fixed-search enumeration is limited to eight points")
    best_order: tuple[int, ...] = tuple(keys)
    best_cost = math.inf
    for order in itertools.permutations(keys):
        position = current_position
        channel = current_channel
        cost = 0.0
        for index in order:
            point = remaining[index]
            cost += math.dist(position, point) / config.speed_mps
            sequence = ordered_channels(scan_channels.get(index, ()), channel)
            cost += len(sequence) * config.measure_time_s
            cost += sequence_switch_count(channel, sequence) * config.switch_time_s
            if sequence:
                channel = sequence[-1]
            position = point
        if cost < best_cost - 1.0e-9 or (abs(cost - best_cost) <= 1.0e-9 and order < best_order):
            best_cost = cost
            best_order = order
    return list(best_order)


class RollingPlanner:
    def __init__(self, config: Q3Config, grid: ConservativeGrid):
        self.config = config
        self.grid = grid
        self._clear_cache: dict[
            tuple[int, int, float, float, float | None, float | None], ClearPlan
        ] = {}
        self.last_candidate_evaluations: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self.last_candidate_rejections: dict[tuple[int, int], list[dict[str, Any]]] = {}

    def clear_plan(
        self, channel: ChannelState, current: Point, reconnect: Point | None = None
    ) -> ClearPlan:
        if channel.region_mask is None:
            raise ValueError("active channel has no conservative region")
        reconnect_key: tuple[float | None, float | None] = (
            (round(reconnect[0], 6), round(reconnect[1], 6))
            if reconnect is not None
            else (None, None)
        )
        key = (
            channel.channel,
            channel.region_version,
            round(current[0], 6),
            round(current[1], 6),
            reconnect_key[0],
            reconnect_key[1],
        )
        if key not in self._clear_cache:
            self._clear_cache[key] = self.grid.build_clear_plan(
                channel.region_mask, current, reconnect
            )
        return self._clear_cache[key]

    def fast_clear_time_proxy(self, mask: np.ndarray, current: Point) -> float:
        """Cheap action-ranking proxy; never used as a completion certificate."""
        points = self.grid.center_points(mask)
        # In a candidate no-signal branch the conservative region may be empty.
        # That branch has no remaining certified-clear task; NumPy reductions on
        # an empty point array would otherwise abort route-direction selection.
        if len(points) == 0:
            return 0.0
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        center = ((float(min_x) + float(max_x)) / 2.0, (float(min_y) + float(max_y)) / 2.0)
        radius_proxy = math.hypot(float(max_x - min_x), float(max_y - min_y)) / 2.0
        if radius_proxy + self.config.cell_radius_m <= self.config.clear_radius_m + 1.0e-9:
            return math.dist(current, center) / self.config.speed_mps + self.config.clear_success_time_s
        cover_capacity = max(1, len(self.grid._cover_offsets()))
        estimated_count = max(1, math.ceil(len(points) / cover_capacity))
        access = math.dist(current, center) / self.config.speed_mps
        spatial_span = 2.0 * radius_proxy / self.config.speed_mps
        return access + spatial_span + self.config.clear_failure_time_s * (estimated_count - 1) + self.config.clear_success_time_s

    def fast_clear_count_proxy(self, mask: np.ndarray) -> int:
        """Deterministic ranking proxy; certified counts come only from build_clear_plan."""
        points = self.grid.center_points(mask)
        if len(points) == 0:
            return 0
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        radius_proxy = math.hypot(float(max_x - min_x), float(max_y - min_y)) / 2.0
        if radius_proxy + self.config.cell_radius_m <= self.config.clear_radius_m + 1.0e-9:
            return 1
        return max(1, math.ceil(len(points) / max(1, len(self.grid._cover_offsets()))))

    def generate_measure_candidates(
        self,
        channel: ChannelState,
        robot: RobotState,
        remaining_search_points: Iterable[Point],
        next_search_point: Point | None,
        retained_points: Sequence[tuple[Point, str]] = (),
        *,
        guaranteed_only: bool = False,
        directional_rescue: bool = False,
    ) -> list[tuple[Point, str, float]]:
        if channel.region_mask is None:
            return []
        points = self.grid.center_points(channel.region_mask)

        def detour(point: Point) -> float:
            if next_search_point is None:
                return math.dist(robot.position, point)
            return math.dist(robot.position, point) + math.dist(point, next_search_point) - math.dist(robot.position, next_search_point)

        self.last_candidate_rejections[(channel.channel, channel.region_version)] = []
        route_points: list[tuple[Point, str]] = [(robot.position, "current")]
        if next_search_point is not None:
            for fraction in (0.25, 0.5, 0.75):
                route_points.append(
                    (
                        (
                            robot.position[0] + fraction * (next_search_point[0] - robot.position[0]),
                            robot.position[1] + fraction * (next_search_point[1] - robot.position[1]),
                        ),
                        "forward_segment",
                    )
                )
            route_points.append((next_search_point, "next_outer"))
        fixed_points = [(point, "future_outer") for point in remaining_search_points]
        geometry_points: list[tuple[Point, str]] = []
        if channel.direction_observations:
            evidence = channel.direction_observations[-1]
            theta = math.radians(evidence.bearing_deg)
            forward = (math.cos(theta), math.sin(theta))
            lateral = (-forward[1], forward[0])
            # Q2-style local candidates.  These are only candidate generators:
            # Q3 still filters every one through its current reception proof
            # and its complete route-time evaluation.
            templates = (
                (500.0, 300.0), (600.0, 500.0), (700.0, 300.0),
                (700.0, 500.0), (800.0, 400.0), (800.0, 600.0),
                (900.0, 400.0), (1000.0, 500.0),
            )
            for along, offset in templates:
                for sign in (-1.0, 1.0):
                    geometry_points.append(
                        (
                            (
                                evidence.position[0] + along * forward[0] + sign * offset * lateral[0],
                                evidence.position[1] + along * forward[1] + sign * offset * lateral[1],
                            ),
                            "side_geometry",
                        )
                    )
        circle = minimum_enclosing_circle(points)
        step = self.config.supplement_grid_step_m
        mean = points.mean(axis=0)
        region_grid = (float(round(float(mean[0]) / step) * step), float(round(float(mean[1]) / step) * step))
        buckets = (
            # A deferred appointment is a real candidate, not a historical
            # suggestion.  Retaining it permits a direct, current-state
            # re-evaluation at its window instead of requiring it to be
            # regenerated by a changing candidate heuristic.
            (list(retained_points), len(retained_points)),
            (route_points, 5),
            (fixed_points, 1),
            (geometry_points, 10 if directional_rescue else 4),
            ([(circle.center, "mec_center")], 1),
            ([(region_grid, "region_grid")], 1),
        )
        unique: dict[tuple[float, float], tuple[Point, str]] = {}
        for bucket, quota in buckets:
            admitted = 0
            for point, source in sorted(bucket, key=lambda item: (detour(item[0]), math.dist(robot.position, item[0]))):
                if admitted >= quota:
                    break
                if not all(math.isfinite(value) and abs(value) <= self.config.coordinate_limit_m for value in point):
                    self.last_candidate_rejections[(channel.channel, channel.region_version)].append({"point": point, "source": source, "reason": "invalid_coordinate"})
                    continue
                key = (round(point[0], 6), round(point[1], 6))
                if key in unique or channel.has_measured_at(point):
                    self.last_candidate_rejections[(channel.channel, channel.region_version)].append({"point": point, "source": source, "reason": "duplicate_or_previously_measured"})
                    continue
                guaranteed = self.grid.point_guarantees_reception_with_history(channel, point)
                if guaranteed_only and not guaranteed:
                    self.last_candidate_rejections[(channel.channel, channel.region_version)].append({"point": point, "source": source, "reason": "directional_rescue_requires_guaranteed_reception"})
                    continue
                route_reuse = source in {
                    "current", "forward_segment", "next_outer", "future_outer",
                    "appointment_retained", "co_measure_reuse",
                }
                possible_direction = self.grid.distance_to_region(channel.region_mask, point) <= self.config.maximum_receive_radius_m + 1.0e-9
                useful_no_signal = self.grid.no_signal_reduction_count(channel.region_mask, point) > 0
                if source == "side_geometry" and not guaranteed:
                    self.last_candidate_rejections[(channel.channel, channel.region_version)].append({"point": point, "source": source, "reason": "side_geometry_not_guaranteed"})
                    continue
                if not guaranteed and not (route_reuse and (possible_direction or useful_no_signal)):
                    self.last_candidate_rejections[(channel.channel, channel.region_version)].append({"point": point, "source": source, "reason": "neither_guaranteed_nor_route_reuse"})
                    continue
                unique[key] = (point, source)
                admitted += 1
        ordered_candidates = sorted(
            ((point, source, detour(point)) for point, source in unique.values()),
            key=lambda item: (item[1] not in {"current", "forward_segment", "next_outer", "appointment_retained", "co_measure_reuse"}, item[1] != "side_geometry", item[2], math.dist(robot.position, item[0])),
        )
        retained_keys = {
            (round(point[0], 6), round(point[1], 6)) for point, _ in retained_points
        }
        retained = [
            item for item in ordered_candidates
            if (round(item[0][0], 6), round(item[0][1], 6)) in retained_keys
        ]
        return (retained + [item for item in ordered_candidates if item not in retained])[ : max(
            self.config.max_measure_candidates, len(retained)
        )]

    def rank_measure_candidates(
        self,
        channel: ChannelState,
        robot: RobotState,
        remaining_search_points: Sequence[Point],
        next_search_point: Point | None,
        deadline: float | None = None,
        next_channel: int | None = None,
        retained_points: Sequence[tuple[Point, str]] = (),
        *,
        guaranteed_only: bool = False,
        directional_rescue: bool = False,
    ) -> list[MeasureCandidate]:
        if channel.region_mask is None or channel.clear_committed:
            return []
        reference_position = next_search_point if next_search_point is not None else robot.position
        direct_reference = self.fast_clear_time_proxy(channel.region_mask, reference_position)
        candidates = self.generate_measure_candidates(
            channel,
            robot,
            remaining_search_points,
            next_search_point,
            retained_points,
            guaranteed_only=guaranteed_only,
            directional_rescue=directional_rescue,
        )
        def score(point: Point, source: str, detour_distance: float, representative: np.ndarray) -> MeasureCandidate | None:
            switch = self.config.switch_time_s if robot.current_channel != channel.channel else 0.0
            restore_switch = (
                self.config.switch_time_s
                if next_channel is not None and channel.channel != next_channel
                else 0.0
            )
            if next_search_point is None:
                movement_component = math.dist(robot.position, point) / self.config.speed_mps
            else:
                movement_component = detour_distance / self.config.speed_mps
            immediate = movement_component + self.config.measure_time_s + switch + restore_switch
            outcomes: list[tuple[str, float, int]] = []
            guaranteed = self.grid.point_guarantees_reception_with_history(channel, point)
            old_cells = int(np.count_nonzero(channel.region_mask))
            if not guaranteed:
                no_signal_mask = self.grid.exclude_no_signal(channel.region_mask, point)
                # An empty no-signal mask contradicts an already ACTIVE channel:
                # it is not a compatible feedback branch and must not be treated
                # as a zero-cost completion outcome.
                if np.any(no_signal_mask):
                    outcomes.append(("no_signal", self.fast_clear_time_proxy(no_signal_mask, reference_position), int(np.count_nonzero(no_signal_mask))))
            for true_point in representative:
                distance = math.dist(point, true_point)
                if distance <= self.config.near_radius_m + 1.0e-9:
                    outcomes.append(("near", self.config.clear_success_time_s, 0))
                    continue
                if distance > self.config.maximum_receive_radius_m + 1.0e-9:
                    continue
                true_bearing = math.degrees(math.atan2(true_point[1] - point[1], true_point[0] - point[0]))
                for error in self.config.scenario_errors_deg:
                    evidence = DirectionEvidence(point, true_bearing + error)
                    new_mask = self.grid.intersect_direction(channel.region_mask, evidence)
                    if not np.any(new_mask):
                        # A sampled cell center can lie in a retained but not truly feasible cell.
                        new_mask = channel.region_mask
                    outcomes.append(("direction", self.fast_clear_time_proxy(new_mask, reference_position), int(np.count_nonzero(new_mask))))
            if not outcomes:
                return None
            estimate = immediate + max(value for _, value, _ in outcomes)
            best_estimate = immediate + min(value for _, value, _ in outcomes)
            branch_estimates = {kind: max(value for branch, value, _ in outcomes if branch == kind) for kind in {branch for branch, _, _ in outcomes}}
            branch_cells = {kind: max(cells for branch, _, cells in outcomes if branch == kind) for kind in {branch for branch, _, _ in outcomes}}
            useful_branch = any(cells < old_cells for _, _, cells in outcomes)
            if estimate <= direct_reference - self.config.gain_buffer_s + 1.0e-9:
                decision_band = "STRONG_ACCEPT"
            elif best_estimate >= direct_reference + self.config.gain_buffer_s - 1.0e-9:
                decision_band = "STRONG_REJECT"
            else:
                decision_band = "GRAY"
            route_reuse = source in {
                "current", "forward_segment", "next_outer", "future_outer",
                "appointment_retained", "co_measure_reuse",
            }
            allowed = useful_branch and (
                decision_band == "STRONG_ACCEPT"
                or (decision_band == "GRAY" and route_reuse and estimate <= direct_reference + self.config.gain_buffer_s + 1.0e-9)
            )
            return MeasureCandidate(
                point=point,
                immediate_cost_s=immediate,
                estimated_total_s=estimate,
                estimated_best_s=best_estimate,
                direct_clear_baseline_s=direct_reference,
                decision_band=decision_band if allowed else f"REJECTED_{decision_band}",
                detour_distance_m=detour_distance,
                source=source,
                forward_preferred=source in {
                    "current", "forward_segment", "next_outer",
                    "appointment_retained", "co_measure_reuse",
                },
                guaranteed_reception=guaranteed,
                useful_branch_exists=useful_branch,
                worst_branch=max(outcomes, key=lambda item: item[1])[0],
                branch_estimates_s=branch_estimates,
                branch_remaining_cells=branch_cells,
            )

        coarse = [
            score(point, source, detour_distance, self.grid.representative_points(channel.region_mask, min(8, self.config.max_representative_points)))
            for point, source, detour_distance in candidates
        ]
        coarse = [candidate for candidate in coarse if candidate is not None]
        ordered_coarse = sorted(coarse, key=lambda item: (item.estimated_total_s, item.detour_distance_m))
        best_forward_coarse = next((item for item in ordered_coarse if item.forward_preferred), None)
        retained_keys = {
            (round(point[0], 6), round(point[1], 6)) for point, _ in retained_points
        }
        retained_coarse = [
            item for item in ordered_coarse
            if (round(item.point[0], 6), round(item.point[1], 6)) in retained_keys
        ]
        finalists = [] if best_forward_coarse is None else [best_forward_coarse]
        for item in retained_coarse:
            if item not in finalists:
                finalists.append(item)
        finalists.extend(
            item for item in ordered_coarse
            if item not in finalists
        )
        finalists = finalists[: max(self.config.max_scored_measure_candidates_per_decision, len(retained_coarse))]
        representative = self.grid.representative_points(channel.region_mask, self.config.max_representative_points)
        detailed = [score(candidate.point, candidate.source, candidate.detour_distance_m, representative) for candidate in finalists]
        detailed = [candidate for candidate in detailed if candidate is not None]
        best_forward = min(
            (candidate for candidate in detailed if candidate.forward_preferred),
            key=lambda candidate: candidate.estimated_total_s,
            default=None,
        )
        adjusted: list[MeasureCandidate] = []
        for candidate in detailed:
            gap = 0.0 if best_forward is None else candidate.estimated_total_s - best_forward.estimated_total_s
            updated = replace(candidate, forward_gap_s=gap)
            if (
                not updated.forward_preferred
                and best_forward is not None
                and updated.decision_band in {"STRONG_ACCEPT", "GRAY"}
                and updated.estimated_total_s > best_forward.estimated_total_s - self.config.gain_buffer_s + 1.0e-9
            ):
                updated = replace(updated, decision_band="REJECTED_NONFORWARD_NO_6S_ADVANTAGE")
            adjusted.append(updated)
        self.last_candidate_evaluations[(channel.channel, channel.region_version)] = [
            {
                "point": candidate.point, "source": candidate.source, "decision_band": candidate.decision_band,
                "guaranteed_reception": candidate.guaranteed_reception, "useful_branch_exists": candidate.useful_branch_exists,
                "worst_branch": candidate.worst_branch, "branch_estimates_s": candidate.branch_estimates_s,
                "branch_remaining_cells": candidate.branch_remaining_cells, "estimated_total_s": candidate.estimated_total_s,
                "direct_clear_baseline_s": candidate.direct_clear_baseline_s, "detour_distance_m": candidate.detour_distance_m,
                "forward_preferred": candidate.forward_preferred, "forward_gap_s": candidate.forward_gap_s,
            }
            for candidate in adjusted
        ]
        self.last_candidate_evaluations[(channel.channel, channel.region_version)].extend(
            self.last_candidate_rejections.get((channel.channel, channel.region_version), [])
        )
        return sorted([candidate for candidate in adjusted if candidate.decision_band in {"STRONG_ACCEPT", "GRAY"}], key=lambda item: (item.estimated_total_s, item.detour_distance_m))

    def best_clear_insertion_delta(
        self, channel: ChannelState, robot: RobotState, next_search_point: Point
    ) -> tuple[float, Point] | None:
        if channel.region_mask is None:
            return None
        q = self.grid.nearest_region_center_to_segment(
            channel.region_mask, robot.position, next_search_point
        )
        detour = (
            math.dist(robot.position, q)
            + math.dist(q, next_search_point)
            - math.dist(robot.position, next_search_point)
        ) / self.config.speed_mps
        deferred = self.fast_clear_time_proxy(channel.region_mask, next_search_point)
        failed_state = replace(
            channel,
            region_mask=channel.region_mask.copy(),
            anomaly_log=list(channel.anomaly_log),
            direction_observations=list(channel.direction_observations),
            no_signal_points=list(channel.no_signal_points),
            fixed_no_signal_indices=set(channel.fixed_no_signal_indices),
            certified_zone_indices=set(channel.certified_zone_indices),
            zone_certificate_points=dict(channel.zone_certificate_points),
            clear_failure_points=list(channel.clear_failure_points),
            measured_positions=list(channel.measured_positions),
            dedicated_measurement_count=channel.dedicated_measurement_count,
            consecutive_low_gain_count=channel.consecutive_low_gain_count,
            clear_committed=channel.clear_committed,
        )
        update = self.grid.update_clear_failure(failed_state, q)
        if update.accepted:
            after_failure = self.fast_clear_time_proxy(failed_state.region_mask, next_search_point)
            worst_action = max(self.config.clear_success_time_s, self.config.clear_failure_time_s + after_failure)
        else:
            worst_action = self.config.clear_success_time_s
        return detour + worst_action - deferred, q

    def mec_trial_clear_delta(
        self, channel: ChannelState, robot: RobotState, next_search_point: Point
    ) -> tuple[float, Point] | None:
        """Evaluate the MEC-center trial with success/failure branches."""
        if channel.region_mask is None:
            return None
        centers = self.grid.center_points(channel.region_mask)
        circle = minimum_enclosing_circle(centers)
        if circle.radius + self.config.cell_radius_m <= self.config.clear_radius_m + 1.0e-9:
            return None
        q = circle.center
        detour = (
            math.dist(robot.position, q)
            + math.dist(q, next_search_point)
            - math.dist(robot.position, next_search_point)
        ) / self.config.speed_mps
        deferred = self.fast_clear_time_proxy(channel.region_mask, next_search_point)
        failed_state = replace(
            channel,
            region_mask=channel.region_mask.copy(),
            anomaly_log=list(channel.anomaly_log),
            direction_observations=list(channel.direction_observations),
            no_signal_points=list(channel.no_signal_points),
            fixed_no_signal_indices=set(channel.fixed_no_signal_indices),
            certified_zone_indices=set(channel.certified_zone_indices),
            zone_certificate_points=dict(channel.zone_certificate_points),
            clear_failure_points=list(channel.clear_failure_points),
            measured_positions=list(channel.measured_positions),
            region_update_history=list(channel.region_update_history),
            clear_plan_history=list(channel.clear_plan_history),
        )
        update = self.grid.update_clear_failure(failed_state, q)
        failure_tail = (
            self.fast_clear_time_proxy(failed_state.region_mask, next_search_point)
            if update.accepted and failed_state.region_mask is not None
            else 0.0
        )
        worst_action = max(
            self.config.clear_success_time_s,
            self.config.clear_failure_time_s + failure_tail,
        )
        return detour + worst_action - deferred, q

    def estimate_state(
        self,
        robot: RobotState,
        channels: dict[int, ChannelState],
        search_order: Sequence[int],
        fixed_points: Sequence[Point],
        scan_channels: dict[int, Sequence[int]],
    ) -> dict[str, Any]:
        """Build the H-hat and safety ledgers without claiming optimality.

        H-hat uses inexpensive clear proxies.  U_clear uses the certified plans.
        U_future is an intentionally loose full-square raster reserve for sources
        not yet discovered; it is reported as a theoretical reserve only.
        """
        position = robot.position
        channel = robot.current_channel
        survey_s = 0.0
        survey_calls = 0
        for index in search_order:
            point = fixed_points[index]
            survey_s += math.dist(position, point) / self.config.speed_mps
            sequence = ordered_channels(scan_channels.get(index, ()), channel)
            survey_s += len(sequence) * self.config.measure_time_s
            survey_s += sequence_switch_count(channel, sequence) * self.config.switch_time_s
            survey_calls += len(sequence)
            if sequence:
                channel = sequence[-1]
            position = point

        active = [state for state in channels.values() if state.active and state.region_mask is not None]
        active.sort(
            key=lambda state: (
                self.fast_clear_time_proxy(state.region_mask, position),
                state.channel,
            )
        )
        h_clear_s = 0.0
        u_clear_s = 0.0
        clear_calls = 0
        certified_counts: dict[str, int] = {}
        for state in active:
            h_clear_s += self.fast_clear_time_proxy(state.region_mask, position)
            # A clear at every retained cell center fully covers that square because
            # cell_radius < 20 m.  Visiting centers in any order therefore gives a
            # cheap, deliberately loose certified upper bound without running the
            # expensive greedy cover during every planning refresh.
            cell_count = int(np.count_nonzero(state.region_mask))
            initial_bound_m = math.hypot(*position) + self.config.target_radius_m
            inter_cell_bound_m = 2.0 * self.config.target_radius_m
            movement_bound_m = initial_bound_m + max(0, cell_count - 1) * inter_cell_bound_m
            u_clear_s += (
                movement_bound_m / self.config.speed_mps
                + self.config.clear_failure_time_s * max(0, cell_count - 1)
                + self.config.clear_success_time_s
            )
            clear_calls += self.fast_clear_count_proxy(state.region_mask)
            certified_counts[str(state.channel)] = cell_count

        cleared = sum(state.phase == ChannelPhase.CLEARED for state in channels.values())
        possible_future_sources = max(0, self.config.source_count_upper - cleared - len(active))
        spacing = math.sqrt(2.0) * self.config.clear_radius_m
        side = 2.0 * self.config.target_radius_m
        nodes = int(math.ceil(side / spacing)) + 1
        full_cover_count = nodes * nodes
        full_cover_route_m = nodes * side + side
        one_future_s = (
            full_cover_route_m / self.config.speed_mps
            + self.config.clear_failure_time_s * (full_cover_count - 1)
            + self.config.clear_success_time_s
        )
        u_future_s = possible_future_sources * one_future_s
        h_hat = survey_s + h_clear_s
        return {
            "H_hat_s": h_hat,
            "U_safe_s": survey_s + u_clear_s + u_future_s,
            "U_survey_s": survey_s,
            "U_clear_s": u_clear_s,
            "U_future_s": u_future_s,
            "possible_future_sources": possible_future_sources,
            "full_disk_reserve_points_per_source": full_cover_count,
            "safe_cellwise_clear_point_counts": certified_counts,
            "estimated_current_action_calls": survey_calls + clear_calls,
            "evidence_boundary": "H_hat ranks actions; U_future is a loose theoretical reserve",
        }
