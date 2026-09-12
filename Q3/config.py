from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q3Config:
    """Centralized problem facts and team-selected strategy parameters."""

    target_radius_m: float = 1800.0
    guaranteed_receive_radius_m: float = 1000.0
    maximum_receive_radius_m: float = 1500.0
    near_radius_m: float = 5.0
    clear_radius_m: float = 20.0
    bearing_error_deg: float = 1.0
    speed_mps: float = 5.0
    measure_time_s: float = 5.0
    switch_time_s: float = 1.0
    clear_failure_time_s: float = 3.0
    clear_success_time_s: float = 5.0
    channel_min: int = 1
    channel_max: int = 20
    source_count_upper: int = 16
    coordinate_limit_m: float = 2_000_000.0

    ring_radius_m: float = 1150.0
    ranking_grid_step_m: float = 15.0
    grid_step_m: float = 5.0
    sensitivity_grid_step_m: float = 2.5
    supplement_grid_step_m: float = 100.0
    route_sample_step_m: float = 100.0
    max_measure_candidates: int = 10
    # Candidate generation is deliberately diverse: route reuse alone can be
    # geometrically uninformative, especially after a single bearing.
    max_scored_measure_candidates_per_decision: int = 4
    max_scored_channels_per_insertion_decision: int = 4
    max_representative_points: int = 16
    scenario_errors_deg: tuple[float, ...] = (-1.0, 0.0, 1.0)
    # Team strategy setting: the first two outer stations are broad discovery
    # stations.  This turns early detections into second bearings before the
    # ring ends, rather than leaving one-bearing channels for tail clearing.
    early_full_outer_scan_count: int = 2
    max_optional_measurements_per_stop: int = 2
    # A dedicated measurement stop may reuse the same movement for its primary
    # channel plus at most this many separately worthwhile ACTIVE channels.
    max_co_measured_active_channels_per_stop: int = 2
    max_measurement_appointments_per_channel: int = 2
    # Kept for audit only.  B+ allocation is constrained by effective bearing
    # count below, because no_signal is not a new triangulation constraint.
    max_active_measurements_per_channel: int = 5
    max_direction_observations_per_channel: int = 5
    max_dedicated_measurements_per_channel: int = 2
    max_rescue_measurements_per_channel: int = 1
    rescue_clear_time_s: float = 300.0
    appointment_large_clear_time_s: float = 300.0
    appointment_large_clear_count: int = 8
    max_consecutive_low_gain_measurements: int = 2
    max_macro_insertions_per_segment: int = 1
    max_directional_rescues_per_segment: int = 1
    gain_buffer_s: float = 6.0
    high_priority_wait_segments: int = 2
    forced_wait_segments: int = 3
    safe_elapsed_s: float = 15.0 * 60.0
    safe_projection_factor: float = 1.5
    safe_projection_deadline_s: float = 18.0 * 60.0
    emergency_elapsed_s: float = 18.0 * 60.0
    decision_budget_s: float = 0.5
    deep_scoring_stop_s: float = 90.0
    real_time_guard_s: float = 30.0
    request_timeout_s: float = 5.0

    @property
    def cell_radius_m(self) -> float:
        return (2.0**0.5) * self.grid_step_m / 2.0

    @property
    def channels(self) -> range:
        return range(self.channel_min, self.channel_max + 1)
