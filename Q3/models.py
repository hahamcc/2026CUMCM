from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Point = tuple[float, float]


class StrategyMode(str, enum.Enum):
    B = "B"
    B_PLUS = "B+"


class SafetyMode(str, enum.Enum):
    NORMAL = "NORMAL"
    SAFE = "SAFE"
    EMERGENCY = "EMERGENCY"


class ChannelPhase(str, enum.Enum):
    UNKNOWN = "UNKNOWN"
    ACTIVE = "ACTIVE"
    CLEAR_COMMITTED = "CLEAR_COMMITTED"
    CLEARED = "CLEARED"
    EXCLUDED = "EXCLUDED"
    INCONSISTENT = "INCONSISTENT"


class ActionKind(str, enum.Enum):
    ENTER = "ENTER"
    MEASURE = "MEASURE"
    CLEAR = "CLEAR"
    EXIT = "EXIT"


@dataclass(frozen=True)
class DirectionEvidence:
    position: Point
    bearing_deg: float


@dataclass
class MeasurementAppointment:
    """A deferred, forward-only opportunity to measure one active channel."""

    appointment_id: str
    point: Point
    source: str
    anchor_segment: tuple[int, int]
    anchor_index: int
    region_version: int
    guaranteed_reception: bool
    estimated_saving_s: float
    estimated_clear_count_reduction: int
    classification: str
    status: str = "PENDING"
    status_reason: str = ""


@dataclass
class ChannelState:
    channel: int
    phase: ChannelPhase = ChannelPhase.UNKNOWN
    direction_observations: list[DirectionEvidence] = field(default_factory=list)
    no_signal_points: list[Point] = field(default_factory=list)
    fixed_no_signal_indices: set[int] = field(default_factory=set)
    certified_zone_indices: set[int] = field(default_factory=set)
    zone_certificate_points: dict[int, Point] = field(default_factory=dict)
    clear_failure_points: list[Point] = field(default_factory=list)
    measured_positions: list[Point] = field(default_factory=list)
    active_measurement_count: int = 0
    region_mask: np.ndarray | None = field(default=None, repr=False)
    region_version: int = 0
    anomaly_log: list[str] = field(default_factory=list)
    dedicated_measurement_count: int = 0
    rescue_measurement_count: int = 0
    consecutive_low_gain_count: int = 0
    clear_committed: bool = False
    near_pending: bool = False
    waiting_outer_segments: int = 0
    committed_order: int | None = None
    last_progress_outer_segment: int = 0
    region_update_history: list[dict[str, Any]] = field(default_factory=list)
    clear_plan_history: list[dict[str, Any]] = field(default_factory=list)
    skipped_clear_decisions: int = 0
    appointments: list[MeasurementAppointment] = field(default_factory=list)
    appointment_history: list[dict[str, Any]] = field(default_factory=list)
    directional_rescue_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def direction_count(self) -> int:
        return len(self.direction_observations)

    @property
    def complete(self) -> bool:
        return self.phase in {ChannelPhase.CLEARED, ChannelPhase.EXCLUDED}

    def has_measured_at(self, point: Point, tolerance: float = 1.0e-6) -> bool:
        return any(math.dist(point, old) <= tolerance for old in self.measured_positions)

    @property
    def active(self) -> bool:
        return self.phase in {ChannelPhase.ACTIVE, ChannelPhase.CLEAR_COMMITTED}

    def commit_clear(self, order: int) -> None:
        self.clear_committed = True
        self.phase = ChannelPhase.CLEAR_COMMITTED
        if self.committed_order is None:
            self.committed_order = order


@dataclass
class RobotState:
    position: Point = (0.0, 0.0)
    current_channel: int = 1
    virtual_time_s: float = 0.0
    entered_real_monotonic: float | None = None
    remaining_real_duration_s: float | None = None
    visited_search_points: set[int] = field(default_factory=set)
    local_insertions_since_search: int = 0
    appointment_measurements_since_search: int = 0
    directional_rescues_since_search: int = 0
    co_measurements_at_stop: int = 0
    co_measurement_position: Point | None = None
    co_measurement_reconnect_point: Point | None = None
    optional_measurements_at_stop: int = 0
    optional_measurement_position: Point | None = None
    last_outer_search_index: int | None = None
    ring_direction: str | None = None
    ring_order: list[int] = field(default_factory=list)
    current_outer_segment: tuple[int, int] | None = None
    completed_outer_segments: int = 0
    safety_mode: SafetyMode = SafetyMode.NORMAL
    time_breakdown: dict[str, float] = field(
        default_factory=lambda: {
            "movement_s": 0.0,
            "switch_s": 0.0,
            "measure_s": 0.0,
            "clear_success_s": 0.0,
            "clear_failure_s": 0.0,
        }
    )
    real_action_time_s: float = 0.0
    real_action_count: int = 0
    entered: bool = False
    exited: bool = False

    @property
    def real_elapsed_s(self) -> float:
        if self.entered_real_monotonic is None:
            return 0.0
        return time.monotonic() - self.entered_real_monotonic

    @property
    def real_remaining_s(self) -> float | None:
        if self.remaining_real_duration_s is None:
            return None
        return max(0.0, self.remaining_real_duration_s - self.real_elapsed_s)


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    request_id: str
    position: Point | None = None
    channel: int | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class RegionUpdate:
    accepted: bool
    before_cells: int
    after_cells: int
    reason: str


@dataclass
class ClearPlan:
    points: list[Point]
    route_length_m: float
    upper_time_s: float
    certificate: dict[str, Any]


@dataclass
class ClearRouteSession:
    """One locally continuous certified-clear route."""

    channel: int
    remaining_points: list[Point]
    reconnect_point: Point | None
    region_version: int


@dataclass(frozen=True)
class MeasureCandidate:
    point: Point
    immediate_cost_s: float
    estimated_total_s: float
    estimated_best_s: float
    direct_clear_baseline_s: float
    decision_band: str
    detour_distance_m: float
    source: str
    forward_preferred: bool = False
    forward_gap_s: float = 0.0
    guaranteed_reception: bool = False
    useful_branch_exists: bool = False
    worst_branch: str = ""
    branch_estimates_s: dict[str, float] = field(default_factory=dict)
    branch_remaining_cells: dict[str, int] = field(default_factory=dict)


@dataclass
class RunSummary:
    mode: str
    cleared_count: int
    channel_states: dict[int, str]
    completion_certificate: dict[str, Any]
    virtual_total_time_s: float
    average_clear_time_s: float | None
    real_program_time_s: float
    anomalies: list[str]
    action_count: int
    time_breakdown: dict[str, float]
    safety_mode: str
    planning_snapshot: dict[str, Any]
    decision_trace: list[dict[str, Any]]
