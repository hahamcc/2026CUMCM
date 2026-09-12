from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from .client import ClientResponse
from .config import Q3Config
from .models import Action, ActionKind, Point


@dataclass
class Source:
    channel: int
    position: Point
    receive_radius_m: float
    cleared: bool = False


class LocalSimulatorClient:
    """Deterministic in-process simulator for Problem 3 all-directional sources."""

    def __init__(
        self,
        sources: list[Source],
        config: Q3Config | None = None,
        remaining_real_duration_s: int = 1200,
    ):
        self.config = config or Q3Config()
        self.sources = {source.channel: source for source in sources}
        self.remaining_real_duration_s = remaining_real_duration_s
        self.position: Point = (0.0, 0.0)
        self.current_channel = 1
        self.virtual_time_s = 0.0
        self.entered = False
        self.exited = False
        self._responses: dict[str, tuple[Action, ClientResponse]] = {}

    def execute(self, action: Action) -> ClientResponse:
        previous = self._responses.get(action.request_id)
        if previous is not None:
            if previous[0] != action:
                return ClientResponse(409, {"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0})
            return previous[1]
        response = self._execute_new(action)
        if response.accepted:
            self._responses[action.request_id] = (action, response)
        return response

    def _execute_new(self, action: Action) -> ClientResponse:
        common = {"accepted": True, "real_timestamp_ms": 0}
        if action.kind == ActionKind.ENTER:
            if self.entered or self.exited:
                return self._rejected()
            self.entered = True
            body = {
                **common,
                "virtual_time_s": self.virtual_time_s,
                "max_virtual_duration_s": 360000,
                "max_real_duration_s": 1200,
                "remaining_real_duration_s": self.remaining_real_duration_s,
            }
            return ClientResponse(200, body)
        if not self.entered or self.exited:
            return self._rejected()
        if action.kind == ActionKind.EXIT:
            self.exited = True
            return ClientResponse(200, {**common, "virtual_time_s": self.virtual_time_s, "exit_reason": "user_exit"})
        if action.position is None or action.channel is None:
            return ClientResponse(400, self._error_body())

        movement = math.dist(self.position, action.position) / self.config.speed_mps
        self.position = action.position
        source = self.sources.get(action.channel)
        if action.kind == ActionKind.MEASURE:
            switch = self.config.switch_time_s if action.channel != self.current_channel else 0.0
            self.current_channel = action.channel
            self.virtual_time_s += movement + switch + self.config.measure_time_s
            body = {**common, "virtual_time_s": self.virtual_time_s}
            if source is None or source.cleared or math.dist(action.position, source.position) > source.receive_radius_m + 1.0e-9:
                body["measure_result"] = "no_signal"
            elif math.dist(action.position, source.position) <= self.config.near_radius_m + 1.0e-9:
                body["measure_result"] = "near"
            else:
                true_bearing = math.degrees(math.atan2(source.position[1] - action.position[1], source.position[0] - action.position[0]))
                body["measure_result"] = "direction"
                body["svd_deg"] = round(normalize_angle(true_bearing + self._fixed_error(action.position, action.channel)), 2)
            return ClientResponse(200, body)

        success = source is not None and not source.cleared and math.dist(action.position, source.position) <= self.config.clear_radius_m + 1.0e-9
        if success:
            source.cleared = True
            self.virtual_time_s += movement + self.config.clear_success_time_s
            result = "success"
        else:
            self.virtual_time_s += movement + self.config.clear_failure_time_s
            result = "no_target_in_range"
        return ClientResponse(200, {**common, "virtual_time_s": self.virtual_time_s, "clear_result": result})

    def _fixed_error(self, point: Point, channel: int) -> float:
        key = f"{point[0]:.6f},{point[1]:.6f},{channel}".encode("ascii")
        integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        # Reserve 0.01 degree for the interface's two-decimal quantization so the
        # displayed bearing remains inside the stated +/-1 degree error bound.
        return -0.99 + 1.98 * integer / (2**64 - 1)

    @staticmethod
    def _error_body() -> dict[str, object]:
        return {"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0}

    def _rejected(self) -> ClientResponse:
        return ClientResponse(200, self._error_body())


def normalize_angle(value: float) -> float:
    normalized = (value + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized


def random_case(seed: int, config: Q3Config | None = None) -> list[Source]:
    config = config or Q3Config()
    rng = random.Random(seed)
    count = rng.randint(10, config.source_count_upper)
    channels = rng.sample(list(config.channels), count)
    result: list[Source] = []
    for channel in channels:
        radius = config.target_radius_m * math.sqrt(rng.random())
        angle = rng.uniform(-math.pi, math.pi)
        result.append(
            Source(
                channel=channel,
                position=(radius * math.cos(angle), radius * math.sin(angle)),
                receive_radius_m=rng.uniform(
                    config.guaranteed_receive_radius_m,
                    config.maximum_receive_radius_m,
                ),
            )
        )
    return result
