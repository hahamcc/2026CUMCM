from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Q3Config
from .models import Action, ActionKind


@dataclass(frozen=True)
class ClientResponse:
    http_status: int | None
    body: dict[str, Any] | None
    network_error: str | None = None

    @property
    def accepted(self) -> bool:
        return bool(self.http_status == 200 and self.body and self.body.get("accepted") is True)


class SimulatorClient:
    """Strict serial HTTP client; uncertain state-changing calls are not retried."""

    def __init__(self, base_url: str, robot_id: str, config: Q3Config):
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.config = config

    def payload_for(self, action: Action) -> tuple[str, dict[str, Any]]:
        path = {
            ActionKind.ENTER: "/enter",
            ActionKind.MEASURE: "/measure",
            ActionKind.CLEAR: "/clear",
            ActionKind.EXIT: "/exit",
        }[action.kind]
        payload: dict[str, Any] = {
            "arena_id": "default",
            "robot_id": self.robot_id,
            "request_id": action.request_id,
        }
        if action.kind in {ActionKind.MEASURE, ActionKind.CLEAR}:
            if action.position is None or action.channel is None:
                raise ValueError(f"{action.kind.value} requires position and channel")
            x, y = action.position
            if not all(isinstance(value, (int, float)) and math_is_finite(value) and abs(value) <= self.config.coordinate_limit_m for value in (x, y)):
                raise ValueError("coordinates must be finite and within the interface range")
            if action.channel not in self.config.channels:
                raise ValueError("channel must be in 1..20")
            payload["position"] = {"x": float(x), "y": float(y)}
            payload["channel"] = int(action.channel)
        return path, payload

    def execute(self, action: Action) -> ClientResponse:
        path, payload = self.payload_for(action)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.request_timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
                return ClientResponse(int(response.status), body)
        except HTTPError as error:
            raw = error.read()
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = None
            return ClientResponse(int(error.code), body, str(error))
        except (URLError, TimeoutError, socket.timeout, ConnectionError) as error:
            # The server may already have executed this request.  Without a
            # read-only status endpoint, retrying would guess the remote state.
            return ClientResponse(None, None, f"uncertain remote state: {error}")


def math_is_finite(value: float) -> bool:
    import math

    return math.isfinite(float(value))
