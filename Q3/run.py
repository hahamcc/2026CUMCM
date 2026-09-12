from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .client import ClientResponse, SimulatorClient
from .config import Q3Config
from .models import Action, RobotState, StrategyMode
from .strategy import Q3Strategy, new_channel_states


SPEC_VERSION = "q3-b-bplus-v7-directional-rescue"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "true_count": int(np.count_nonzero(value))}
    if hasattr(value, "value"):
        return value.value
    return value


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(_jsonable(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def run_strategy(
    client: Any,
    strategy: Q3Strategy,
    log_dir: Path,
    max_actions: int = 100_000,
) -> dict[str, Any]:
    # A live user may deliberately keep a common parent directory for drills.
    # Never append a second run into the prior run's audit trail.
    active_log_dir = log_dir
    if any((log_dir / name).exists() for name in ("actions.jsonl", "checkpoint.json", "summary.json")):
        active_log_dir = log_dir / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    robot = RobotState()
    channels = new_channel_states(strategy.config)
    logger = JsonlLogger(active_log_dir / "actions.jsonl")
    terminal_error: str | None = None

    for _ in range(max_actions):
        action = strategy.next_action(robot, channels)
        if action is None:
            break
        logger.write(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "stage": "intent",
                "action": action,
            }
        )
        request_started = time.monotonic()
        response: ClientResponse = client.execute(action)
        request_elapsed = time.monotonic() - request_started
        logger.write(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "stage": "response",
                "action": action,
                "http_status": response.http_status,
                "response": response.body,
                "network_error": response.network_error,
            }
        )
        if response.network_error is not None and response.body is None:
            terminal_error = response.network_error
            break
        if response.http_status != 200 or response.body is None or response.body.get("accepted") is not True:
            terminal_error = f"request not executed: HTTP {response.http_status}, body={response.body}"
            break
        robot.real_action_time_s += request_elapsed
        robot.real_action_count += 1
        strategy.apply_response(action, response.body, robot, channels)
        checkpoint = {
            "spec_version": SPEC_VERSION,
            "mode": strategy.mode.value,
            "robot": robot,
            "channels": channels,
            "last_request_id": action.request_id,
        }
        (active_log_dir / "checkpoint.json").write_text(
            json.dumps(_jsonable(checkpoint), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if robot.exited:
            break
    else:
        terminal_error = f"action limit {max_actions} reached"

    summary = _jsonable(strategy.make_summary(robot, channels))
    summary["spec_version"] = SPEC_VERSION
    summary["run_id"] = active_log_dir.name
    summary["config"] = _jsonable(strategy.config)
    summary["terminal_error"] = terminal_error
    summary_path = active_log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Problem 3 robot-dog strategy.")
    parser.add_argument("--robot-id", required=True, help="Current simulator login team id; never stored in source.")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--log-dir", type=Path, default=Path("Q3/runs/live"))
    parser.add_argument("--mode", choices=[item.value for item in StrategyMode], default="B+")
    parser.add_argument("--config", type=Path, help="Optional JSON object overriding team settings.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.config is not None:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--config must contain one JSON object")
        overrides = loaded
    config = Q3Config(**overrides)
    strategy = Q3Strategy(config, mode=StrategyMode(args.mode))
    client = SimulatorClient(args.base_url, args.robot_id, config)
    summary = run_strategy(client, strategy, args.log_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["completion_certificate"]["valid"] and summary["terminal_error"] is None else 2


if __name__ == "__main__":
    sys.exit(main())
