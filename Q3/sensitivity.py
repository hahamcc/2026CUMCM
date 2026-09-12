from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .evaluate import percentile90, run_batch
from .models import StrategyMode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Small fixed-seed sensitivity check for Q3 team parameters."
    )
    parser.add_argument("--seed-start", type=int, default=30_000)
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--output", type=Path, default=Path("Q3/runs/latest_sensitivity.json")
    )
    args = parser.parse_args()

    variants: list[tuple[str, dict[str, object]]] = []
    for buffer_s in (0.0, 6.0, 12.0):
        variants.append((f"buffer_{int(buffer_s)}", {"gain_buffer_s": buffer_s}))
    for limit in (1, 2, 3):
        variants.append(
            (
                f"limits_{limit}",
                {
                    "max_dedicated_measurements_per_channel": limit,
                    "max_consecutive_low_gain_measurements": limit,
                },
            )
        )
    for grid_step in (2.5, 5.0, 10.0):
        variants.append((f"grid_{grid_step:g}", {"grid_step_m": grid_step}))
    for high, forced in ((1, 2), (2, 3), (3, 4)):
        variants.append(
            (
                f"starvation_{high}_{forced}",
                {
                    "high_priority_wait_segments": high,
                    "forced_wait_segments": forced,
                },
            )
        )

    results: dict[str, object] = {
        "seed_start": args.seed_start,
        "case_count": args.cases,
        "note": "Local deterministic simulator only; parameters are team simplifications.",
        "variants": {},
    }
    for name, overrides in variants:
        cases = run_batch(
            args.seed_start,
            args.cases,
            StrategyMode.B_PLUS,
            args.workers,
            overrides,
        )
        times = [float(item["virtual_time_s"]) for item in cases]
        valid = all(
            item["valid"]
            and item["cleared_count"] == item["expected_count"]
            and item["terminal_error"] is None
            and item["anomaly_count"] == 0
            for item in cases
        )
        results["variants"][name] = {
            "overrides": overrides,
            "valid": valid,
            "mean_virtual_time_s": statistics.fmean(times),
            "p90_virtual_time_s": percentile90(times),
            "max_real_compute_time_s": max(
                float(item["real_compute_time_s"]) for item in cases
            ),
            "cases": cases,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = {
        name: {
            key: value
            for key, value in record.items()
            if key not in {"cases", "overrides"}
        }
        for name, record in results["variants"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if all(record["valid"] for record in results["variants"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
