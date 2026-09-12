from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import tempfile
from pathlib import Path

from .config import Q3Config
from .local_simulator import LocalSimulatorClient, random_case
from .models import StrategyMode
from .run import run_strategy
from .strategy import Q3Strategy


def _run_case(arguments: tuple[int, str, str, dict[str, object]]) -> dict[str, object]:
    seed, mode_text, root_text, config_overrides = arguments
    config = Q3Config(**config_overrides)
    sources = random_case(seed, config)
    client = LocalSimulatorClient(sources, config)
    strategy = Q3Strategy(config, mode=StrategyMode(mode_text))
    summary = run_strategy(client, strategy, Path(root_text) / str(seed))
    return {
        "seed": seed,
        "mode": mode_text,
        "valid": bool(summary["completion_certificate"]["valid"]),
        "cleared_count": int(summary["cleared_count"]),
        "expected_count": len(sources),
        "virtual_time_s": float(summary["virtual_total_time_s"]),
        "real_compute_time_s": float(summary["real_program_time_s"]),
        "terminal_error": summary["terminal_error"],
        "anomaly_count": len(summary["anomalies"]),
    }


def run_batch(
    seed_start: int,
    count: int,
    mode: StrategyMode | str,
    workers: int = 1,
    config_overrides: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="q3-eval-") as temp:
        arguments = [
            (seed_start + offset, StrategyMode(mode).value, temp, config_overrides or {})
            for offset in range(count)
        ]
        if workers <= 1:
            return [_run_case(item) for item in arguments]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_run_case, arguments))


def percentile90(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = 0.9 * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def compare(seed_start: int, count: int, workers: int = 1) -> dict[str, object]:
    baseline = run_batch(seed_start, count, StrategyMode.B, workers)
    main = run_batch(seed_start, count, StrategyMode.B_PLUS, workers)
    baseline_times = [float(item["virtual_time_s"]) for item in baseline]
    main_times = [float(item["virtual_time_s"]) for item in main]
    baseline_valid = all(item["valid"] and item["cleared_count"] == item["expected_count"] and item["terminal_error"] is None for item in baseline)
    main_valid = all(item["valid"] and item["cleared_count"] == item["expected_count"] and item["terminal_error"] is None and item["anomaly_count"] == 0 for item in main)
    baseline_mean = statistics.fmean(baseline_times)
    main_mean = statistics.fmean(main_times)
    baseline_p90 = percentile90(baseline_times)
    main_p90 = percentile90(main_times)
    split_consistent: bool | None = None
    split_differences: list[float] = []
    if count >= 2:
        split = count // 2
        ranges = (range(0, split), range(split, count))
        for indices in ranges:
            b_mean = statistics.fmean(baseline_times[index] for index in indices)
            p_mean = statistics.fmean(main_times[index] for index in indices)
            split_differences.append(p_mean - b_mean)
        split_consistent = all(value < 0.0 for value in split_differences)
    return {
        "seed_start": seed_start,
        "case_count": count,
        "B": {"valid": baseline_valid, "mean_virtual_time_s": baseline_mean, "p90_virtual_time_s": baseline_p90},
        "B+": {"valid": main_valid, "mean_virtual_time_s": main_mean, "p90_virtual_time_s": main_p90},
        "admission": {
            "all_complete": main_valid,
            "mean_better": main_mean < baseline_mean,
            "two_modes_complete": baseline_valid and main_valid,
            "independent_split_mean_differences_s": split_differences,
            "independent_splits_both_improve": split_consistent,
            "passed": baseline_valid and main_valid and main_mean < baseline_mean and split_consistent is not False,
        },
        "cases": {"B": baseline, "B+": main},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Q3 baseline and rolling strategy on deterministic local cases.")
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--mode", choices=["both", "B", "B+"], default="both")
    parser.add_argument("--output", type=Path, default=Path("Q3/runs/evaluation.json"))
    args = parser.parse_args()
    if args.mode == "both":
        result = compare(args.seed_start, args.cases, args.workers)
        passed = bool(result["admission"]["passed"])
        compact = {key: result[key] for key in ("seed_start", "case_count", "B", "B+", "admission")}
    else:
        cases = run_batch(args.seed_start, args.cases, StrategyMode(args.mode), args.workers)
        passed = all(
            item["valid"]
            and item["cleared_count"] == item["expected_count"]
            and item["terminal_error"] is None
            and item["anomaly_count"] == 0
            for item in cases
        )
        result = {
            "seed_start": args.seed_start,
            "case_count": args.cases,
            "mode": args.mode,
            "valid": passed,
            "mean_virtual_time_s": statistics.fmean(float(item["virtual_time_s"]) for item in cases),
            "p90_virtual_time_s": percentile90([float(item["virtual_time_s"]) for item in cases]),
            "cases": cases,
        }
        compact = {key: value for key, value in result.items() if key != "cases"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
