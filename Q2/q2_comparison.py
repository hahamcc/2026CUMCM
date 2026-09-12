from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import q2_model as model


SCRIPT_DIR = Path(__file__).resolve().parent
Point = tuple[float, float]


@dataclass(frozen=True)
class SimpleCandidate:
    point: Point
    score: float


def distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def distance_priority_score(candidate: Point, sources: list[Point]) -> float:
    """距离优先：到最远可能源的距离越小越好。"""

    return max(distance(candidate, source) for source in sources)


def effective_crossing_angle(
    first_station: Point, second_station: Point, source: Point
) -> float:
    """把0度和180度都视为共线，返回0至90度的有效交会角。"""

    angle = model.crossing_angle(first_station, second_station, source)
    return min(angle, 180.0 - angle)


def angle_priority_score(
    config: model.ModelConfig, candidate: Point, sources: list[Point]
) -> float:
    """交会角优先：最不利有效交会角越大越好。"""

    angles = [
        effective_crossing_angle(config.s1, candidate, source)
        for source in sources
        if distance(candidate, source) > config.bearing_failure_radius
    ]
    return min(angles) if angles else 90.0


def local_candidates(
    config: model.ModelConfig,
    seeds: list[SimpleCandidate],
    constraint_sources: list[Point],
    radius: float,
    step: float,
    top_k: int,
    smaller_is_better: bool,
) -> list[Point]:
    ranked = sorted(seeds, key=lambda item: item.score, reverse=not smaller_is_better)
    offsets = model._frange(-radius, radius, step)
    points: dict[tuple[int, int], Point] = {}

    for seed in ranked[:top_k]:
        for dx in offsets:
            for dy in offsets:
                candidate = (seed.point[0] + dx, seed.point[1] + dy)
                if distance(candidate, config.s1) <= config.feasibility_tolerance:
                    continue
                margin = model.safe_margin(config, candidate, constraint_sources)
                if margin >= -config.feasibility_tolerance:
                    key = (
                        round(candidate[0] * 1.0e6),
                        round(candidate[1] * 1.0e6),
                    )
                    points[key] = candidate
    return list(points.values())


def search_simple_strategy(
    config: model.ModelConfig,
    initial_points: list[Point],
    constraint_sources: list[Point],
    verification_constraints: list[Point],
    metric_sources: list[Point],
    score_function: Callable[[Point, list[Point]], float],
    smaller_is_better: bool,
) -> Point:
    """沿用主程序的空间尺度逐级细化，但只计算一个简单选点指标。"""

    current = [
        SimpleCandidate(point, score_function(point, metric_sources))
        for point in initial_points
    ]
    for step, radius in zip(
        config.refinement_steps, config.refinement_radii, strict=True
    ):
        points = local_candidates(
            config,
            current,
            constraint_sources,
            radius=radius,
            step=step,
            top_k=6,
            smaller_is_better=smaller_is_better,
        )
        current = [
            SimpleCandidate(point, score_function(point, metric_sources))
            for point in points
        ]

    ranked = sorted(
        current, key=lambda item: item.score, reverse=not smaller_is_better
    )
    for item in ranked:
        if (
            model.safe_margin(config, item.point, verification_constraints)
            >= -config.feasibility_tolerance
        ):
            return item.point
    raise RuntimeError("简单策略未找到能够通过加密候选域复核的第二检测点。")


def load_main_result(path: Path) -> tuple[model.ModelConfig, Point]:
    if not path.exists():
        raise FileNotFoundError("请先运行q2_model.py生成q2_result.json。")
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {item.name for item in fields(model.ModelConfig)}
    values = {key: value for key, value in data["config"].items() if key in names}
    for name in ("refinement_steps", "refinement_radii", "refinement_top_k"):
        if name in values:
            values[name] = tuple(values[name])
    config = model.ModelConfig(**values)
    recommended = data["recommended_point"]
    return config, (float(recommended["x"]), float(recommended["y"]))


def evaluate_worst_diameter(
    config: model.ModelConfig, candidate: Point, sources: list[Point]
) -> float:
    result = model.evaluate_candidate(
        config,
        candidate,
        sources,
        config.verification_epsilon_step_deg,
    )
    return result.worst_diameter


def format_number(value: float) -> str:
    return "无界" if math.isinf(value) else f"{value:.3f}"


def run_comparison(result_json: Path, output_csv: Path) -> list[dict[str, str]]:
    config, proposed_point = load_main_result(result_json)

    # 简单策略搜索使用加密候选域；最终三个点再用最密候选域共同复核。
    search_constraints = model.build_source_samples(
        config,
        config.refined_constraint_angle_step_deg,
        radial_step=None,
        critical_only=True,
    )
    verification_constraints = model.build_source_samples(
        config,
        config.verification_constraint_angle_step_deg,
        radial_step=None,
        critical_only=True,
    )
    metric_sources = model.build_source_samples(
        config,
        config.verification_source_angle_step_deg,
        config.verification_source_radial_step,
        critical_only=False,
    )
    initial = model.generate_candidate_grid(
        config, search_constraints, config.candidate_step
    )
    initial_points = [point for point, _ in initial]

    distance_point = search_simple_strategy(
        config,
        initial_points,
        search_constraints,
        verification_constraints,
        metric_sources,
        distance_priority_score,
        smaller_is_better=True,
    )
    angle_point = search_simple_strategy(
        config,
        initial_points,
        search_constraints,
        verification_constraints,
        metric_sources,
        lambda point, sources: angle_priority_score(config, point, sources),
        smaller_is_better=False,
    )

    strategies = [
        ("距离优先", "尽量靠近全部可能源", distance_point),
        ("交会角优先", "尽量避免两次方向共线", angle_point),
        ("本文方案", "最坏定位直径最小", proposed_point),
    ]

    checked: list[tuple[str, str, Point, float]] = []
    for name, idea, point in strategies:
        margin = model.safe_margin(config, point, verification_constraints)
        if margin < -config.feasibility_tolerance:
            raise RuntimeError(f"{name}没有通过统一的候选域复核。")
        worst_diameter = evaluate_worst_diameter(config, point, metric_sources)
        checked.append((name, idea, point, worst_diameter))

    proposed_diameter = checked[-1][3]
    rows: list[dict[str, str]] = []
    for name, idea, point, worst_diameter in checked:
        u, v = model.local_coordinates(config, point)
        if name == "本文方案":
            comparison = "基准"
        elif math.isinf(worst_diameter):
            comparison = "本文方案可形成有限定位区域"
        elif math.isinf(proposed_diameter):
            comparison = "不适用"
        else:
            saved = worst_diameter - proposed_diameter
            reduction = 100.0 * saved / worst_diameter if worst_diameter > 0.0 else 0.0
            comparison = (
                f"本文方案缩小{max(0.0, saved):.3f} m"
                f"（降低{max(0.0, reduction):.2f}%）"
            )
        rows.append(
            {
                "策略": name,
                "直观做法": idea,
                "第二点相对坐标_米": f"({u:.1f}, {v:.1f})",
                "最坏定位直径_米": (
                    "无法限定（无界）"
                    if math.isinf(worst_diameter)
                    else format_number(worst_diameter)
                ),
                "与本文方案相比": comparison,
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("三个策略均已通过同一保证接收候选域检查。")
    print("策略        最坏定位直径/m    与本文方案相比")
    for row in rows:
        print(
            f"{row['策略']:<10}"
            f"{row['最坏定位直径_米']:>14}"
            f"    {row['与本文方案相比']}"
        )
    print(f"结果已写入：{output_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="问题二三个直观选点策略的简洁对照")
    parser.add_argument(
        "--result-json",
        type=Path,
        default=SCRIPT_DIR / "q2_result.json",
        help="q2_model.py生成的主结果",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SCRIPT_DIR / "q2_comparison.csv",
        help="对照表输出位置",
    )
    args = parser.parse_args()
    run_comparison(args.result_json, args.output_csv)


if __name__ == "__main__":
    main()
