"""绘制问题一等边三角形验证图。

本文件只负责可视化，全部几何计算调用 q1_algorithm.py。图中不设置标题，
使用较大的坐标轴、图例和标注字体。默认输出 q1_result.png。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from q1_algorithm import VALID_POLYGON, equilateral_example, solve_localization


def configure_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    selected = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "font.size": 17,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "axes.unicode_minus": False,
        }
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    axis.tick_params(width=1.4, length=6)
    for spine in axis.spines.values():
        spine.set_linewidth(1.3)


def create_figure(output_path: Path, show: bool = False) -> None:
    observations = equilateral_example()
    result = solve_localization(observations)
    if result.status != VALID_POLYGON:
        raise RuntimeError(f"示例未形成正常定位多边形：{result.status}")

    assert result.farthest_pair is not None
    assert result.circle_center is not None
    assert result.circle_radius is not None

    configure_fonts()
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    global_axis, detail_axis = axes

    colors = ["#0072B2", "#D55E00", "#009E73"]
    ray_length = 760.0
    for index, observation in enumerate(observations):
        color = colors[index]
        station = (observation.x, observation.y)

        for offset, linestyle, label in [
            (-1.0, "-", "±1°边界" if index == 0 else None),
            (1.0, "-", None),
        ]:
            angle = math.radians(observation.bearing_deg + offset)
            end = (
                station[0] + ray_length * math.cos(angle),
                station[1] + ray_length * math.sin(angle),
            )
            global_axis.plot(
                [station[0], end[0]],
                [station[1], end[1]],
                color=color,
                linewidth=2.2,
                linestyle=linestyle,
                alpha=0.88,
                label=label,
            )

        center_angle = math.radians(observation.bearing_deg)
        center_end = (
            station[0] + ray_length * math.cos(center_angle),
            station[1] + ray_length * math.sin(center_angle),
        )
        global_axis.plot(
            [station[0], center_end[0]],
            [station[1], center_end[1]],
            color=color,
            linewidth=1.6,
            linestyle="--",
            alpha=0.75,
            label="示向度中心线" if index == 0 else None,
        )
        global_axis.scatter(
            station[0],
            station[1],
            s=115,
            color=color,
            marker="s",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            label="检测点" if index == 0 else None,
        )
        global_axis.annotate(
            f"$S_{index + 1}$",
            station,
            xytext=(10, 9),
            textcoords="offset points",
            fontsize=18,
            weight="bold",
        )

    polygon_global = Polygon(
        result.vertices,
        closed=True,
        facecolor="#F0E442",
        edgecolor="black",
        linewidth=2.4,
        alpha=0.9,
        label="定位多边形",
        zorder=4,
    )
    global_axis.add_patch(polygon_global)
    global_axis.set_xlim(-700.0, 430.0)
    global_axis.set_ylim(-650.0, 650.0)
    global_axis.text(
        0.025,
        0.96,
        "(a)",
        transform=global_axis.transAxes,
        fontsize=21,
        weight="bold",
        va="top",
    )
    _style_axis(global_axis)
    global_axis.legend(loc="lower left", framealpha=0.94)

    polygon_detail = Polygon(
        result.vertices,
        closed=True,
        facecolor="#F0E442",
        edgecolor="black",
        linewidth=2.8,
        alpha=0.72,
        label="定位多边形",
        zorder=3,
    )
    detail_axis.add_patch(polygon_detail)

    candidate_circle = Circle(
        result.circle_center,
        result.circle_radius,
        facecolor="#56B4E9",
        edgecolor="#0072B2",
        linewidth=2.8,
        alpha=0.25,
        label="同直径候选圆盘",
        zorder=2,
    )
    detail_axis.add_patch(candidate_circle)

    first, second = result.farthest_pair
    detail_axis.plot(
        [first[0], second[0]],
        [first[1], second[1]],
        color="#D55E00",
        linewidth=3.3,
        label="最远顶点对",
        zorder=5,
    )
    detail_axis.scatter(
        [point[0] for point in result.vertices],
        [point[1] for point in result.vertices],
        s=120,
        color="black",
        zorder=6,
        label="多边形顶点",
    )
    detail_axis.scatter(
        result.circle_center[0],
        result.circle_center[1],
        s=155,
        marker="x",
        linewidth=3.0,
        color="#CC0000",
        zorder=7,
        label="候选圆心",
    )

    labels = ["A", "B", "C"]
    sorted_for_labels = sorted(result.vertices, key=lambda point: (point[1], point[0]))
    for label, point in zip(labels, sorted_for_labels):
        detail_axis.annotate(
            label,
            point,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=18,
            weight="bold",
        )

    detail_axis.text(
        0.025,
        0.96,
        "(b)",
        transform=detail_axis.transAxes,
        fontsize=21,
        weight="bold",
        va="top",
    )
    detail_axis.set_xlim(-22.0, 22.0)
    detail_axis.set_ylim(-16.0, 18.0)
    _style_axis(detail_axis)
    detail_axis.legend(loc="upper right", framealpha=0.94)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制问题一定位与覆盖判定图")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("q1_result.png"),
        help="输出图片路径",
    )
    parser.add_argument("--show", action="store_true", help="保存后显示图片")
    args = parser.parse_args()
    create_figure(args.output, args.show)
    print(f"Figure saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
