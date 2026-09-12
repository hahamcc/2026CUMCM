from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

import q2_model as model


SCRIPT_DIR = Path(__file__).resolve().parent
Point = tuple[float, float]


COLORS = {
    "candidate_fill": "#DCE6EE",
    "candidate_edge": "#355F7D",
    "source_fill": "#E9D8D0",
    "source_edge": "#A85B45",
    "second_line": "#1F708B",
    "movement": "#68737D",
    "target": "#7C8287",
    "text": "#202428",
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 18,
            "font.weight": "bold",
            "axes.labelsize": 23,
            "axes.labelweight": "bold",
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "bf",
        }
    )


def absolute_to_local(config: model.ModelConfig, point: Point) -> Point:
    """把绝对坐标转换为以第一次示向方向为 u 轴的局部坐标。"""

    return model.local_coordinates(config, point)


def local_arrays_to_absolute(
    config: model.ModelConfig, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(config.theta1_deg)
    x = config.s1_x + u * math.cos(angle) - v * math.sin(angle)
    y = config.s1_y + u * math.sin(angle) + v * math.cos(angle)
    return x, y


def display_point(local_point: Point) -> Point:
    """图中横轴画 v、纵轴画 u，使几何图更紧凑且保持等比例。"""

    u, v = local_point
    return v, u


def safe_margin_field(
    config: model.ModelConfig,
    u: np.ndarray,
    v: np.ndarray,
    constraint_sources: list[Point],
    chunk_size: int = 800,
) -> np.ndarray:
    """计算每个第二检测点对全部可能源位置的最小接收余量。"""

    x, y = local_arrays_to_absolute(config, u, v)
    flat_points = np.column_stack((x.ravel(), y.ravel()))
    sources = np.asarray(constraint_sources, dtype=float)
    s1 = np.asarray(config.s1, dtype=float)
    source_radii = np.maximum(
        config.receive_radius_min,
        np.hypot(sources[:, 0] - s1[0], sources[:, 1] - s1[1]),
    )
    margins = np.empty(len(flat_points), dtype=float)
    for start in range(0, len(flat_points), chunk_size):
        stop = min(start + chunk_size, len(flat_points))
        points = flat_points[start:stop]
        distances = np.hypot(
            points[:, None, 0] - sources[None, :, 0],
            points[:, None, 1] - sources[None, :, 1],
        )
        margins[start:stop] = np.min(source_radii[None, :] - distances, axis=1)
    return margins.reshape(u.shape)


def omega_mask(
    config: model.ModelConfig, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """第一次测量后干扰源可能区域 Ω1 的网格掩膜。"""

    x, y = local_arrays_to_absolute(config, u, v)
    distance_to_s1 = np.hypot(x - config.s1_x, y - config.s1_y)
    distance_to_center = np.hypot(
        x - config.target_center_x, y - config.target_center_y
    )
    bearing = np.degrees(np.arctan2(y - config.s1_y, x - config.s1_x))
    angle_difference = (bearing - config.theta1_deg + 180.0) % 360.0 - 180.0
    return (
        (np.abs(angle_difference) <= config.angle_error_deg)
        & (distance_to_s1 > config.bearing_failure_radius)
        & (distance_to_s1 <= config.receive_radius_max)
        & (distance_to_center <= config.target_radius)
    )


def exact_equivalent_points(result_data: dict) -> list[Point]:
    """读取与推荐点最坏直径数值相同的对称等效点。"""

    best = result_data["recommended_point"]
    best_diameter = float(best["worst_diameter"])
    points: list[Point] = []
    for item in result_data.get("near_equivalent_finalists", [best]):
        if math.isclose(
            float(item["worst_diameter"]),
            best_diameter,
            rel_tol=1.0e-10,
            abs_tol=1.0e-6,
        ):
            point = (float(item["x"]), float(item["y"]))
            if point not in points:
                points.append(point)
    return points or [(float(best["x"]), float(best["y"]))]


def draw_crossing_angle(
    ax: plt.Axes,
    station1: Point,
    station2: Point,
    source: Point,
) -> None:
    """在最坏源位置处画出两条观测线的交会角。"""

    first_angle = math.degrees(
        math.atan2(station1[1] - source[1], station1[0] - source[0])
    )
    second_angle = math.degrees(
        math.atan2(station2[1] - source[1], station2[0] - source[0])
    )
    difference = (second_angle - first_angle + 180.0) % 360.0 - 180.0
    if difference >= 0.0:
        start, stop = first_angle, first_angle + difference
    else:
        start, stop = second_angle, second_angle - difference

    radius = 105.0
    ax.add_patch(
        Arc(
            source,
            2.0 * radius,
            2.0 * radius,
            theta1=start,
            theta2=stop,
            color=COLORS["text"],
            linewidth=2.0,
            linestyle=(0, (1, 2)),
            zorder=10,
        )
    )
    middle = math.radians((start + stop) / 2.0)
    arc_point = (
        source[0] + radius * math.cos(middle),
        source[1] + radius * math.sin(middle),
    )
    ax.annotate(
        "交会角",
        xy=arc_point,
        xytext=(arc_point[0] - 180.0, arc_point[1] + 30.0),
        textcoords="data",
        fontsize=19,
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["text"], "lw": 1.8},
        ha="right",
        va="center",
        color=COLORS["text"],
        zorder=11,
    )


def plot_candidate_region(
    config: model.ModelConfig,
    result_data: dict,
    constraint_sources: list[Point],
    output_path: Path,
) -> None:
    source_points = model.build_source_samples(
        config,
        angle_step_deg=0.02,
        radial_step=25.0,
        critical_only=False,
    )
    source_local = np.asarray(
        [absolute_to_local(config, point) for point in source_points]
    )

    bounds = model.candidate_bounding_box(config, constraint_sources)
    candidate_corners = [
        (bounds[0], bounds[2]),
        (bounds[0], bounds[3]),
        (bounds[1], bounds[2]),
        (bounds[1], bounds[3]),
    ]
    candidate_local = np.asarray(
        [absolute_to_local(config, point) for point in candidate_corners]
    )
    all_u = np.concatenate((source_local[:, 0], candidate_local[:, 0]))
    all_v = np.concatenate((source_local[:, 1], candidate_local[:, 1]))
    u_pad = 0.065 * max(1.0, np.ptp(all_u))
    v_pad = 0.075 * max(1.0, np.ptp(all_v))
    u_min = float(np.min(all_u) - u_pad)
    u_max = float(np.max(all_u) + u_pad)
    v_min = float(np.min(all_v) - v_pad)
    v_max = float(np.max(all_v) + v_pad)

    u_values = np.linspace(u_min, u_max, 420)
    v_values = np.linspace(v_min, v_max, 420)
    uu, vv = np.meshgrid(u_values, v_values)
    margins = safe_margin_field(config, uu, vv, constraint_sources)
    source_mask = omega_mask(config, uu, vv)

    # 横轴画 v、纵轴画 u 后，数据纵横比约为 1.3，可用较扁的图幅等比例呈现。
    fig, ax = plt.subplots(figsize=(10.6, 7.8))
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.125, top=0.975)

    ax.contourf(
        vv,
        uu,
        source_mask.astype(float),
        levels=[0.5, 1.5],
        colors=[COLORS["source_fill"]],
        alpha=0.90,
        zorder=2,
    )
    ax.contour(
        vv,
        uu,
        source_mask.astype(float),
        levels=[0.5],
        colors=[COLORS["source_edge"]],
        linewidths=2.2,
        linestyles=[(0, (7, 3))],
        zorder=4,
    )
    ax.contourf(
        vv,
        uu,
        margins,
        levels=[0.0, float(np.nanmax(margins))],
        colors=[COLORS["candidate_fill"]],
        alpha=0.92,
        zorder=1,
    )
    ax.contour(
        vv,
        uu,
        margins,
        levels=[0.0],
        colors=[COLORS["candidate_edge"]],
        linewidths=3.0,
        linestyles=["solid"],
        zorder=5,
    )

    center_u, center_v = absolute_to_local(config, config.target_center)
    ax.add_patch(
        Circle(
            (center_v, center_u),
            config.target_radius,
            fill=False,
            edgecolor=COLORS["target"],
            linewidth=1.9,
            linestyle=(0, (8, 3, 1, 3)),
            zorder=0,
        )
    )
    ax.plot(
        [0.0, 0.0],
        [0.0, u_max],
        color="#33383C",
        linewidth=1.8,
        linestyle=(0, (1, 3)),
        zorder=6,
    )

    recommended = result_data["recommended_point"]
    primary_local = absolute_to_local(
        config, (float(recommended["x"]), float(recommended["y"]))
    )
    primary = display_point(primary_local)
    worst_source_local = absolute_to_local(
        config,
        (
            float(recommended["worst_source_x"]),
            float(recommended["worst_source_y"]),
        ),
    )
    worst_source = display_point(worst_source_local)

    # 三条具有不同物理意义的线采用不同线型，灰度打印时仍可区分。
    ax.plot(
        [0.0, worst_source[0]],
        [0.0, worst_source[1]],
        color=COLORS["source_edge"],
        linewidth=2.2,
        linestyle=(0, (8, 3)),
        zorder=7,
    )
    ax.plot(
        [primary[0], worst_source[0]],
        [primary[1], worst_source[1]],
        color=COLORS["second_line"],
        linewidth=2.4,
        linestyle=(0, (6, 2, 1, 2)),
        zorder=7,
    )
    ax.plot(
        [0.0, primary[0]],
        [0.0, primary[1]],
        color=COLORS["movement"],
        linewidth=2.0,
        linestyle=(0, (2, 3)),
        zorder=7,
    )

    # 区域和边界采用图内直接标注，不另设图例。
    ax.text(
        465.0,
        440.0,
        "第二检测点候选区域" + "\n" + r"$C_{\mathrm{safe}}$",
        fontsize=19,
        weight="bold",
        ha="center",
        va="center",
        color=COLORS["candidate_edge"],
        zorder=8,
    )
    ax.annotate(
        r"干扰源可能区域 $\Omega_1$",
        xy=(12.0, 1140.0),
        xytext=(235.0, 1175.0),
        fontsize=20,
        color=COLORS["source_edge"],
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["source_edge"], "lw": 1.7},
        ha="left",
        va="center",
        zorder=9,
    )
    ax.annotate(
        "目标圆边界",
        xy=(-1065.0, 1450.0),
        xytext=(-900.0, 1400.0),
        textcoords="data",
        fontsize=19,
        weight="bold",
        color="#5E6469",
        arrowprops={"arrowstyle": "->", "color": COLORS["target"], "lw": 1.5},
        ha="left",
        zorder=9,
    )

    # 点标记加大，并在点旁直接写出物理含义和关键数值。
    ax.scatter(
        [0.0],
        [0.0],
        marker="s",
        s=180,
        color=COLORS["text"],
        edgecolor="white",
        linewidth=1.0,
        zorder=12,
    )
    ax.annotate(
        r"第一次检测点 $S_1$",
        (0.0, 0.0),
        xytext=(15, -7),
        textcoords="offset points",
        fontsize=19,
        weight="bold",
        ha="left",
        va="top",
        zorder=13,
    )

    equivalent_points = exact_equivalent_points(result_data)
    for index, absolute_point in enumerate(equivalent_points):
        local_point = absolute_to_local(config, absolute_point)
        plotted_point = display_point(local_point)
        ax.scatter(
            [plotted_point[0]],
            [plotted_point[1]],
            marker="*",
            s=480,
            color=COLORS["candidate_edge"],
            edgecolor="white",
            linewidth=1.3,
            zorder=13,
        )
        if index == 0:
            text = r"推荐点 $S_2^*$"
            text_position = (-14, 14)
            horizontal = "right"
            vertical = "bottom"
        else:
            text = r"对称等效点 $S_{2,\mathrm{eq}}^*$"
            text_position = (-30, 14)
            horizontal = "left"
            vertical = "bottom"
        ax.annotate(
            text,
            plotted_point,
            xytext=text_position,
            textcoords="offset points",
            fontsize=21,
            weight="bold",
            ha=horizontal,
            va=vertical,
            zorder=14,
        )

    ax.scatter(
        [worst_source[0]],
        [worst_source[1]],
        marker="X",
        s=270,
        color=COLORS["source_edge"],
        edgecolor="white",
        linewidth=1.1,
        zorder=13,
    )
    ax.annotate(
        r"最坏情景源位置 $G^*$",
        worst_source,
        xytext=(18, 0),
        textcoords="offset points",
        fontsize=21,
        weight="bold",
        ha="left",
        va="center",
        color=COLORS["source_edge"],
        zorder=14,
    )

    draw_crossing_angle(
        ax,
        (0.0, 0.0),
        primary,
        worst_source,
    )

    # 直接标注三条线，避免设置占空间的图例。
    ax.text(
        -250.0,
        700.0,
        "第一次观测线",
        fontsize=19,
        weight="bold",
        rotation=0,
        color=COLORS["source_edge"],
        ha="center",
        va="center",
        zorder=9,
    )
    ax.text(
        -450.0,
        1160.0,
        "第二次观测线",
        fontsize=19,
        weight="bold",
        rotation=50,
        color=COLORS["second_line"],
        ha="center",
        va="center",
        zorder=9,
    )
    ax.text(
        -435.0,
        355.0,
        "机器狗移动路径",
        fontsize=19,
        weight="bold",
        rotation=-53,
        color=COLORS["movement"],
        ha="center",
        va="center",
        zorder=9,
    )

    ax.set_xlabel("垂直第一次示向方向的横向偏移 $v$ / m")
    ax.set_ylabel("沿第一次示向方向的前进距离 $u$ / m")
    ax.set_xlim(v_min, v_max)
    ax.set_ylim(u_min, u_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(
        True,
        color="#C9CED2",
        linewidth=0.75,
        linestyle=(0, (2, 4)),
        alpha=0.65,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#363B3F")
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")

    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="问题二候选区域与推荐点可视化")
    parser.add_argument(
        "--result-json",
        type=Path,
        default=SCRIPT_DIR / "q2_result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "q2_candidate_region.png",
    )
    args = parser.parse_args()

    result_data = json.loads(args.result_json.read_text(encoding="utf-8"))
    config = model.ModelConfig(**result_data["config"])
    constraint_sources = model.build_source_samples(
        config,
        config.verification_constraint_angle_step_deg,
        radial_step=None,
        critical_only=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_candidate_region(config, result_data, constraint_sources, args.output)
    print(f"已生成：{args.output}")


if __name__ == "__main__":
    set_plot_style()
    main()
