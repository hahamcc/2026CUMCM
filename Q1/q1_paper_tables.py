"""生成问题一正文验证表和问题一至问题二衔接表。

本脚本只组织算例、调用 q1_algorithm.py 并导出数据，不重复实现几何算法。
默认在当前 Q1 文件夹内生成两个带 UTF-8 BOM 的 CSV 文件，可直接用 Excel
打开后复制到论文表格中。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from q1_algorithm import (
    COVERED,
    DEGENERATE,
    EMPTY,
    NOT_COVERED,
    UNBOUNDED,
    VALID_POLYGON,
    NumericTolerance,
    Observation,
    build_halfplanes,
    diameter_and_coverage,
    direction,
    equilateral_example,
    line_intersection,
    relative_area_measure,
    satisfies_all,
    solve_localization,
)


VALIDATION_FILENAME = "表1_第一问算法验证.csv"
TRANSITION_FILENAME = "表2_监测点增减与直径变化.csv"


def _yes_no(status: str | None) -> str:
    if status == COVERED:
        return "能覆盖"
    if status == NOT_COVERED:
        return "不能覆盖"
    return "未判定"


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig 使 Excel 在 Windows 下直接识别中文，不产生乱码。
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def _boundary_checks() -> dict[str, bool]:
    """逐项执行不适合在正文中展开的边界与异常测试。"""

    checks: dict[str, bool] = {}

    # 方位角跨越 0°：359.5° 和 0° 应在角域内，反方向 180° 应在角域外。
    _, east_wedge = build_halfplanes([Observation(0.0, 0.0, 0.0)])
    checks["跨零度"] = (
        satisfies_all(direction(359.5), east_wedge, 1.0e-12)
        and satisfies_all(direction(0.0), east_wedge, 1.0e-12)
        and not satisfies_all(direction(180.0), east_wedge, 1.0e-12)
    )

    # 两边界夹角约 10^-12 度但并不平行，应触发高精度并保留有限交点。
    _, near_parallel = build_halfplanes(
        [(0.0, 0.0, 1.0), (0.0, 1.0, 1.0 + 1.0e-12)]
    )
    near_point, used_high_precision = line_intersection(
        near_parallel[0], near_parallel[2], NumericTolerance()
    )
    checks["近平行"] = (
        used_high_precision
        and near_point is not None
        and abs(near_point[0]) > 1.0e10
    )

    # 圆周内外 10^-9 的扰动必须得到相反判断。
    inside = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0 - 1.0e-9)]
    outside = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0 + 1.0e-9)]
    checks["圆周微扰"] = (
        diameter_and_coverage(inside)[7] == COVERED
        and diameter_and_coverage(outside)[7] == NOT_COVERED
    )

    checks["空集"] = (
        solve_localization([(0.0, 0.0, 0.0), (-10.0, 0.0, 180.0)]).status
        == EMPTY
    )
    checks["无界"] = (
        solve_localization([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]).status
        == UNBOUNDED
    )
    checks["退化"] = (
        solve_localization([(0.0, 0.0, 0.0), (0.0, 0.0, 180.0)]).status
        == DEGENERATE
    )

    thin = [(0.0, 0.0), (10.0, 0.0), (10.0, 1.0e-5), (0.0, 1.0e-5)]
    thin_scaled = [(1.0e6 * x, 1.0e6 * y) for x, y in thin]
    checks["尺度变换"] = math.isclose(
        relative_area_measure(thin),
        relative_area_measure(thin_scaled),
        rel_tol=1.0e-12,
        abs_tol=0.0,
    )

    original = solve_localization(equilateral_example())
    reordered = solve_localization(list(reversed(equilateral_example())))
    checks["输入重排"] = (
        original.status == reordered.status == VALID_POLYGON
        and math.isclose(
            original.diameter or 0.0,
            reordered.diameter or 0.0,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        )
        and original.coverage_status == reordered.coverage_status
    )
    return checks


def build_validation_rows() -> list[list[str]]:
    """形成正文建议采用的三层验证表。"""

    triangle = solve_localization(equilateral_example())
    if triangle.status != VALID_POLYGON or triangle.diameter is None:
        raise RuntimeError(f"等边三角形完整流程失败：{triangle.status}")
    triangle_error = abs(triangle.diameter - 20.0)

    square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    square_result = diameter_and_coverage(square)
    square_diameter = square_result[0]
    square_status = square_result[7]
    square_theory = 2.0 * math.sqrt(2.0)
    square_error = abs(square_diameter - square_theory)

    checks = _boundary_checks()
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("边界测试失败：" + "、".join(failed))
    check_names = "、".join(checks)

    return [
        [
            "完整流程",
            "三组测向角域交成等边三角形",
            "D=20 m，不能覆盖",
            f"D={triangle.diameter:.12f} m，{_yes_no(triangle.coverage_status)}",
            f"直径绝对误差 {triangle_error:.3e} m",
            "验证角域构造、顶点筛选、直径与覆盖判定的完整流程",
        ],
        [
            "几何模块",
            "边长为2的正方形",
            "D=2√2 m，能覆盖",
            f"D={square_diameter:.12f} m，{_yes_no(square_status)}",
            f"直径绝对误差 {square_error:.3e} m",
            "验证最远顶点对、候选圆心及圆周临界等号",
        ],
        [
            "边界情况",
            check_names,
            f"{len(checks)}项结果均与解析预期一致",
            f"{sum(checks.values())}/{len(checks)}项通过",
            "无失败项",
            "验证数值稳定性、输入顺序不变性及异常状态识别",
        ],
    ]


def _bearing_to_target(
    station: tuple[float, float], target: tuple[float, float] = (0.0, 0.0)
) -> float:
    return math.degrees(
        math.atan2(target[1] - station[1], target[0] - station[0])
    ) % 360.0


def _observation_at(
    station: tuple[float, float], target: tuple[float, float] = (0.0, 0.0)
) -> Observation:
    return Observation(station[0], station[1], _bearing_to_target(station, target))


def build_transition_rows() -> list[list[str]]:
    """构造监测点增减实验，为问题二的第二检测点选择提供动机。"""

    # 所有点距 G=(0,0) 不超过 1000 m，保证处于题设最小有效接收半径内。
    stations = {
        "S1": (-650.0, -180.0),
        "S2": (-420.0, 610.0),
        "S3": (180.0, 720.0),
        "S4": (690.0, 260.0),
        "S5": (400.0, -70.0),   # 对直径有明显改善的新增点
        "S6": (900.0, 0.0),     # 对当前定位区域完全冗余的新增点
    }
    observations = {name: _observation_at(point) for name, point in stations.items()}
    if any(math.hypot(x, y) > 1000.0 for x, y in stations.values()):
        raise AssertionError("衔接算例的监测点必须位于最小有效接收半径1000 m内。")
    baseline_names = ["S1", "S2", "S3", "S4"]

    cases: list[tuple[str, str, list[str], str]] = [
        (
            "A（基准）",
            "S1(-650,-180)，S2(-420,610)，S3(180,720)，S4(690,260)",
            baseline_names,
            "G=(0,0)，示向度取真实方位（零误差），由4点自然形成不规则六边形",
        ),
        (
            "B（增加有效点）",
            "A+S5(400,-70)",
            baseline_names + ["S5"],
            "新增约束切除原最远点附近区域，直径减小",
        ),
        (
            "C（增加冗余点）",
            "A+S6(900,0)",
            baseline_names + ["S6"],
            "新增角域包含原定位区域，直径不变",
        ),
        (
            "D（删除冗余点）",
            "A-S4(690,260)",
            ["S1", "S2", "S3"],
            "删除后可行域和直径均不变",
        ),
        (
            "E（删除关键点）",
            "A-S2(-420,610)",
            ["S1", "S3", "S4"],
            "关键约束缺失使定位区域及直径明显扩大",
        ),
    ]

    results = []
    for case_name, adjustment, names, explanation in cases:
        result = solve_localization([observations[name] for name in names])
        if result.status != VALID_POLYGON or result.diameter is None:
            raise RuntimeError(f"衔接算例 {case_name} 失败：{result.status}")
        results.append((case_name, adjustment, names, explanation, result))

    baseline_diameter = results[0][4].diameter
    assert baseline_diameter is not None

    rows: list[list[str]] = []
    for case_name, adjustment, names, explanation, result in results:
        assert result.diameter is not None
        relative_change = 100.0 * (result.diameter - baseline_diameter) / baseline_diameter
        rows.append(
            [
                case_name,
                adjustment,
                str(len(names)),
                "正常有界多边形",
                str(len(result.vertices)),
                f"{result.diameter:.6f}",
                f"{relative_change:+.2f}%",
                explanation,
            ]
        )
    return rows


def generate_tables(output_dir: Path) -> tuple[Path, Path]:
    validation_path = output_dir / VALIDATION_FILENAME
    transition_path = output_dir / TRANSITION_FILENAME

    _write_csv(
        validation_path,
        [
            "验证层次",
            "测试内容",
            "理论或预期结果",
            "程序结果",
            "误差或通过数",
            "验证目的",
        ],
        build_validation_rows(),
    )
    _write_csv(
        transition_path,
        [
            "方案",
            "监测点调整（坐标单位：m）",
            "监测点数n",
            "区域状态",
            "顶点数",
            "定位区域直径D/m",
            "相对基准变化率",
            "现象解释",
        ],
        build_transition_rows(),
    )
    return validation_path, transition_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成问题一论文正文所需的两个数据表")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="CSV 输出目录；默认是脚本所在的 Q1 文件夹",
    )
    args = parser.parse_args()

    validation_path, transition_path = generate_tables(args.output_dir)
    print(f"Generated: {validation_path.resolve()}")
    print(f"Generated: {transition_path.resolve()}")


if __name__ == "__main__":
    main()
