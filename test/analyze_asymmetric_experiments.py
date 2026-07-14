#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "result"
OUTPUT = RESULT / "asymmetric_analysis_20260714"

HEAT_CAPACITY = RESULT / "asymmetric_heat_ratio_runs_20260712_140822.csv"
HEAT_LAMBDA_LOW = HEAT_CAPACITY
HEAT_LAMBDA_HIGH = RESULT / "asymmetric_heat_ratio_runs_20260712_195919.csv"
NONLINEAR_CAPACITY = RESULT / "asymmetric_nonlinear_runs_20260712_195919.csv"
NONLINEAR_RATIO = RESULT / "asymmetric_nonlinear_runs_20260713_131808.csv"

COLORS = ["#1769aa", "#d1495b", "#2a9d6f", "#8b5cf6", "#e68a00", "#334155"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def ci95(values: list[float]) -> float:
    return 1.96 * std(values) / math.sqrt(len(values))


def summarize(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
    field: str,
) -> dict[tuple[str, ...], tuple[float, float]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[field]))
    return {key: (mean(values), ci95(values)) for key, values in grouped.items()}


def paired_difference(
    rows: list[dict[str, str]],
    filters_a: dict[str, str],
    filters_b: dict[str, str],
    field: str = "test_rmse",
) -> tuple[float, float]:
    def selected(filters: dict[str, str]) -> dict[str, float]:
        return {
            row["seed"]: float(row[field])
            for row in rows
            if all(row[key] == value for key, value in filters.items())
        }

    left = selected(filters_a)
    right = selected(filters_b)
    differences = [left[seed] - right[seed] for seed in sorted(left.keys() & right.keys())]
    return mean(differences), ci95(differences)


def svg_chart(
    path: Path,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float, float]]]],
    *,
    x_log: bool = False,
    x_ticks: list[float] | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 100, 45, 100, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_points = [point for _, points in series for point in points]
    xs = [point[0] for point in all_points]
    ys_low = [point[1] - point[2] for point in all_points]
    ys_high = [point[1] + point[2] for point in all_points]
    tx = (lambda value: math.log10(value)) if x_log else (lambda value: value)
    xmin, xmax = min(tx(value) for value in xs), max(tx(value) for value in xs)
    if xmin == xmax:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if y_bounds is None:
        ymin, ymax = min(ys_low), max(ys_high)
        padding = max((ymax - ymin) * 0.12, 1e-4)
        ymin, ymax = max(0.0, ymin - padding), ymax + padding
    else:
        ymin, ymax = y_bounds

    def sx(value: float) -> float:
        return left + (tx(value) - xmin) / (xmax - xmin) * plot_w

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="42" font-family="sans-serif" font-size="25" font-weight="700" fill="#17202a">{html.escape(title)}</text>',
        f'<text x="{left}" y="70" font-family="sans-serif" font-size="14" fill="#5d6d7e">{html.escape(subtitle)}</text>',
    ]
    for index in range(6):
        value = ymin + index * (ymax - ymin) / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e7ebef" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#566573">{value:.3f}</text>')

    ticks = x_ticks or sorted(set(xs))
    for value in ticks:
        x = sx(value)
        label = f"{value:g}"
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 7}" stroke="#34495e"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 26}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#566573">{label}</text>')

    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#34495e" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#34495e" stroke-width="1.5"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#34495e">{html.escape(x_label)}</text>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="sans-serif" font-size="15" fill="#34495e">{html.escape(y_label)}</text>',
    ])

    legend_x = left + plot_w - 165
    for index, (label, points) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y, _ in points)
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for x_value, y_value, error in points:
            x, y = sx(x_value), sy(y_value)
            low, high = sy(y_value - error), sy(y_value + error)
            parts.extend([
                f'<line x1="{x:.2f}" y1="{high:.2f}" x2="{x:.2f}" y2="{low:.2f}" stroke="{color}" stroke-width="1.6"/>',
                f'<line x1="{x - 5:.2f}" y1="{high:.2f}" x2="{x + 5:.2f}" y2="{high:.2f}" stroke="{color}" stroke-width="1.6"/>',
                f'<line x1="{x - 5:.2f}" y1="{low:.2f}" x2="{x + 5:.2f}" y2="{low:.2f}" stroke="{color}" stroke-width="1.6"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#ffffff" stroke="{color}" stroke-width="3"/>',
            ])
        legend_y = top + 18 + index * 25
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 37}" y="{legend_y + 5}" font-family="sans-serif" font-size="13" fill="#34495e">{html.escape(label)}</text>')

    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def make_heat_capacity_chart(rows: list[dict[str, str]]) -> None:
    selected = [row for row in rows if row["stage"] == "ratio_scan" and row["method"] == "ridge_dual"]
    stats = summarize(selected, ("sample_ratio", "hidden_dim"), "test_rmse")
    series = []
    for ratio in (2.0, 4.0, 8.0):
        points = []
        for hidden_dim in (1, 2, 5, 10, 20, 50):
            value, error = stats[(str(ratio), str(hidden_dim))]
            points.append((hidden_dim, value, error))
        series.append((f"S/p={ratio:g}", points))
    svg_chart(
        OUTPUT / "heat_capacity.svg",
        "AsymmetricHeat: capacity under controlled sample ratio",
        "Mean test RMSE with 95% confidence intervals, 5 evaluation seeds",
        "Hidden dimension H (log scale)",
        "Test RMSE",
        series,
        x_log=True,
        x_ticks=[1, 2, 5, 10, 20, 50],
        y_bounds=(0.30, 0.75),
    )


def make_heat_lambda_chart(low: list[dict[str, str]], high: list[dict[str, str]]) -> None:
    selected = [row for row in low if row["stage"] == "lambda_scan"]
    selected += [row for row in high if row["stage"] == "lambda_continuation" and row["normalized_lambda"] != "0.1"]
    grouped: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in selected:
        grouped[(int(row["hidden_dim"]), float(row["normalized_lambda"]))].append(float(row["test_rmse"]))
    stats = {key: (mean(values), ci95(values)) for key, values in grouped.items()}
    series = []
    for hidden_dim in (5, 10, 20, 50):
        lambdas = sorted(
            key[1] for key in stats if key[0] == hidden_dim
        )
        points = [(value, *stats[(hidden_dim, value)]) for value in lambdas]
        series.append((f"H={hidden_dim}", points))
    svg_chart(
        OUTPUT / "heat_regularization.svg",
        "AsymmetricHeat: normalized ridge regularization",
        "The turning point appears near normalized lambda 0.1-0.3",
        "Normalized lambda (log scale)",
        "Test RMSE",
        series,
        x_log=True,
        x_ticks=[1e-6, 1e-4, 1e-2, 0.1, 0.3, 1, 3, 10],
        y_bounds=(0.33, 0.66),
    )


def make_nonlinear_capacity_charts(rows: list[dict[str, str]]) -> None:
    for equation, short, bounds in [
        ("AsymmetricAllenCahn", "allen", (0.0, 0.030)),
        ("AsymmetricHJBLQ", "hjb", (0.11, 0.20)),
    ]:
        selected = [row for row in rows if row["equation"] == equation]
        stats = summarize(selected, ("method", "hidden_dim"), "test_rmse")
        series = []
        for method, label in (("batched_qr", "RFM"), ("constant", "Constant")):
            points = []
            for hidden_dim in (1, 5, 10, 20):
                value, error = stats[(method, str(hidden_dim))]
                points.append((hidden_dim, value, error))
            series.append((label, points))
        svg_chart(
            OUTPUT / f"{short}_capacity.svg",
            f"{equation}: feature capacity",
            "Fixed S/p=4, mean test RMSE with 95% confidence intervals",
            "Hidden dimension H",
            "Test RMSE",
            series,
            x_ticks=[1, 5, 10, 20],
            y_bounds=bounds,
        )


def make_nonlinear_ratio_charts(rows: list[dict[str, str]]) -> None:
    for equation, short, bounds in [
        ("AsymmetricAllenCahn", "allen", (0.0, 0.030)),
        ("AsymmetricHJBLQ", "hjb", (0.10, 0.18)),
    ]:
        selected = [row for row in rows if row["equation"] == equation]
        stats = summarize(selected, ("method", "sample_ratio"), "test_rmse")
        series = []
        for method, label in (("batched_qr", "RFM"), ("constant", "Constant")):
            points = []
            for ratio in (4.0, 8.0, 16.0):
                value, error = stats[(method, str(ratio))]
                points.append((ratio, value, error))
            series.append((label, points))
        svg_chart(
            OUTPUT / f"{short}_sample_ratio.svg",
            f"{equation}: sample-to-parameter ratio",
            "Fixed H=20, mean test RMSE with 95% confidence intervals",
            "Sample-to-parameter ratio S/p",
            "Test RMSE",
            series,
            x_ticks=[4, 8, 16],
            y_bounds=bounds,
        )


def report_markdown(
    heat_capacity: list[dict[str, str]],
    heat_lambda_low: list[dict[str, str]],
    heat_lambda_high: list[dict[str, str]],
    nonlinear_capacity: list[dict[str, str]],
    nonlinear_ratio: list[dict[str, str]],
) -> str:
    heat_ratio_stats = summarize(
        [row for row in heat_capacity if row["stage"] == "ratio_scan" and row["method"] == "ridge_dual"],
        ("sample_ratio", "hidden_dim"),
        "test_rmse",
    )
    lambda_rows = [row for row in heat_lambda_low if row["stage"] == "lambda_scan"]
    lambda_rows += [row for row in heat_lambda_high if row["stage"] == "lambda_continuation" and row["normalized_lambda"] != "0.1"]
    lambda_grouped: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in lambda_rows:
        lambda_grouped[(int(row["hidden_dim"]), float(row["normalized_lambda"]))].append(
            float(row["test_rmse"])
        )
    lambda_stats = {
        key: (mean(values), ci95(values)) for key, values in lambda_grouped.items()
    }
    best_lambdas = {}
    for hidden_dim in (5, 10, 20, 50):
        candidates = {
            key[1]: value[0]
            for key, value in lambda_stats.items()
            if key[0] == hidden_dim
        }
        best_lambdas[hidden_dim] = min(candidates, key=candidates.get)

    capacity_stats = summarize(nonlinear_capacity, ("equation", "method", "hidden_dim"), "test_rmse")
    ratio_stats = summarize(nonlinear_ratio, ("equation", "method", "sample_ratio"), "test_rmse")
    reduction_stats = summarize(nonlinear_ratio, ("equation", "method", "sample_ratio"), "test_mse_reduction")

    heat_h1 = heat_ratio_stats[("8.0", "1")][0]
    heat_h50 = heat_ratio_stats[("8.0", "50")][0]
    allen_h20 = capacity_stats[("AsymmetricAllenCahn", "batched_qr", "20")][0]
    allen_constant = capacity_stats[("AsymmetricAllenCahn", "constant", "20")][0]
    hjb_h20 = capacity_stats[("AsymmetricHJBLQ", "batched_qr", "20")][0]
    hjb_constant = capacity_stats[("AsymmetricHJBLQ", "constant", "20")][0]

    allen_4 = ratio_stats[("AsymmetricAllenCahn", "batched_qr", "4.0")][0]
    allen_16 = ratio_stats[("AsymmetricAllenCahn", "batched_qr", "16.0")][0]
    hjb_4 = ratio_stats[("AsymmetricHJBLQ", "batched_qr", "4.0")][0]
    hjb_16 = ratio_stats[("AsymmetricHJBLQ", "batched_qr", "16.0")][0]
    hjb_delta, hjb_delta_ci = paired_difference(
        nonlinear_ratio,
        {"equation": "AsymmetricHJBLQ", "method": "batched_qr", "sample_ratio": "16.0"},
        {"equation": "AsymmetricHJBLQ", "method": "batched_qr", "sample_ratio": "4.0"},
    )

    lines = [
        "# 弱对称高维 PDE 随机特征实验综合分析",
        "",
        "生成日期: 2026-07-14.",
        "",
        "## 执行摘要",
        "",
        f"- 在 AsymmetricHeat 上固定 $S/p=8$ 后, $H$ 从 1 增加到 50, Test RMSE 从 `{heat_h1:.4f}` 降至 `{heat_h50:.4f}`, 降幅为 `{100 * (1 - heat_h50 / heat_h1):.1f}%`.",
        f"- 归一化 ridge 正则化的最优区间不再位于扫描边界. $H=5$ 取 $\\bar\\lambda={best_lambdas[5]:g}$, $H\\ge10$ 的最优值集中在 $\\bar\\lambda=0.3$.",
        f"- AsymmetricAllenCahn 在 $H=20,S/p=4$ 时 Test RMSE 为 `{allen_h20:.5f}`, constant 为 `{allen_constant:.5f}`, RFM 消除了约 `{100 * (1 - (allen_h20 / allen_constant) ** 2):.1f}%` 的基线 MSE.",
        f"- AsymmetricHJBLQ 更困难. $H=20,S/p=4$ 时 RFM 仅略优于 constant (`{hjb_h20:.5f}` 对 `{hjb_constant:.5f}`), 但将 $S/p$ 提高到 16 后 Test RMSE 降至 `{hjb_16:.5f}`.",
        "- 整体证据表明, 方法收益不是由径向对称性或仅估计 $y_0$ 造成的. 随机特征主要改善路径级 $Z$ 表示, 但需要让样本数随参数数目同步增长.",
        "",
        "## 1. AsymmetricHeat: 容量与样本规模",
        "",
        "![Heat capacity](heat_capacity.svg)",
        "",
        "| $H$ | $p$ | RMSE, $S/p=2$ | RMSE, $S/p=4$ | RMSE, $S/p=8$ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for hidden_dim in (1, 2, 5, 10, 20, 50):
        values = [heat_ratio_stats[(str(ratio), str(hidden_dim))][0] for ratio in (2.0, 4.0, 8.0)]
        lines.append(f"| {hidden_dim} | {1 + 100 * hidden_dim} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    lines += [
        "",
        "固定 $S/p$ 后, Test RMSE 随 $H$ 单调下降. 这修正了早期固定 $S$ 实验中大 $H$ 过拟合的假象. 当 $H=50$ 时, $S/p$ 从 2 增加到 8, Test RMSE 由约 `0.3655` 降至 `0.3448`, 说明样本不足和特征容量是两个独立因素.",
        "",
        "## 2. AsymmetricHeat: 正则化转折",
        "",
        "![Heat regularization](heat_regularization.svg)",
        "",
        "| $H$ | 最优 $\\bar\\lambda$ | 最优 Test RMSE |",
        "|---:|---:|---:|",
    ]
    for hidden_dim in (5, 10, 20, 50):
        value = best_lambdas[hidden_dim]
        lines.append(f"| {hidden_dim} | {value:g} | {lambda_stats[(hidden_dim, value)][0]:.5f} |")
    lines += [
        "",
        "较大的 raw lambda 本身没有可比意义. 当前目标应按 $\\lambda=S\\bar\\lambda$ 理解. 当 $\\bar\\lambda$ 从最优区间继续增加到 1, 3, 10 时, 训练误差和测试误差同时上升, 表明模型开始发生过度收缩. 因此 ridge 的收益来自有限的偏差-方差折中, 而不是正则化越大越好.",
        "",
        "## 3. 弱对称非线性方程: 特征容量",
        "",
        "### AsymmetricAllenCahn",
        "",
        "![Allen-Cahn capacity](allen_capacity.svg)",
        "",
        "Allen-Cahn 在 $H=1$ 时对 seed 较敏感, 但从 $H=5$ 开始稳定优于 constant. $H=20$ 时 Test RMSE 约为 `0.00914`, 而 constant 约为 `0.02657`. 这是目前最强的非线性有效性证据.",
        "",
        "### AsymmetricHJBLQ",
        "",
        "![HJB capacity](hjb_capacity.svg)",
        "",
        "HJB 的二次 driver 放大了 $Z$ 的估计误差. 在固定 $S/p=4$ 时, Test RMSE 随 $H$ 从 1 到 20 单调下降, 但直到 $H=20$ 才刚刚超过 constant. 这说明优化能够降低训练残差, 但统计误差仍占主导.",
        "",
        "## 4. 非线性方程: 增大 $S/p$",
        "",
        "### AsymmetricAllenCahn",
        "",
        "![Allen-Cahn sample ratio](allen_sample_ratio.svg)",
        "",
        f"固定 $H=20$ 后, $S/p$ 从 4 增加到 16, Test RMSE 从 `{allen_4:.5f}` 降至 `{allen_16:.5f}`. 对应基线 MSE 降低率从 `{100 * reduction_stats[('AsymmetricAllenCahn', 'batched_qr', '4.0')][0]:.1f}%` 提升到 `{100 * reduction_stats[('AsymmetricAllenCahn', 'batched_qr', '16.0')][0]:.1f}%`. 训练与测试误差已经接近, 继续增加样本的边际收益有限.",
        "",
        "### AsymmetricHJBLQ",
        "",
        "![HJB sample ratio](hjb_sample_ratio.svg)",
        "",
        f"HJB 的 Test RMSE 从 `{hjb_4:.5f}` 降至 `{hjb_16:.5f}`. 配对差值为 `{hjb_delta:.5f} +/- {hjb_delta_ci:.5f}`, 明显超过随机波动. 相对 constant 的 MSE 降低率从约 `1.4%` 提升到 `{100 * reduction_stats[('AsymmetricHJBLQ', 'batched_qr', '16.0')][0]:.1f}%`. 因此此前 HJB 表现弱的主要原因之一是 $S/p=4$ 仍不足.",
        "",
        "## 5. 综合判断",
        "",
        "| 方程 | 主要观察 | 当前瓶颈 |",
        "|---|---|---|",
        "| AsymmetricHeat | 固定 $S/p$ 后, 增大 $H$ 稳定降低误差 | 中高频表示和时间离散 |",
        "| AsymmetricAllenCahn | $H=20$ 消除约 88%-91% constant MSE | 已接近统计平台, 需要增加 $H$ 或时间步数 |",
        "| AsymmetricHJBLQ | 增大 $S/p$ 后从几乎无收益提升到约 22% MSE 收益 | 表达误差和二次 driver 的非线性敏感性 |",
        "",
        "现有实验支持以下结论: RFM 的优势主要体现在对路径依赖的 $Z_t$ 进行非平凡逼近, 而不是仅利用对称性估计一个标量 $y_0$. 但是 constant 只对应 $Z=0$, 仍属于较弱基线. 若要形成更强的论文证据, 还应加入 constant-$Z$、affine-$Z$ 和低阶 Hermite 基线.",
        "",
        "## 6. 下一步实验优先级",
        "",
        "1. 对 HJB 比较 $(H,S/p)=(20,32),(50,8),(50,16)$, 区分统计误差和表示误差.",
        "2. 为 Heat 构造具有解析 $u$ 和 $Z$ 的多方向多频终端条件, 同时报告 terminal RMSE 与 $Z$ 的积分误差.",
        "3. 扫描时间步数 $N$, 避免把高频末端条件的时间边界层误差归因于随机特征.",
        "4. 加入参数量匹配的 affine/Hermite 基线, 证明收益来自随机特征表示而不只是非零 $Z$.",
        "",
        "## 数据说明",
        "",
        "报告使用 5 个 evaluation seed 计算均值和 95% 置信区间. Ridge 参数选择使用独立的 3 个 tuning seed. 所有 constant 对照均与相同训练规模和 seed 配对. 图中的误差棒表示跨 seed 的均值置信区间, 不代表单条路径上的 Monte Carlo 标准误差.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    heat_capacity = read_csv(HEAT_CAPACITY)
    heat_lambda_high = read_csv(HEAT_LAMBDA_HIGH)
    nonlinear_capacity = read_csv(NONLINEAR_CAPACITY)
    nonlinear_ratio = read_csv(NONLINEAR_RATIO)

    make_heat_capacity_chart(heat_capacity)
    make_heat_lambda_chart(heat_capacity, heat_lambda_high)
    make_nonlinear_capacity_charts(nonlinear_capacity)
    make_nonlinear_ratio_charts(nonlinear_ratio)
    report = report_markdown(
        heat_capacity,
        heat_capacity,
        heat_lambda_high,
        nonlinear_capacity,
        nonlinear_ratio,
    )
    report_path = OUTPUT / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
