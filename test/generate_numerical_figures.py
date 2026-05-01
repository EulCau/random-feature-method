#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "test" / "result"
DEFAULT_ARTICLE_FIGURE_DIR = ROOT / "article" / "tex" / "figures"

REFS = {
    "Heat": {
        20: 0.385543289429532,
        30: 0.379812405815246,
        50: 0.375116802253964,
        100: 0.371527882126961,
        200: 0.369711212329119,
        300: 0.369102310996184,
    },
    "BSM": {
        20: 0.05173661,
        30: 0.05021835,
        50: 0.04921751,
        100: 0.04881270,
        200: 0.04877301,
        300: 0.04876993,
    },
    "HJBLQ": {
        20: 2.921014735738689,
        30: 3.351223237182463,
        50: 3.882006682195226,
        100: 4.590161724604434,
        200: 5.290814736008237,
        300: 5.698781231260821,
    },
    "AllenCahn": {
        20: 0.20635880,
        30: 0.15199385,
        50: 0.09908593,
        100: 0.05278464,
        200: 0.02724542,
        300: 0.01835709,
    },
}

DISPLAY = {
    "Heat": "Heat",
    "BSM": "BSM",
    "HJBLQ": "HJB-LQ",
    "AllenCahn": "Allen-Cahn",
}

COLORS = {
    "Heat": (0.12, 0.36, 0.62),
    "BSM": (0.78, 0.36, 0.12),
    "HJBLQ": (0.22, 0.48, 0.24),
    "AllenCahn": (0.58, 0.22, 0.55),
    "DeepBSDE": (0.75, 0.22, 0.15),
    "RFM": (0.10, 0.28, 0.55),
}

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

DEEP_HJB = [
    (100, 17.167496, 0.462873, 1.150),
    (200, 14.738851, 0.761065, 2.013),
    (300, 10.847116, 1.265573, 2.753),
    (400, 5.972445, 1.968510, 3.505),
    (500, 2.662984, 2.836105, 4.254),
    (600, 1.007146, 3.736159, 5.016),
    (700, 0.072763, 4.441928, 5.766),
    (800, 0.030459, 4.567953, 6.495),
    (900, 0.028222, 4.579910, 7.214),
    (1000, 0.028216, 4.581603, 7.930),
    (1100, 0.028610, 4.586910, 8.645),
    (1200, 0.027830, 4.583618, 9.364),
    (1300, 0.027505, 4.585285, 10.090),
    (1400, 0.027134, 4.586036, 10.812),
    (1500, 0.027612, 4.584651, 11.528),
    (1600, 0.026401, 4.585598, 12.244),
    (1700, 0.026220, 4.586278, 12.984),
    (1800, 0.026024, 4.586332, 13.739),
    (1900, 0.026034, 4.586622, 14.474),
    (2000, 0.026012, 4.586627, 15.221),
]

DEEP_ALLEN = [
    (100, 0.049345, 0.323025, 0.830),
    (200, 0.044001, 0.308723, 1.554),
    (300, 0.036535, 0.285738, 2.273),
    (400, 0.028204, 0.256332, 2.994),
    (500, 0.020289, 0.222964, 3.718),
    (600, 0.013923, 0.190379, 4.433),
    (700, 0.009690, 0.162434, 5.154),
    (800, 0.006898, 0.139353, 5.871),
    (900, 0.005046, 0.120123, 6.589),
    (1000, 0.003907, 0.104526, 7.306),
    (1100, 0.003171, 0.091811, 8.027),
    (1200, 0.002731, 0.082178, 8.745),
    (1300, 0.002469, 0.074067, 9.464),
    (1400, 0.002324, 0.068221, 10.180),
    (1500, 0.002229, 0.064202, 10.898),
    (1600, 0.002155, 0.061134, 11.615),
    (1700, 0.002154, 0.058570, 12.365),
    (1800, 0.002128, 0.056699, 13.078),
    (1900, 0.002120, 0.055484, 13.794),
    (2000, 0.002116, 0.054896, 14.514),
    (2100, 0.002117, 0.054034, 15.239),
    (2200, 0.002101, 0.054045, 15.954),
    (2300, 0.002082, 0.053368, 16.724),
    (2400, 0.002073, 0.053005, 17.578),
    (2500, 0.002061, 0.052545, 18.302),
    (2600, 0.002041, 0.052323, 19.034),
    (2700, 0.002033, 0.052334, 19.787),
    (2800, 0.002011, 0.052430, 20.542),
    (2900, 0.002027, 0.052464, 21.278),
    (3000, 0.002032, 0.052546, 21.998),
    (3100, 0.002027, 0.052511, 22.719),
    (3200, 0.002018, 0.052756, 23.438),
    (3300, 0.002013, 0.052816, 24.153),
    (3400, 0.002013, 0.052754, 24.869),
    (3500, 0.002013, 0.052768, 25.587),
    (3600, 0.002010, 0.052803, 26.304),
    (3700, 0.002009, 0.052842, 27.019),
    (3800, 0.002009, 0.052844, 27.735),
    (3900, 0.002009, 0.052845, 28.468),
    (4000, 0.002009, 0.052846, 29.192),
]


@dataclass
class Run:
    config: str
    equation: str
    dim: int
    intervals: int
    h: int
    samples: int
    seed: str
    y0: float
    rmse: float
    test_rmse: float
    time_ms: float
    lm_logs: list[dict]


@dataclass
class DimensionSummary:
    equation: str
    dim: int
    repeat_count: int
    y0_mean: float
    y0_std: float
    rmse_mean: float
    rmse_std: float
    test_rmse_mean: float
    test_rmse_std: float
    time_ms_mean: float
    time_ms_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis figures from rfm experiment results.")
    parser.add_argument("--result", type=Path, default=None, help="Result text file. Defaults to latest test/result/result_*.txt.")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_RESULT_DIR, help="Directory receiving generated figures.")
    parser.add_argument("--article-figure-dir", type=Path, default=DEFAULT_ARTICLE_FIGURE_DIR, help="Directory receiving copied figures.")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy generated figures into the thesis figure directory.")
    parser.add_argument("--interactive", action="store_true", help="Show each figure before saving so positions can be adjusted manually.")
    return parser.parse_args()


def latest_result_file() -> Path:
    files = sorted(DEFAULT_RESULT_DIR.glob("result_*.txt"))
    if not files:
        raise FileNotFoundError(f"no result_*.txt under {DEFAULT_RESULT_DIR}")
    return files[-1]


def parse_result(path: Path) -> tuple[list[Run], list[DimensionSummary]]:
    text = path.read_text(encoding="utf-8")
    runs: list[Run] = []
    for block in text.split("============================================================\n")[1:]:
        config_match = re.search(r"^config: (.*)$", block, re.M)
        summary_match = re.search(r"^summary: (.*)$", block, re.M)
        if not config_match or not summary_match:
            continue

        sm = dict(re.findall(r"([A-Za-z_]+)=([^,]+)", summary_match.group(1)))
        lm_logs = []
        for m in re.finditer(
            r"^\[LM\] iter=(\d+) retry=(\d+) loss=([^ ]+) error=([^ ]+) "
            r"trial_error=([^ ]+) lambda=([^ ]+) step_norm=([^ ]+) "
            r"accepted=(true|false) y_0=([^\n]+)$",
            block,
            re.M,
        ):
            lm_logs.append(
                {
                    "iter": int(m.group(1)),
                    "retry": int(m.group(2)),
                    "loss": float(m.group(3)),
                    "error": float(m.group(4)),
                    "trial_error": float(m.group(5)),
                    "lambda": float(m.group(6)),
                    "step_norm": float(m.group(7)),
                    "accepted": m.group(8) == "true",
                    "y0": float(m.group(9)),
                }
            )

        runs.append(
            Run(
                config=config_match.group(1),
                equation=sm["equation"],
                dim=int(sm["d"]),
                intervals=int(float(sm["N"])),
                h=int(sm["H"]),
                samples=int(sm["S"]),
                seed=re.search(r"^seed: (.*)$", block, re.M).group(1),
                y0=float(re.search(r"^y0 = ([^\n]+)$", block, re.M).group(1)),
                rmse=float(re.search(r"^rmse = ([^\n]+)$", block, re.M).group(1)),
                test_rmse=float(re.search(r"^test rmse = ([^\n]+)$", block, re.M).group(1)),
                time_ms=float(re.search(r"^total time: ([^ ]+) ms$", block, re.M).group(1)),
                lm_logs=lm_logs,
            )
        )

    summaries: list[DimensionSummary] = []
    marker = "equation,d,repeat_count,y0_mean,y0_std,rmse_mean,rmse_std,test_rmse_mean,test_rmse_std,time_ms_mean,time_ms_std"
    if marker in text:
        for line in text.split(marker, 1)[1].strip().splitlines():
            parts = line.split(",")
            if len(parts) != 11:
                continue
            summaries.append(
                DimensionSummary(
                    equation=parts[0],
                    dim=int(parts[1]),
                    repeat_count=int(parts[2]),
                    y0_mean=float(parts[3]),
                    y0_std=float(parts[4]),
                    rmse_mean=float(parts[5]),
                    rmse_std=float(parts[6]),
                    test_rmse_mean=float(parts[7]),
                    test_rmse_std=float(parts[8]),
                    time_ms_mean=float(parts[9]),
                    time_ms_std=float(parts[10]),
                )
            )
    return runs, summaries


def plot_panel(ax, series, title, xlabel, ylabel, logx=False, logy=False, legend=True):
    for item in series:
        points = [(x, y) for x, y in item["points"] if (x > 0 or not logx) and (y > 0 or not logy)]
        if not points:
            continue
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        linestyle = "-" if item.get("line", True) else "None"
        ax.plot(
            xs,
            ys,
            linestyle=linestyle,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=item["color"],
            label=item["label"],
        )

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.45)
    if legend:
        ax.legend(fontsize=8, frameon=False)


def save_after_adjustment(fig: Figure, path: Path, interactive: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if interactive:
        try:
            fig.canvas.manager.set_window_title(path.name)
        except AttributeError:
            pass
        fig.canvas.draw_idle()
        print(f"{path.name}: adjust the figure window, then close it to save.")
        plt.show(block=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def rel_error(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


def first_run(runs: list[Run], config: str, seed="C02E7A5B3F91A8C3") -> Run:
    for run in runs:
        if run.config == config and run.seed == seed:
            return run
    raise KeyError(config)


def accepted_points(run: Run):
    if not run.lm_logs:
        return []
    points = [(0, run.lm_logs[0]["error"])]
    accepted_index = 1
    for log in run.lm_logs:
        if log["accepted"]:
            points.append((accepted_index, log["trial_error"]))
            accepted_index += 1
    return points


def plot_lm_residual(runs: list[Run], figure_dir: Path) -> tuple[Path, Figure]:
    path = figure_dir / "nonlinear-lm-residual.pdf"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plot_panel(
        ax,
        [
            {"label": "HJB-LQ", "points": accepted_points(first_run(runs, "hjblq_d100_h50_s16384.json")), "color": COLORS["HJBLQ"]},
            {"label": "Allen-Cahn", "points": accepted_points(first_run(runs, "allencahn_d100_h50_s16384.json")), "color": COLORS["AllenCahn"]},
        ],
        "LM residual history",
        "accepted step",
        "train RMSE",
        logy=True,
    )
    fig.tight_layout()
    return path, fig


def plot_sample_size(runs: list[Run], figure_dir: Path) -> tuple[Path, Figure]:
    groups = {
        "Heat": ["heat_d100_h50_s4096.json", "heat_d100_h50_s8192.json", "heat_d100_h50_s16384.json"],
        "BSM": ["bsm_d100_h50_s8192.json", "bsm_d100_h50_s16384.json"],
        "HJBLQ": ["hjblq_d100_h50_s4096.json", "hjblq_d100_h50_s8192.json", "hjblq_d100_h50_s16384.json"],
        "AllenCahn": ["allencahn_d100_h50_s8192.json", "allencahn_d100_h50_s16384.json"],
    }
    path = figure_dir / "sample-size-generalization.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    ratio_series = []
    err_series = []
    for eq, cfgs in groups.items():
        rr = [first_run(runs, cfg) for cfg in cfgs]
        ref = REFS[eq][100]
        ratio_series.append({"label": DISPLAY[eq], "points": [(r.samples, r.test_rmse / r.rmse) for r in rr], "color": COLORS[eq]})
        err_series.append({"label": DISPLAY[eq], "points": [(r.samples, rel_error(r.y0, ref)) for r in rr], "color": COLORS[eq]})
    plot_panel(axes[0], ratio_series, "", "S", "test RMSE / train RMSE", logx=True, logy=True)
    plot_panel(axes[1], err_series, "", "S", "relative error", logx=True, logy=True)
    fig.tight_layout()
    return path, fig


def plot_hidden_dim(runs: list[Run], figure_dir: Path) -> tuple[Path, Figure]:
    groups = {
        "Heat": ["heat_d100_h8_s16384.json", "heat_d100_h20_s16384.json", "heat_d100_h50_s16384.json"],
        "BSM": ["bsm_d100_h20_s16384.json", "bsm_d100_h50_s16384.json"],
        "HJBLQ": ["hjblq_d100_h8_s16384.json", "hjblq_d100_h12_s16384.json", "hjblq_d100_h20_s16384.json", "hjblq_d100_h50_s16384.json"],
        "AllenCahn": ["allencahn_d100_h20_s16384.json", "allencahn_d100_h50_s16384.json"],
    }
    path = figure_dir / "hidden-dim-generalization.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    rmse_series = []
    err_series = []
    for eq, cfgs in groups.items():
        rr = [first_run(runs, cfg) for cfg in cfgs]
        ref = REFS[eq][100]
        rmse_series.append({"label": DISPLAY[eq], "points": [(r.h, r.test_rmse) for r in rr], "color": COLORS[eq]})
        err_series.append({"label": DISPLAY[eq], "points": [(r.h, rel_error(r.y0, ref)) for r in rr], "color": COLORS[eq]})
    plot_panel(axes[0], rmse_series, "", "H", "test RMSE", logx=True, logy=True)
    plot_panel(axes[1], err_series, "", "H", "relative error", logx=True, logy=True)
    fig.tight_layout()
    return path, fig


def plot_seed_stability(runs: list[Run], figure_dir: Path) -> tuple[Path, Figure]:
    configs = {
        "Heat": "heat_d100_h50_s16384.json",
        "BSM": "bsm_d100_h50_s16384.json",
        "HJBLQ": "hjblq_d100_h50_s16384.json",
        "AllenCahn": "allencahn_d100_h50_s16384.json",
    }
    points = []
    labels = []
    for idx, (eq, cfg) in enumerate(configs.items(), start=1):
        rr = [run for run in runs if run.config == cfg]
        ref = REFS[eq][100]
        for j, run in enumerate(rr):
            points.append((idx + (j - 1) * 0.08, rel_error(run.y0, ref)))
        labels.append((idx, DISPLAY[eq]))

    path = figure_dir / "seed-stability.pdf"
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    plot_panel(
        ax,
        [{"label": "seed runs", "points": points, "color": COLORS["RFM"], "line": False}],
        "",
        "equation",
        "relative y0 error",
        logy=True,
        legend=False,
    )
    ax.set_xticks([x for x, _ in labels])
    ax.set_xticklabels([label for _, label in labels])
    fig.tight_layout()
    return path, fig


def rfm_y0_relative_error_path(run: Run, reference: float) -> list[tuple[float, float]]:
    accepted = [log for log in run.lm_logs if log["accepted"]]
    if not accepted:
        return [(run.time_ms / 1000, rel_error(run.y0, reference))]
    y0_values = [log["y0"] for log in accepted] + [run.y0]
    interval = run.time_ms / 1000 / len(y0_values)
    return [((i + 1) * interval, rel_error(y0, reference)) for i, y0 in enumerate(y0_values)]


def plot_deepbsde_compare(runs: list[Run], figure_dir: Path) -> tuple[Path, Figure]:
    hjb = first_run(runs, "hjblq_d100_h50_s16384.json")
    allen = first_run(runs, "allencahn_d100_h50_s16384.json")
    hjb_ref = REFS["HJBLQ"][100]
    allen_ref = REFS["AllenCahn"][100]

    path = figure_dir / "deepbsde-rfm-comparison.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    plot_panel(
        axes[0],
        [
            {"label": "DeepBSDE", "points": [(t, rel_error(y0, hjb_ref)) for _, _, y0, t in DEEP_HJB], "color": COLORS["DeepBSDE"]},
            {"label": "RFM", "points": rfm_y0_relative_error_path(hjb, hjb_ref), "color": COLORS["RFM"]},
        ],
        "",
        "time (s)",
        "relative error",
        logy=True,
    )
    plot_panel(
        axes[1],
        [
            {"label": "DeepBSDE", "points": [(t, rel_error(y0, allen_ref)) for _, _, y0, t in DEEP_ALLEN], "color": COLORS["DeepBSDE"]},
            {"label": "RFM", "points": rfm_y0_relative_error_path(allen, allen_ref), "color": COLORS["RFM"]},
        ],
        "",
        "time (s)",
        "relative error",
        logy=True,
    )
    fig.tight_layout()
    return path, fig


def power_slope(points: list[tuple[float, float]]) -> float:
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)


def plot_dimension_time(summaries: list[DimensionSummary], figure_dir: Path) -> tuple[Path, Figure]:
    by_eq: dict[str, list[DimensionSummary]] = {}
    for item in summaries:
        by_eq.setdefault(item.equation, []).append(item)

    series = []
    for eq in ["Heat", "BSM", "HJBLQ", "AllenCahn"]:
        arr = sorted(by_eq.get(eq, []), key=lambda item: item.dim)
        if not arr:
            continue
        points = [(item.dim, item.time_ms_mean) for item in arr]
        series.append({"label": DISPLAY[eq], "points": points, "color": COLORS[eq]})

    path = figure_dir / "dimension-time-scaling.pdf"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plot_panel(
        ax,
        series,
        "",
        "dimension d",
        "time (ms)",
        logx=True,
        logy=True,
    )
    fig.tight_layout()
    return path, fig


def plot_dimension_relative_error(runs: list[Run], figure_dir: Path) -> tuple[Path, Figure]:
    seed_order = ["C02E7A5B3F91A8C3", "0000000000000001", "0000000000000002"]
    seed_offsets = {
        "C02E7A5B3F91A8C3": 0.965,
        "0000000000000001": 1.0,
        "0000000000000002": 1.035,
    }
    path = figure_dir / "dimension-relative-error.pdf"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for eq in ["Heat", "BSM", "HJBLQ", "AllenCahn"]:
        for seed in seed_order:
            seed_runs = [
                run
                for run in runs
                if run.equation == eq
                and run.seed == seed
                and run.h == 50
                and run.samples == 16384
                and run.dim in REFS[eq]
                and run.config.endswith("_h50_s16384.json")
            ]
            seed_runs.sort(key=lambda run: run.dim)
            xs = [run.dim * seed_offsets[seed] for run in seed_runs]
            ys = [rel_error(run.y0, REFS[eq][run.dim]) for run in seed_runs]
            if not xs:
                continue
            label = DISPLAY[eq] if seed == seed_order[0] else None
            ax.plot(
                xs,
                ys,
                linestyle="None",
                color=COLORS[eq],
                marker="o",
                markersize=3.8,
                alpha=0.72,
                label=label,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dimension d", fontsize=10)
    ax.set_ylabel("relative y0 error", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.45)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return path, fig


def main() -> int:
    args = parse_args()
    result_path = args.result.resolve() if args.result else latest_result_file()
    figure_dir = args.figure_dir.resolve()
    article_dir = args.article_figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_copy:
        article_dir.mkdir(parents=True, exist_ok=True)

    runs, summaries = parse_result(result_path)
    figure_builders = [
        lambda: plot_lm_residual(runs, figure_dir),
        lambda: plot_sample_size(runs, figure_dir),
        lambda: plot_hidden_dim(runs, figure_dir),
        lambda: plot_seed_stability(runs, figure_dir),
        lambda: plot_deepbsde_compare(runs, figure_dir),
        lambda: plot_dimension_time(summaries, figure_dir),
        lambda: plot_dimension_relative_error(runs, figure_dir),
    ]
    outputs = []
    for build in figure_builders:
        path, fig = build()
        outputs.append(save_after_adjustment(fig, path, args.interactive))

    if not args.no_copy:
        for path in outputs:
            shutil.copy2(path, article_dir / path.name)

    print(f"result: {result_path}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
