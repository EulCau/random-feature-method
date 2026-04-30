#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "test" / "result"
DEFAULT_ARTICLE_FIGURE_DIR = ROOT / "article" / "tex" / "figures"

REFS = {
    "Heat": {20: 0.385543289429532, 50: 0.375116802253964, 100: 0.371527882126961},
    "BSM": {50: 0.04921354, 100: 0.04880920},
    "HJBLQ": {50: 3.882006682195226, 100: 4.590161724604434},
    "AllenCahn": {50: 0.09908593, 100: 0.05278464},
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


class Pdf:
    def __init__(self, path: Path, width: int = 720, height: int = 460):
        self.path = path
        self.w = width
        self.h = height
        self.ops: list[str] = []

    def line(self, x1, y1, x2, y2, color=(0, 0, 0), width=1.0):
        self.ops.append(f"{color[0]} {color[1]} {color[2]} RG {width} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def polyline(self, pts, color=(0, 0, 0), width=1.5):
        if len(pts) < 2:
            return
        parts = [f"{color[0]} {color[1]} {color[2]} RG {width} w {pts[0][0]:.2f} {pts[0][1]:.2f} m"]
        parts.extend(f"{x:.2f} {y:.2f} l" for x, y in pts[1:])
        parts.append("S")
        self.ops.append(" ".join(parts))

    def circle(self, x, y, r=3, color=(0, 0, 0)):
        c = 0.55228475 * r
        self.ops.append(
            f"{color[0]} {color[1]} {color[2]} rg "
            f"{x+r:.2f} {y:.2f} m "
            f"{x+r:.2f} {y+c:.2f} {x+c:.2f} {y+r:.2f} {x:.2f} {y+r:.2f} c "
            f"{x-c:.2f} {y+r:.2f} {x-r:.2f} {y+c:.2f} {x-r:.2f} {y:.2f} c "
            f"{x-r:.2f} {y-c:.2f} {x-c:.2f} {y-r:.2f} {x:.2f} {y-r:.2f} c "
            f"{x+c:.2f} {y-r:.2f} {x+r:.2f} {y-c:.2f} {x+r:.2f} {y:.2f} c f"
        )

    def text(self, x, y, s, size=10, color=(0, 0, 0)):
        safe = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.ops.append(f"BT /F1 {size} Tf {color[0]} {color[1]} {color[2]} rg {x:.2f} {y:.2f} Td ({safe}) Tj ET")

    def write(self):
        stream = "\n".join(self.ops).encode("latin-1")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.w} {self.h}] "
                f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            ).encode("latin-1"),
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        out = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i, obj in enumerate(objects, 1):
            offsets.append(len(out))
            out.extend(f"{i} 0 obj\n".encode("latin-1"))
            out.extend(obj)
            out.extend(b"\nendobj\n")
        xref = len(out)
        out.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode("latin-1"))
        for off in offsets[1:]:
            out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
        out.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1"))
        self.path.write_bytes(out)


def tick_values(vmin: float, vmax: float, count=5, log=False):
    if log:
        lo = math.floor(math.log10(vmin))
        hi = math.ceil(math.log10(vmax))
        return [10**k for k in range(lo, hi + 1)]
    if vmax == vmin:
        return [vmin]
    step = (vmax - vmin) / (count - 1)
    return [vmin + i * step for i in range(count)]


def plot_panel(pdf: Pdf, rect, series, title, xlabel, ylabel, logx=False, logy=False, legend=True):
    x0, y0, w, h = rect
    xs = [x for s in series for x, _ in s["points"] if x > 0 or not logx]
    ys = [y for s in series for _, y in s["points"] if y > 0 or not logy]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin -= 1
        xmax += 1
    if ymin == ymax:
        ymin *= 0.9
        ymax *= 1.1

    if logx:
        xmin = max(xmin * 0.85, 1e-12)
        xmax *= 1.15
        lxmin, lxmax = math.log10(xmin), math.log10(xmax)
    else:
        pad = 0.08 * (xmax - xmin)
        xmin -= pad
        xmax += pad
    if logy:
        ymin = max(ymin * 0.7, 1e-12)
        ymax *= 1.5
        lymin, lymax = math.log10(ymin), math.log10(ymax)
    else:
        pad = 0.08 * (ymax - ymin)
        ymin -= pad
        ymax += pad

    def px(x):
        if logx:
            return x0 + (math.log10(x) - lxmin) / (lxmax - lxmin) * w
        return x0 + (x - xmin) / (xmax - xmin) * w

    def py(y):
        if logy:
            return y0 + (math.log10(y) - lymin) / (lymax - lymin) * h
        return y0 + (y - ymin) / (ymax - ymin) * h

    pdf.line(x0, y0, x0 + w, y0, width=0.8)
    pdf.line(x0, y0, x0, y0 + h, width=0.8)

    for t in tick_values(xmin, xmax, 5, logx):
        if t <= 0:
            continue
        x = px(t)
        pdf.line(x, y0, x, y0 - 4, width=0.6)
        label = f"{t:.0e}" if logx and t < 1 else f"{t:g}"
        pdf.text(x - 12, y0 - 18, label, 8)
    for t in tick_values(ymin, ymax, 5, logy):
        if t <= 0:
            continue
        y = py(t)
        pdf.line(x0 - 4, y, x0, y, width=0.6)
        label = f"{t:.0e}" if logy else f"{t:.3g}"
        pdf.text(x0 - 50, y - 3, label, 8)

    pdf.text(x0 + w * 0.38, y0 - 36, xlabel, 10)
    pdf.text(x0 - 52, y0 + h + 8, ylabel, 10)
    pdf.text(x0, y0 + h + 24, title, 12)

    for s in series:
        pts = [(px(x), py(y)) for x, y in s["points"] if (x > 0 or not logx) and (y > 0 or not logy)]
        if s.get("line", True):
            pdf.polyline(pts, s["color"], 1.6)
        for x, y in pts:
            pdf.circle(x, y, 2.6, s["color"])

    if legend:
        lx, ly = x0 + w - 126, y0 + h - 12
        for i, s in enumerate(series):
            yy = ly - 15 * i
            pdf.line(lx, yy + 3, lx + 18, yy + 3, s["color"], 1.6)
            pdf.circle(lx + 9, yy + 3, 2.4, s["color"])
            pdf.text(lx + 24, yy, s["label"], 8)


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


def plot_lm_residual(runs: list[Run], figure_dir: Path) -> Path:
    pdf = Pdf(figure_dir / "nonlinear-lm-residual.pdf")
    plot_panel(
        pdf,
        (78, 76, 560, 310),
        [
            {"label": "HJB-LQ", "points": accepted_points(first_run(runs, "hjblq_d100_h50_s16384.json")), "color": COLORS["HJBLQ"]},
            {"label": "Allen-Cahn", "points": accepted_points(first_run(runs, "allencahn_d100_h50_s16384.json")), "color": COLORS["AllenCahn"]},
        ],
        "LM residual history",
        "accepted step",
        "train RMSE",
        logy=True,
    )
    pdf.write()
    return pdf.path


def plot_sample_size(runs: list[Run], figure_dir: Path) -> Path:
    groups = {
        "Heat": ["heat_d100_h50_s4096.json", "heat_d100_h50_s8192.json", "heat_d100_h50_s16384.json"],
        "BSM": ["bsm_d100_h50_s8192.json", "bsm_d100_h50_s16384.json"],
        "HJBLQ": ["hjblq_d100_h50_s4096.json", "hjblq_d100_h50_s8192.json", "hjblq_d100_h50_s16384.json"],
        "AllenCahn": ["allencahn_d100_h50_s8192.json", "allencahn_d100_h50_s16384.json"],
    }
    pdf = Pdf(figure_dir / "sample-size-generalization.pdf", 760, 500)
    rmse_series = []
    err_series = []
    for eq, cfgs in groups.items():
        rr = [first_run(runs, cfg) for cfg in cfgs]
        ref = REFS[eq][100]
        rmse_series.append({"label": DISPLAY[eq], "points": [(r.samples, r.test_rmse) for r in rr], "color": COLORS[eq]})
        err_series.append({"label": DISPLAY[eq], "points": [(r.samples, abs(r.y0 - ref)) for r in rr], "color": COLORS[eq]})
    plot_panel(pdf, (80, 290, 560, 135), rmse_series, "Sample size and test residual", "S", "test RMSE", logx=True, logy=True)
    plot_panel(pdf, (80, 78, 560, 135), err_series, "Sample size and y0 error", "S", "abs error", logx=True, logy=True)
    pdf.write()
    return pdf.path


def plot_hidden_dim(runs: list[Run], figure_dir: Path) -> Path:
    groups = {
        "Heat": ["heat_d100_h8_s16384.json", "heat_d100_h20_s16384.json", "heat_d100_h50_s16384.json"],
        "BSM": ["bsm_d100_h20_s16384.json", "bsm_d100_h50_s16384.json"],
        "HJBLQ": ["hjblq_d100_h8_s16384.json", "hjblq_d100_h12_s16384.json", "hjblq_d100_h20_s16384.json", "hjblq_d100_h50_s16384.json"],
        "AllenCahn": ["allencahn_d100_h20_s16384.json", "allencahn_d100_h50_s16384.json"],
    }
    pdf = Pdf(figure_dir / "hidden-dim-generalization.pdf", 760, 500)
    rmse_series = []
    err_series = []
    for eq, cfgs in groups.items():
        rr = [first_run(runs, cfg) for cfg in cfgs]
        ref = REFS[eq][100]
        rmse_series.append({"label": DISPLAY[eq], "points": [(r.h, r.test_rmse) for r in rr], "color": COLORS[eq]})
        err_series.append({"label": DISPLAY[eq], "points": [(r.h, abs(r.y0 - ref)) for r in rr], "color": COLORS[eq]})
    plot_panel(pdf, (80, 290, 560, 135), rmse_series, "Random feature width and test residual", "H", "test RMSE", logx=True, logy=True)
    plot_panel(pdf, (80, 78, 560, 135), err_series, "Random feature width and y0 error", "H", "abs error", logx=True, logy=True)
    pdf.write()
    return pdf.path


def plot_seed_stability(runs: list[Run], figure_dir: Path) -> Path:
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
            points.append((idx + (j - 1) * 0.08, abs(run.y0 - ref)))
        labels.append((idx, DISPLAY[eq]))

    pdf = Pdf(figure_dir / "seed-stability.pdf", 680, 420)
    plot_panel(
        pdf,
        (84, 78, 500, 265),
        [{"label": "seed runs", "points": points, "color": COLORS["RFM"], "line": False}],
        "Seed stability of y0",
        "equation",
        "absolute y0 error",
        logy=True,
        legend=False,
    )
    for x, label in labels:
        pdf.text(84 + (x - 1) / 3 * 500 - 24, 44, label, 9)
    pdf.write()
    return pdf.path


def plot_deepbsde_compare(runs: list[Run], figure_dir: Path) -> Path:
    hjb = first_run(runs, "hjblq_d100_h50_s16384.json")
    allen = first_run(runs, "allencahn_d100_h50_s16384.json")
    hjb_ref = REFS["HJBLQ"][100]
    allen_ref = REFS["AllenCahn"][100]

    pdf = Pdf(figure_dir / "deepbsde-rfm-comparison.pdf", 760, 620)
    plot_panel(
        pdf,
        (76, 392, 260, 150),
        [
            {"label": "DeepBSDE", "points": [(t, loss) for _, loss, _, t in DEEP_HJB], "color": COLORS["DeepBSDE"]},
            {"label": "RFM", "points": [(hjb.time_ms / 1000, hjb.rmse * hjb.rmse)], "color": COLORS["RFM"], "line": False},
        ],
        "HJB-LQ loss",
        "time (s)",
        "loss",
        logy=True,
    )
    plot_panel(
        pdf,
        (448, 392, 230, 150),
        [
            {"label": "DeepBSDE", "points": [(t, abs(y0 - hjb_ref)) for _, _, y0, t in DEEP_HJB], "color": COLORS["DeepBSDE"]},
            {"label": "RFM", "points": [(hjb.time_ms / 1000, abs(hjb.y0 - hjb_ref))], "color": COLORS["RFM"], "line": False},
        ],
        "HJB-LQ y0 error",
        "time (s)",
        "abs error",
        logy=True,
    )
    plot_panel(
        pdf,
        (76, 112, 260, 150),
        [
            {"label": "DeepBSDE", "points": [(t, loss) for _, loss, _, t in DEEP_ALLEN], "color": COLORS["DeepBSDE"]},
            {"label": "RFM", "points": [(allen.time_ms / 1000, allen.rmse * allen.rmse)], "color": COLORS["RFM"], "line": False},
        ],
        "Allen-Cahn loss",
        "time (s)",
        "loss",
        logy=True,
    )
    plot_panel(
        pdf,
        (448, 112, 230, 150),
        [
            {"label": "DeepBSDE", "points": [(t, abs(y0 - allen_ref)) for _, _, y0, t in DEEP_ALLEN], "color": COLORS["DeepBSDE"]},
            {"label": "RFM", "points": [(allen.time_ms / 1000, abs(allen.y0 - allen_ref))], "color": COLORS["RFM"], "line": False},
        ],
        "Allen-Cahn y0 error",
        "time (s)",
        "abs error",
        logy=True,
    )
    pdf.write()
    return pdf.path


def power_slope(points: list[tuple[float, float]]) -> float:
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)


def plot_dimension_time(summaries: list[DimensionSummary], figure_dir: Path) -> Path:
    by_eq: dict[str, list[DimensionSummary]] = {}
    for item in summaries:
        by_eq.setdefault(item.equation, []).append(item)

    series = []
    for eq in ["Heat", "BSM", "HJBLQ", "AllenCahn"]:
        arr = sorted(by_eq.get(eq, []), key=lambda item: item.dim)
        if not arr:
            continue
        points = [(item.dim, item.time_ms_mean) for item in arr]
        slope = power_slope(points)
        series.append({"label": f"{DISPLAY[eq]} p={slope:.2f}", "points": points, "color": COLORS[eq]})

    pdf = Pdf(figure_dir / "dimension-time-scaling.pdf", 720, 460)
    plot_panel(
        pdf,
        (86, 82, 520, 295),
        series,
        "Runtime scaling with dimension",
        "dimension d",
        "time (ms)",
        logx=True,
        logy=True,
    )
    pdf.text(86, 40, "p is fitted from time = C d^p on d = 20, 50, 100.", 9)
    pdf.write()
    return pdf.path


def main() -> int:
    args = parse_args()
    result_path = args.result.resolve() if args.result else latest_result_file()
    figure_dir = args.figure_dir.resolve()
    article_dir = args.article_figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_copy:
        article_dir.mkdir(parents=True, exist_ok=True)

    runs, summaries = parse_result(result_path)
    outputs = [
        plot_lm_residual(runs, figure_dir),
        plot_sample_size(runs, figure_dir),
        plot_hidden_dim(runs, figure_dir),
        plot_seed_stability(runs, figure_dir),
        plot_deepbsde_compare(runs, figure_dir),
        plot_dimension_time(summaries, figure_dir),
    ]

    if not args.no_copy:
        for path in outputs:
            shutil.copy2(path, article_dir / path.name)

    print(f"result: {result_path}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
