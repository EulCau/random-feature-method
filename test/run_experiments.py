#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [
    "C02E7A5B3F91A8C3",
    "0000000000000001",
    "0000000000000002",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run rfm_solver on every JSON config in test/bin/config."
    )
    parser.add_argument(
        "--solver",
        type=Path,
        default=root / "bin" / "rfm_solver",
        help="Path to rfm_solver executable.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=root / "bin" / "config",
        help="Directory containing JSON config files.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=root / "result",
        help="Directory where result_{timestamp}.txt is written.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Hex seeds. The first seed is used for all configs by default.",
    )
    parser.add_argument(
        "--all-seeds-for-all",
        action="store_true",
        help="Run every config with every seed.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for config selection.",
    )
    return parser.parse_args()


def is_standard_config(config_path: Path) -> bool:
    name = config_path.stem.lower()
    return "d100" in name and "h50" in name and "s16384" in name


def read_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_config_summary(config_path: Path) -> str:
    cfg = read_config(config_path)

    eqn = cfg["eqn_config"]
    solver = cfg["solver_config"]
    return (
        f"equation={eqn['equation_name']}, "
        f"linear={eqn['is_linear']}, "
        f"d={eqn['dimension']}, "
        f"T={eqn['total_time']}, "
        f"N={eqn['num_time_intervals']}, "
        f"H={solver['hidden_dim']}, "
        f"S={solver['sample_size']}, "
        f"use_linear_solver={solver['use_linear_solver']}"
    )


def is_dimension_config(config_path: Path, cfg: dict) -> bool:
    solver = cfg["solver_config"]
    eqn = cfg["eqn_config"]
    name = config_path.stem.lower()

    return (
        "_h50_s16384" in name
        and solver["hidden_dim"] == 50
        and solver["sample_size"] == 16384
        and eqn["dimension"] in {20, 50, 100}
    )


def config_group(config_path: Path, cfg: dict) -> tuple[int, str]:
    name = config_path.stem.lower()

    if name in {"heat_d100", "bsm_d100", "hjb_lq_d100", "allencahn_d100"}:
        return (0, name)
    if is_dimension_config(config_path, cfg):
        return (2, name)
    if is_standard_config(config_path):
        return (0, name)
    if "_h" in name or "_s" in name:
        return (1, name)
    return (3, name)


def ordered_configs(configs: list[Path]) -> list[Path]:
    cached = {path: read_config(path) for path in configs}
    return sorted(configs, key=lambda path: config_group(path, cached[path]))


def parse_metrics(output: str) -> dict[str, float | str]:
    patterns = {
        "y0": r"^y0 = ([^\n]+)$",
        "rmse": r"^rmse = ([^\n]+)$",
        "test_rmse": r"^test rmse = ([^\n]+)$",
        "time_ms": r"^total time: ([^ ]+) ms$",
        "device": r"^device: ([^\n]+)$",
    }
    metrics: dict[str, float | str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.MULTILINE)
        if not match:
            continue
        value = match.group(1)
        metrics[key] = value if key == "device" else float(value)
    return metrics


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def run_one(solver: Path, config_path: Path, seed: str) -> tuple[int, str]:
    command = [
        str(solver),
        "--config",
        str(config_path),
        "--seed",
        seed,
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)

    return process.wait(), "".join(lines)


def main() -> int:
    args = parse_args()
    solver = args.solver.resolve()
    config_dir = args.config_dir.resolve()
    result_dir = args.result_dir.resolve()

    if not solver.exists():
        raise FileNotFoundError(f"solver not found: {solver}")
    if not config_dir.exists():
        raise FileNotFoundError(f"config directory not found: {config_dir}")

    configs = ordered_configs(sorted(config_dir.glob(args.pattern)))
    if not configs:
        raise RuntimeError(f"no configs found under {config_dir} with pattern {args.pattern}")

    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = result_dir / f"result_{timestamp}.txt"

    with result_path.open("w", encoding="utf-8") as out:
        out.write(f"timestamp: {timestamp}\n")
        out.write(f"solver: {solver}\n")
        out.write(f"config_dir: {config_dir}\n")
        out.write(f"pattern: {args.pattern}\n")
        out.write(f"seeds: {' '.join(args.seeds)}\n")
        out.write(f"all_seeds_for_all: {args.all_seeds_for_all}\n\n")
        out.write("Run order: baseline, parameter scans, dimension scans, remaining configs.\n")
        out.write("Dimension scan configs are repeated for every listed seed and summarized at the end.\n\n")

        dimension_records: list[dict] = []

        for config_path in configs:
            cfg = read_config(config_path)
            dimension_config = is_dimension_config(config_path, cfg)
            seeds = args.seeds if args.all_seeds_for_all or is_standard_config(config_path) or dimension_config else args.seeds[:1]
            summary = read_config_summary(config_path)

            for seed in seeds:
                command_text = f"{solver} --config {config_path} --seed {seed}"
                header = (
                    "\n"
                    "============================================================\n"
                    f"config: {config_path.name}\n"
                    f"summary: {summary}\n"
                    f"seed: {seed}\n"
                    f"command: {command_text}\n"
                    "------------------------------------------------------------\n"
                )
                print(header, end="")
                out.write(header)
                out.flush()

                return_code, output = run_one(solver, config_path, seed)
                out.write(output)
                out.write(f"\nreturn_code: {return_code}\n")
                out.flush()

                if dimension_config and return_code == 0:
                    metrics = parse_metrics(output)
                    if metrics:
                        dimension_records.append({
                            "config": config_path.name,
                            "equation": cfg["eqn_config"]["equation_name"],
                            "dimension": cfg["eqn_config"]["dimension"],
                            "hidden_dim": cfg["solver_config"]["hidden_dim"],
                            "sample_size": cfg["solver_config"]["sample_size"],
                            "seed": seed,
                            **metrics,
                        })

                if return_code != 0:
                    print(f"warning: run failed with return_code={return_code}", file=sys.stderr)

        if dimension_records:
            out.write("\n")
            out.write("============================================================\n")
            out.write("DIMENSION SCAN AVERAGE SUMMARY\n")
            out.write("------------------------------------------------------------\n")
            groups: dict[tuple[str, int], list[dict]] = {}
            for record in dimension_records:
                groups.setdefault((record["equation"], record["dimension"]), []).append(record)

            out.write(
                "equation,d,repeat_count,y0_mean,y0_std,rmse_mean,rmse_std,"
                "test_rmse_mean,test_rmse_std,time_ms_mean,time_ms_std\n"
            )
            for (equation, dimension), records in sorted(groups.items()):
                y0_values = [float(r["y0"]) for r in records]
                rmse_values = [float(r["rmse"]) for r in records]
                test_values = [float(r["test_rmse"]) for r in records]
                time_values = [float(r["time_ms"]) for r in records]
                out.write(
                    f"{equation},{dimension},{len(records)},"
                    f"{mean(y0_values):.8g},{std(y0_values):.4g},"
                    f"{mean(rmse_values):.8g},{std(rmse_values):.4g},"
                    f"{mean(test_values):.8g},{std(test_values):.4g},"
                    f"{mean(time_values):.8g},{std(time_values):.4g}\n"
                )

    print(f"\nresults written to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
