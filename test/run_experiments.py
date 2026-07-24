#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


DEFAULT_EVALUATION_SEEDS = [
    "C02E7A5B3F91A8C3",
    "0000000000000001",
    "0000000000000002",
    "0000000000000003",
    "0000000000000004",
]

DEFAULT_TUNING_SEEDS = [
    "A5A5A5A5A5A5A5A5",
    "5A5A5A5A5A5A5A5A",
    "0123456789ABCDEF",
]


def parse_args() -> argparse.Namespace:
    test_dir = Path(__file__).resolve().parent
    root = test_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run asymmetric linear and nonlinear PDE experiments, including "
            "a high-frequency HJB variant."
        )
    )
    parser.add_argument("--solver", type=Path, default=test_dir / "bin" / "rfm_solver")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=root / "config" / "asymmetric_heat_d100.json",
    )
    parser.add_argument("--result-dir", type=Path, default=test_dir / "result")
    parser.add_argument("--seeds", nargs="+", default=DEFAULT_EVALUATION_SEEDS)
    parser.add_argument("--tuning-seeds", nargs="+", default=DEFAULT_TUNING_SEEDS)
    parser.add_argument(
        "--hidden-dims", nargs="+", type=int, default=[5, 10, 20, 50]
    )
    parser.add_argument(
        "--sample-ratios", nargs="+", type=float, default=[2.0, 4.0, 8.0]
    )
    parser.add_argument("--tuning-ratio", type=float, default=4.0)
    parser.add_argument(
        "--normalized-lambdas",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 1.0, 3.0, 10.0],
        help=(
            "Ridge strengths for the normalized objective. The solver receives "
            "raw_lambda = sample_size * normalized_lambda."
        ),
    )
    parser.add_argument(
        "--run-completed-heat-ratio-scan",
        action="store_true",
        help="Also rerun the completed AsymmetricHeat H and S/p scan.",
    )
    parser.add_argument(
        "--run-heat-lambda-continuation",
        action="store_true",
        help="Run the additional AsymmetricHeat normalized-lambda scan.",
    )
    parser.add_argument(
        "--skip-nonlinear-variants",
        action="store_true",
        help=(
            "Skip AsymmetricAllenCahn, AsymmetricHJBLQ, and "
            "HighFrequencyAsymmetricHJBLQ experiments."
        ),
    )
    parser.add_argument(
        "--nonlinear-hidden-dims", nargs="+", type=int, default=[20]
    )
    parser.add_argument(
        "--nonlinear-sample-ratios", nargs="+", type=float, default=[4.0, 8.0, 16.0]
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def exact_y0(cfg: dict) -> float:
    eqn = cfg["eqn_config"]
    params = eqn["params"]
    if float(params.get("x_init", 0.0)) != 0.0:
        raise ValueError("the analytic y0 formula currently requires x_init=0")
    count = int(params.get("direction_count", 1))
    weight = float(params.get("cos_weight", 0.5))
    frequency = float(params.get("cos_frequency", 1.0))
    final_time = float(eqn["total_time"])
    return weight * math.sqrt(count) * math.exp(-0.5 * frequency**2 * final_time)


def parameter_count(dimension: int, hidden_dim: int) -> int:
    return 1 + dimension * hidden_dim


def sample_size_for_ratio(dimension: int, hidden_dim: int, ratio: float) -> int:
    return math.ceil(ratio * parameter_count(dimension, hidden_dim))


def make_config(
    base: dict,
    method: str,
    hidden_dim: int,
    sample_size: int,
    raw_lambda: float,
) -> dict:
    cfg = json.loads(json.dumps(base))
    options = cfg["solver_config"]
    options["hidden_dim"] = hidden_dim
    options["sample_size"] = sample_size
    options["initial_lambda"] = raw_lambda
    options["linear"]["solver"] = method
    options["linear"]["ridge_lambda"] = raw_lambda
    return cfg


METRIC_PATTERNS = {
    "y0": r"^y0 = ([^\n]+)$",
    "fit_rmse": r"^rmse = ([^\n]+)$",
    "test_rmse": r"^test rmse = ([^\n]+)$",
    "time_ms": r"^total time: ([^ ]+) ms$",
}


def parse_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"missing metric {key!r} in solver output:\n{output}")
        metrics[key] = float(match.group(1))
    return metrics


def run_solver(solver: Path, config_path: Path, seed: str) -> dict[str, float]:
    process = subprocess.run(
        [str(solver), "--config", str(config_path), "--seed", seed],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"solver failed with return code {process.returncode}:\n{process.stdout}"
        )
    return parse_metrics(process.stdout)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def ci95(values: list[float]) -> float:
    return 1.96 * sample_std(values) / math.sqrt(len(values))


def run_nonlinear_variant_experiments(
    args: argparse.Namespace,
    solver: Path,
    result_dir: Path,
    timestamp: str,
) -> tuple[Path, Path]:
    test_dir = Path(__file__).resolve().parent
    config_paths = [
        test_dir.parent / "config" / "asymmetric_allencahn_d100.json",
        test_dir.parent / "config" / "asymmetric_hjblq_d100.json",
        test_dir.parent / "config" / "high_frequency_asymmetric_hjblq_d100.json",
    ]
    records: list[dict[str, str | int | float]] = []

    with tempfile.TemporaryDirectory(prefix="asymmetric_nonlinear_") as tmp:
        tmp_dir = Path(tmp)
        for config_path in config_paths:
            base = load_json(config_path)
            equation = str(base["eqn_config"]["equation_name"])
            dimension = int(base["eqn_config"]["dimension"])
            for hidden_dim in args.nonlinear_hidden_dims:
                params = parameter_count(dimension, hidden_dim)
                for requested_ratio in args.nonlinear_sample_ratios:
                    sample_size = sample_size_for_ratio(dimension, hidden_dim, requested_ratio)
                    ratio = sample_size / params
                    for method in ("batched_qr", "constant"):
                        for seed in args.seeds:
                            cfg = json.loads(json.dumps(base))
                            cfg["solver_config"]["hidden_dim"] = hidden_dim
                            cfg["solver_config"]["sample_size"] = sample_size
                            cfg["solver_config"]["nonlinear"]["step_solver"] = method
                            generated = tmp_dir / (
                                f"{equation}_{method}_h{hidden_dim}_s{sample_size}_{seed}.json"
                            )
                            with generated.open("w", encoding="utf-8") as stream:
                                json.dump(cfg, stream, indent=2)
                            metrics = run_solver(solver, generated, seed)
                            records.append({
                                "equation": equation,
                                "method": method,
                                "hidden_dim": hidden_dim,
                                "parameter_count": params,
                                "sample_ratio": ratio,
                                "sample_size": sample_size,
                                "seed": seed,
                                **metrics,
                            })
                            print(
                                f"[{len(records):03d}] {equation:22s} {method:10s} "
                                f"H={hidden_dim:3d} S/p={ratio:4.1f} "
                                f"test={metrics['test_rmse']:.6g}"
                            )

    baselines = {
        (str(record["equation"]), int(record["hidden_dim"]),
         int(record["sample_size"]), str(record["seed"])): float(record["test_rmse"])
        for record in records
        if record["method"] == "constant"
    }
    for record in records:
        key = (str(record["equation"]), int(record["hidden_dim"]),
               int(record["sample_size"]), str(record["seed"]))
        baseline = baselines[key]
        ratio = float(record["test_rmse"]) / baseline
        record["test_rmse_ratio"] = ratio
        record["test_mse_reduction"] = 1.0 - ratio**2

    csv_path = result_dir / f"asymmetric_nonlinear_runs_{timestamp}.csv"
    summary_path = result_dir / f"asymmetric_nonlinear_summary_{timestamp}.txt"
    fields = [
        "equation", "method", "hidden_dim", "parameter_count", "sample_ratio",
        "sample_size", "seed", "y0", "fit_rmse", "test_rmse",
        "test_rmse_ratio", "test_mse_reduction", "time_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    groups: dict[tuple[str, str, int, float], list[dict]] = defaultdict(list)
    for record in records:
        groups[(str(record["equation"]), str(record["method"]),
                int(record["hidden_dim"]), float(record["sample_ratio"]))].append(record)
    with summary_path.open("w", encoding="utf-8") as out:
        out.write("Asymmetric nonlinear PDE experiments\n")
        out.write(f"timestamp={timestamp}\nsolver={solver}\n")
        out.write(f"evaluation_seeds={','.join(args.seeds)}\n")
        out.write("step_solver=batched_qr; baseline=constant alpha\n")
        out.write("initial_lambda=Levenberg-Marquardt damping, not ridge regularization\n\n")
        out.write(
            "equation,method,H,p,S_over_p,S,repeats,y0_mean,y0_ci95,fit_rmse_mean,"
            "test_rmse_mean,test_rmse_ci95,test_rmse_ratio_mean,"
            "test_mse_reduction_mean,time_ms_mean\n"
        )
        for key, group in sorted(groups.items()):
            equation, method, hidden_dim, ratio = key
            first = group[0]
            y0_values = [float(record["y0"]) for record in group]
            fit = [float(record["fit_rmse"]) for record in group]
            test = [float(record["test_rmse"]) for record in group]
            rmse_ratios = [float(record["test_rmse_ratio"]) for record in group]
            reductions = [float(record["test_mse_reduction"]) for record in group]
            times = [float(record["time_ms"]) for record in group]
            out.write(
                f"{equation},{method},{hidden_dim},{first['parameter_count']},{ratio:.8g},"
                f"{first['sample_size']},{len(group)},{mean(y0_values):.8g},{ci95(y0_values):.4g},"
                f"{mean(fit):.8g},{mean(test):.8g},{ci95(test):.4g},"
                f"{mean(rmse_ratios):.8g},{mean(reductions):.8g},{mean(times):.8g}\n"
            )
    return csv_path, summary_path


def main() -> int:
    args = parse_args()
    solver = args.solver.resolve()
    base_path = args.base_config.resolve()
    result_dir = args.result_dir.resolve()
    if not solver.is_file():
        raise FileNotFoundError(f"solver not found: {solver}")
    if not base_path.is_file():
        raise FileNotFoundError(f"base config not found: {base_path}")

    base = load_json(base_path)
    if base["eqn_config"]["equation_name"] != "AsymmetricHeat":
        raise ValueError("base config must use AsymmetricHeat")
    if any(value <= 0 for value in args.hidden_dims):
        raise ValueError("hidden dimensions must be positive")
    if any(value < 1.0 for value in args.sample_ratios + [args.tuning_ratio]):
        raise ValueError("sample ratios must be at least one")
    if any(value <= 0 for value in args.normalized_lambdas):
        raise ValueError("normalized lambdas must be positive")
    if any(value <= 0 for value in args.nonlinear_hidden_dims):
        raise ValueError("nonlinear hidden dimensions must be positive")
    if any(value < 1.0 for value in args.nonlinear_sample_ratios):
        raise ValueError("nonlinear sample ratios must be at least one")

    dimension = int(base["eqn_config"]["dimension"])
    exact = exact_y0(base)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = result_dir / f"asymmetric_heat_ratio_runs_{timestamp}.csv"
    summary_path = result_dir / f"asymmetric_heat_ratio_summary_{timestamp}.txt"
    records: list[dict[str, str | int | float]] = []
    metric_cache: dict[tuple[str, int, int, float, str], dict[str, float]] = {}

    with tempfile.TemporaryDirectory(prefix="asymmetric_heat_ratio_") as tmp:
        tmp_dir = Path(tmp)

        def evaluate(
            stage: str,
            method: str,
            hidden_dim: int,
            ratio: float,
            normalized_lambda: float,
            seed: str,
        ) -> dict[str, str | int | float]:
            params = parameter_count(dimension, hidden_dim)
            sample_size = sample_size_for_ratio(dimension, hidden_dim, ratio)
            raw_lambda = sample_size * normalized_lambda
            key = (method, hidden_dim, sample_size, raw_lambda, seed)

            if key in metric_cache:
                metrics = metric_cache[key]
                reused = True
            else:
                cfg = make_config(base, method, hidden_dim, sample_size, raw_lambda)
                config_path = tmp_dir / (
                    f"{method}_h{hidden_dim}_s{sample_size}_nl{normalized_lambda:g}_{seed}.json"
                )
                with config_path.open("w", encoding="utf-8") as stream:
                    json.dump(cfg, stream, indent=2)
                metrics = run_solver(solver, config_path, seed)
                metric_cache[key] = metrics
                reused = False

            record: dict[str, str | int | float] = {
                "stage": stage,
                "method": method,
                "hidden_dim": hidden_dim,
                "parameter_count": params,
                "sample_ratio": sample_size / params,
                "sample_size": sample_size,
                "normalized_lambda": normalized_lambda,
                "raw_lambda": raw_lambda,
                "seed": seed,
                **metrics,
                "y0_abs_error": abs(metrics["y0"] - exact),
            }
            records.append(record)
            print(
                f"[{len(records):03d}] {stage:12s} {method:10s} H={hidden_dim:3d} "
                f"S/p={sample_size / params:4.1f} S={sample_size:6d} "
                f"nlambda={normalized_lambda:8.1e} test={metrics['test_rmse']:.6g}"
                f"{' [reused]' if reused else ''}"
            )
            return record

        # The smaller lambda grid and full ratio scan were completed previously.
        if args.run_heat_lambda_continuation:
            for hidden_dim in args.hidden_dims:
                for normalized_lambda in args.normalized_lambdas:
                    for seed in args.tuning_seeds:
                        evaluate(
                            "lambda_continuation",
                            "ridge_dual",
                            hidden_dim,
                            args.tuning_ratio,
                            normalized_lambda,
                            seed,
                        )

        selected_lambdas: dict[int, float] = {}
        if args.run_heat_lambda_continuation:
            for hidden_dim in args.hidden_dims:
                candidates: dict[float, float] = {}
                for normalized_lambda in args.normalized_lambdas:
                    values = [
                        float(record["test_rmse"])
                        for record in records
                        if record["stage"] == "lambda_continuation"
                        and record["hidden_dim"] == hidden_dim
                        and record["normalized_lambda"] == normalized_lambda
                    ]
                    candidates[normalized_lambda] = mean(values)
                selected_lambdas[hidden_dim] = min(
                    candidates, key=lambda value: (candidates[value], value)
                )

        if args.run_completed_heat_ratio_scan:
            if not selected_lambdas:
                raise ValueError("the heat ratio scan requires the lambda continuation")
            for hidden_dim in args.hidden_dims:
                normalized_lambda = selected_lambdas[hidden_dim]
                for ratio in args.sample_ratios:
                    for seed in args.seeds:
                        evaluate(
                            "ratio_scan",
                            "ridge_dual",
                            hidden_dim,
                            ratio,
                            normalized_lambda,
                            seed,
                        )

        # Pair every evaluated training size with an alpha=0 baseline.
        evaluated_settings = {
            (int(record["hidden_dim"]), float(record["sample_ratio"]))
            for record in records
            if record["stage"] == "ratio_scan"
        }
        for hidden_dim, ratio in sorted(evaluated_settings):
            for seed in args.seeds:
                evaluate("baseline", "constant", hidden_dim, ratio, 0.0, seed)

    baseline = {
        (int(record["sample_size"]), str(record["seed"])): float(record["test_rmse"])
        for record in records
        if record["stage"] == "baseline"
    }
    for record in records:
        base_rmse = baseline.get((int(record["sample_size"]), str(record["seed"])))
        if record["method"] == "ridge_dual" and base_rmse is not None:
            ratio = float(record["test_rmse"]) / base_rmse
            record["test_rmse_ratio"] = ratio
            record["test_mse_reduction"] = 1.0 - ratio**2
        else:
            record["test_rmse_ratio"] = 1.0
            record["test_mse_reduction"] = 0.0

    fields = [
        "stage", "method", "hidden_dim", "parameter_count", "sample_ratio",
        "sample_size", "normalized_lambda", "raw_lambda", "seed", "y0",
        "y0_abs_error", "fit_rmse", "test_rmse", "test_rmse_ratio",
        "test_mse_reduction", "time_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    groups: dict[tuple[str, str, int, float, float], list[dict]] = defaultdict(list)
    for record in records:
        groups[(
            str(record["stage"]),
            str(record["method"]),
            int(record["hidden_dim"]),
            float(record["sample_ratio"]),
            float(record["normalized_lambda"]),
        )].append(record)

    with summary_path.open("w", encoding="utf-8") as out:
        out.write("AsymmetricHeat fixed-S-over-p RFM experiment\n")
        out.write(f"timestamp={timestamp}\nsolver={solver}\nbase_config={base_path}\n")
        out.write(f"dimension={dimension}\nexact_y0={exact:.10g}\n")
        out.write(f"tuning_seeds={','.join(args.tuning_seeds)}\n")
        out.write(f"evaluation_seeds={','.join(args.seeds)}\n")
        out.write(f"test_sample_size={base['solver_config']['test_sample_size']}\n")
        out.write(f"test_batch_size={base['solver_config']['test_batch_size']}\n")
        out.write("lambda_convention=raw_lambda=sample_size*normalized_lambda\n")
        out.write("selection_rule=minimum mean tuning-seed test RMSE for each H\n")
        out.write(
            "selected_normalized_lambdas="
            + ",".join(f"H{hidden_dim}:{value:g}" for hidden_dim, value in selected_lambdas.items())
            + "\n\n"
        )
        out.write(
            "stage,method,H,p,S_over_p,S,normalized_lambda,raw_lambda,repeats,"
            "y0_abs_error_mean,y0_abs_error_ci95,fit_rmse_mean,test_rmse_mean,"
            "test_rmse_ci95,test_rmse_ratio_mean,test_mse_reduction_mean,time_ms_mean\n"
        )
        for key, group in sorted(groups.items()):
            stage, method, hidden_dim, ratio, normalized_lambda = key
            first = group[0]
            y0_errors = [float(record["y0_abs_error"]) for record in group]
            fit = [float(record["fit_rmse"]) for record in group]
            test = [float(record["test_rmse"]) for record in group]
            rmse_ratios = [float(record["test_rmse_ratio"]) for record in group]
            reductions = [float(record["test_mse_reduction"]) for record in group]
            times = [float(record["time_ms"]) for record in group]
            out.write(
                f"{stage},{method},{hidden_dim},{first['parameter_count']},{ratio:.8g},"
                f"{first['sample_size']},{normalized_lambda:.8g},{first['raw_lambda']:.8g},"
                f"{len(group)},{mean(y0_errors):.8g},{ci95(y0_errors):.4g},"
                f"{mean(fit):.8g},{mean(test):.8g},{ci95(test):.4g},"
                f"{mean(rmse_ratios):.8g},{mean(reductions):.8g},{mean(times):.8g}\n"
            )

    print("\nselected normalized lambdas:")
    for hidden_dim, value in selected_lambdas.items():
        print(f"  H={hidden_dim}: {value:g}")
    print(f"raw runs: {csv_path}")
    print(f"summary: {summary_path}")
    if not args.skip_nonlinear_variants:
        nonlinear_csv, nonlinear_summary = run_nonlinear_variant_experiments(
            args, solver, result_dir, timestamp
        )
        print(f"nonlinear raw runs: {nonlinear_csv}")
        print(f"nonlinear summary: {nonlinear_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
