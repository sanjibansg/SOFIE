#!/usr/bin/env python3
"""
Create comparison plots for benchmark CSVs.

Each benchmark directory may contain backend subdirectories such as:

benchmark_YYYYMMDD_HHMMSS/
├── sofie/benchmark.csv
├── ort/benchmark.csv
└── tensorrt/benchmark.csv

Usage:
    python3 plot_cms_results.py /path/to/run1
    python3 plot_cms_results.py /path/to/run1 /path/to/run2 /path/to/run3

Individual benchmark.csv files may also be passed if their parent directory is
one of the backend folders configured in BACKENDS.

When multiple runs are provided, results are averaged per model/backend and
standard deviations are shown as error bars. Output is always written to the
plots/ directory of the first input run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BACKENDS = {
    "sofie": {"label": "SOFIE", "short": "SOFIE"},
    "ort": {"label": "ONNX Runtime", "short": "ORT"},
    "tensorrt": {"label": "TensorRT", "short": "TensorRT"},
    "pytorch_aot": {"label": "Pytorch AOT", "short": "AOT"},
}

REFERENCE_BACKEND = "ONNX Runtime"

MODEL_FAMILIES = [
    {
        "name": "GNN",
        "pattern": r"(?P<architecture>gnn_h\d+_k\d+)_n(?P<scale>\d+)_e\d+",
        "x_label": "Number of nodes",
        "log_x": True,
        "latency_log_y": True,
        "memory_log_y": True,
    },
    {
        "name": "GNN Large",
        "pattern": r"(?P<architecture>gnn_large)_n(?P<scale>\d+)_e\d+",
        "x_label": "Number of nodes",
        "log_x": True,
        "latency_log_y": True,
        "memory_log_y": True,
    },
    {
        "name": "PUNet",
        "pattern": r"(?P<architecture>punet_h\d+_k\d+_heads\d+_layers\d+)_n(?P<scale>\d+)_e\d+",
        "x_label": "Number of nodes",
        "log_x": True,
        "latency_log_y": True,
        "memory_log_y": True,
    },
    {
        "name": "Transformer",
        "pattern": r"(?P<architecture>transformer_.+)_n\d+_s(?P<scale>\d+)",
        "x_label": "Sequence length",
        "log_x": False,
        "latency_log_y": False,
        "memory_log_y": False,
    },
    {
        "name": "MLPF",
        "pattern": r"(?P<architecture>mlpf_fp32_(?:unfused|fused)_b1)_n(?P<scale>\d+)",
        "x_label": "Input size (n)",
        "log_x": True,
        "latency_log_y": True,
        "memory_log_y": True,
    },
]


def backend_labels() -> list[str]:
    return [backend["label"] for backend in BACKENDS.values()]


def backend_short_name(label: str) -> str:
    for backend in BACKENDS.values():
        if backend["label"] == label:
            return backend["short"]
    return re.sub(r"\W+", "_", label).strip("_")


def load_benchmark_csv(path: Path, backend: str, run: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Model", "infer_ms", "gpu_peak_mem_mb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[["Model", "infer_ms", "gpu_peak_mem_mb"]].copy()
    df["backend"] = backend
    df["run"] = run
    return df


def load_results(inputs: list[Path]) -> pd.DataFrame:
    frames = []

    for input_path in inputs:
        if input_path.is_dir():
            found = False
            for folder, backend in BACKENDS.items():
                path = input_path / folder / "benchmark.csv"
                if not path.exists():
                    continue
                frames.append(load_benchmark_csv(path, backend["label"], input_path.name))
                found = True

            if not found:
                raise FileNotFoundError(f"No configured backend benchmark files found in: {input_path}")
            continue

        if input_path.is_file():
            folder = input_path.parent.name
            if folder not in BACKENDS:
                raise ValueError(
                    f"Cannot infer backend for {input_path}. "
                    f"The parent directory must be one of: {', '.join(BACKENDS)}."
                )
            frames.append(load_benchmark_csv(input_path, BACKENDS[folder]["label"], input_path.parent.parent.name))
            continue

        raise FileNotFoundError(f"Input does not exist: {input_path}")

    raw = pd.concat(frames, ignore_index=True)

    aggregated = raw.groupby(["Model", "backend"], as_index=False).agg(
        infer_ms=("infer_ms", "mean"),
        infer_ms_std=("infer_ms", "std"),
        gpu_peak_mem_mb=("gpu_peak_mem_mb", "mean"),
        gpu_peak_mem_mb_std=("gpu_peak_mem_mb", "std"),
        n_runs=("infer_ms", "count"),
    )

    aggregated["infer_ms_std"] = aggregated["infer_ms_std"].fillna(0.0)
    aggregated["gpu_peak_mem_mb_std"] = aggregated["gpu_peak_mem_mb_std"].fillna(0.0)
    aggregated["throughput_inf_s"] = 1000.0 / aggregated["infer_ms"]

    return aggregated


def parse_model(model: str) -> dict:
    for family in MODEL_FAMILIES:
        match = re.fullmatch(family["pattern"], model)
        if match:
            return {
                "family": family["name"],
                "architecture": match.group("architecture"),
                "scale": int(match.group("scale")),
            }

    return {
        "family": None,
        "architecture": None,
        "scale": None,
    }


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.DataFrame([parse_model(name) for name in df["Model"]])
    return pd.concat([df.reset_index(drop=True), parsed], axis=1)


def first_input_root(path: Path) -> Path:
    if path.is_dir():
        return path

    if path.is_file() and path.parent.name in BACKENDS:
        return path.parent.parent

    return path.parent


def save_figure(fig, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_scaling(
        df: pd.DataFrame,
        architecture: str,
        metric: str,
        ylabel: str,
        out_dir: Path,
        filename: str,
        x_label: str,
        log_x: bool,
        log_y: bool,
) -> None:
    sub = df[df["architecture"] == architecture].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    std_col = f"{metric}_std"

    for backend in backend_labels():
        part = sub[sub["backend"] == backend].sort_values("scale")
        if part.empty:
            continue

        ax.errorbar(
            part["scale"],
            part[metric],
            yerr=part[std_col],
            marker="o",
            linewidth=2,
            capsize=3,
            label=backend,
        )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(architecture)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, out_dir, filename)


def plot_family_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    for family in MODEL_FAMILIES:
        family_df = df[df["family"] == family["name"]]
        architectures = sorted(family_df["architecture"].dropna().unique())

        for architecture in architectures:
            plot_metric_scaling(
                df,
                architecture=architecture,
                metric="infer_ms",
                ylabel="Inference time (ms)",
                out_dir=out_dir,
                filename=f"{architecture}_latency",
                x_label=family["x_label"],
                log_x=family["log_x"],
                log_y=family["latency_log_y"],
            )

            plot_metric_scaling(
                df,
                architecture=architecture,
                metric="gpu_peak_mem_mb",
                ylabel="Peak GPU memory (MB)",
                out_dir=out_dir,
                filename=f"{architecture}_memory",
                x_label=family["x_label"],
                log_x=family["log_x"],
                log_y=family["memory_log_y"],
            )


def plot_speedup_vs_reference(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot(index="Model", columns="backend", values="infer_ms")
    if REFERENCE_BACKEND not in pivot.columns:
        return

    comparisons = {}
    for backend in backend_labels():
        if backend == REFERENCE_BACKEND or backend not in pivot.columns:
            continue
        comparisons[backend] = pivot[REFERENCE_BACKEND] / pivot[backend]

    if not comparisons:
        return

    speedups = pd.DataFrame(comparisons)

    sort_backend = "SOFIE" if "SOFIE" in speedups.columns else speedups.columns[0]
    speedups = speedups.sort_values(sort_backend)

    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.31 * len(speedups))))
    speedups.plot(kind="barh", ax=ax, width=0.8)

    ax.set_xlabel(f"Speedup over {REFERENCE_BACKEND} (×)")
    ax.set_ylabel("")
    ax.set_title("Inference speedup by model")
    ax.axvline(1.0, linewidth=1)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(title="")
    save_figure(fig, out_dir, "speedup_vs_ort")


def plot_peak_memory_by_model(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot(index="Model", columns="backend", values="gpu_peak_mem_mb")
    available = [backend for backend in backend_labels() if backend in pivot.columns]
    if not available:
        return

    pivot = pivot[available]

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.30 * len(pivot))))
    pivot.plot(kind="barh", ax=ax, width=0.8)

    ax.set_xlabel("Peak GPU memory (MB)")
    ax.set_ylabel("")
    ax.set_title("Peak GPU memory by model")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(title="")
    save_figure(fig, out_dir, "peak_gpu_memory_all_models")


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    pivot_latency = df.pivot(index="Model", columns="backend", values="infer_ms")
    pivot_latency_std = df.pivot(index="Model", columns="backend", values="infer_ms_std")
    pivot_memory = df.pivot(index="Model", columns="backend", values="gpu_peak_mem_mb")
    pivot_runs = df.pivot(index="Model", columns="backend", values="n_runs")

    def value(pivot: pd.DataFrame, model: str, backend: str) -> float:
        if backend not in pivot.columns or model not in pivot.index:
            return float("nan")
        return pivot.loc[model, backend]

    rows = []
    available = [backend for backend in backend_labels() if backend in pivot_latency.columns]

    for model in pivot_latency.index:
        row = {"Model": model}

        for backend in available:
            key = backend_short_name(backend)
            row[f"{key}_ms"] = value(pivot_latency, model, backend)
            row[f"{key}_ms_std"] = value(pivot_latency_std, model, backend)
            row[f"{key}_peak_MB"] = value(pivot_memory, model, backend)
            row[f"{key}_runs"] = value(pivot_runs, model, backend)

        if REFERENCE_BACKEND in available:
            reference_ms = value(pivot_latency, model, REFERENCE_BACKEND)
            reference_key = backend_short_name(REFERENCE_BACKEND)
            for backend in available:
                if backend == REFERENCE_BACKEND:
                    continue
                key = backend_short_name(backend)
                backend_ms = value(pivot_latency, model, backend)
                row[f"{key}_speedup_vs_{reference_key}"] = reference_ms / backend_ms

        rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "comparison_summary.csv", index=False)


def main() -> None:
    inputs = [Path(arg).resolve() for arg in sys.argv[1:]] if len(sys.argv) > 1 else [Path.cwd()]

    out_dir = first_input_root(inputs[0]) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = enrich(load_results(inputs))
    df.to_csv(out_dir / "aggregated_results.csv", index=False)
    write_summary(df, out_dir)

    plot_family_scaling(df, out_dir)
    plot_speedup_vs_reference(df, out_dir)
    plot_peak_memory_by_model(df, out_dir)

    print(f"Plots written to: {out_dir}")
    print("Generated PNG + PDF versions of each figure.")
    print("Also wrote: aggregated_results.csv and comparison_summary.csv")


if __name__ == "__main__":
    main()