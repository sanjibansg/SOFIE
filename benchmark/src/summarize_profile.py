#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BACKENDS = ("sofie", "ort", "tensorrt", "pytorch_aot")

METRICS = {
    "duration_us": ("gpu__time_duration.sum", "gpu__time_duration.avg"),
    "registers_per_thread": ("launch__registers_per_thread",),
    "occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active", "sm__maximum_warps_per_active_cycle_pct"),
    "shared_mem_kb": ("launch__shared_mem_per_block",),
    "local_spill_requests": ("derived__local_spilling_requests", "sass__inst_executed_register_spilling_mem_local"),
    "shared_spill_requests": ("derived__shared_spilling_requests", "sass__inst_executed_register_spilling_mem_shared"),
    "dram_throughput_pct": ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",),
    "dram_bandwidth_gbs": ("dram__bytes.avg.per_second", "dram__bytes.sum.per_second"),
    "l1_hit_rate_pct": ("l1tex__t_sector_hit_rate.pct",),
    "l2_hit_rate_pct": ("lts__t_sector_hit_rate.pct",),
    "tensor_pipe_pct": ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",),
    "memory_throughput_pct": ("sm__memory_throughput.avg.pct_of_peak_sustained_elapsed",),
    "sm_throughput_pct": ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
    "grid_size": ("launch__grid_size",),
    "block_size": ("launch__block_size",),
}

KERNEL_NAME_COLUMNS = ("Kernel Name", "launch__kernel_name")


@dataclass
class KernelRecord:
    kernel_name: str
    duration_us: Optional[float]
    registers_per_thread: Optional[float]
    occupancy_pct: Optional[float]
    shared_mem_kb: Optional[float]
    local_spill_requests: Optional[float]
    shared_spill_requests: Optional[float]
    dram_throughput_pct: Optional[float]
    dram_bandwidth_gbs: Optional[float]
    l1_hit_rate_pct: Optional[float]
    l2_hit_rate_pct: Optional[float]
    tensor_pipe_pct: Optional[float]
    memory_throughput_pct: Optional[float]
    sm_throughput_pct: Optional[float]
    grid_size: Optional[float]
    block_size: Optional[float]


@dataclass
class ModelSummary:
    model: str
    csv_file: str
    kernels: int
    unique_kernels: int
    total_gpu_time_us: Optional[float]
    average_kernel_time_us: Optional[float]
    maximum_kernel_time_us: Optional[float]
    average_registers_per_thread: Optional[float]
    maximum_registers_per_thread: Optional[float]
    average_occupancy_pct: Optional[float]
    minimum_occupancy_pct: Optional[float]
    average_shared_mem_kb: Optional[float]
    maximum_shared_mem_kb: Optional[float]
    average_dram_throughput_pct: Optional[float]
    average_dram_bandwidth_gbs: Optional[float]
    average_l1_hit_rate_pct: Optional[float]
    average_l2_hit_rate_pct: Optional[float]
    average_tensor_pipe_pct: Optional[float]
    spilling_kernels: int
    shared_spilling_kernels: int
    shared_memory_kernels: int
    tensor_core_kernels: int
    likely_memory_bound_kernels: int
    likely_compute_bound_kernels: int
    top_slowest_kernels: List[Dict[str, Any]]
    top_register_kernels: List[Dict[str, Any]]
    lowest_occupancy_kernels: List[Dict[str, Any]]
    conclusions: List[str]


@dataclass
class BackendSummary:
    backend: str
    models: int
    kernels: int
    unique_kernels: int
    total_gpu_time_us: Optional[float]
    average_kernel_time_us: Optional[float]
    average_registers_per_thread: Optional[float]
    maximum_registers_per_thread: Optional[float]
    average_occupancy_pct: Optional[float]
    average_shared_mem_kb: Optional[float]
    average_dram_throughput_pct: Optional[float]
    average_dram_bandwidth_gbs: Optional[float]
    average_l1_hit_rate_pct: Optional[float]
    average_l2_hit_rate_pct: Optional[float]
    average_tensor_pipe_pct: Optional[float]
    spilling_kernels: int
    shared_spilling_kernels: int
    shared_memory_kernels: int
    tensor_core_kernels: int
    likely_memory_bound_kernels: int
    likely_compute_bound_kernels: int
    model_summaries: List[ModelSummary]
    conclusions: List[str]


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "nan", "inf", "-inf"}:
        return None
    text = text.replace(",", "").replace("%", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def first_metric(row: Dict[str, str], candidates: Iterable[str]) -> Optional[float]:
    for name in candidates:
        if name in row:
            value = parse_number(row.get(name))
            if value is not None:
                return value
    return None


def first_text(row: Dict[str, str], candidates: Iterable[str]) -> str:
    for name in candidates:
        value = row.get(name)
        if value and value.strip():
            return value.strip()
    return "<unknown kernel>"


def read_ncu_csv(path: Path) -> List[KernelRecord]:
    records: List[KernelRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("missing CSV header")
        for row_index, row in enumerate(reader):
            if row_index == 0 and not any(str(row.get(name, "")).strip() for name in KERNEL_NAME_COLUMNS):
                continue
            kernel_name = first_text(row, KERNEL_NAME_COLUMNS)
            if kernel_name == "<unknown kernel>":
                continue
            values = {key: first_metric(row, candidates) for key, candidates in METRICS.items()}
            records.append(KernelRecord(kernel_name=kernel_name, **values))
    return records


def values(records: Iterable[KernelRecord], field: str) -> List[float]:
    result = []
    for record in records:
        value = getattr(record, field)
        if value is not None:
            result.append(value)
    return result


def mean_or_none(items: Iterable[float]) -> Optional[float]:
    data = list(items)
    return statistics.fmean(data) if data else None


def max_or_none(items: Iterable[float]) -> Optional[float]:
    data = list(items)
    return max(data) if data else None


def min_or_none(items: Iterable[float]) -> Optional[float]:
    data = list(items)
    return min(data) if data else None


def weighted_mean(records: Iterable[KernelRecord], field: str) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    fallback = []
    for record in records:
        value = getattr(record, field)
        if value is None:
            continue
        fallback.append(value)
        if record.duration_us is not None and record.duration_us > 0:
            numerator += value * record.duration_us
            denominator += record.duration_us
    if denominator > 0:
        return numerator / denominator
    return mean_or_none(fallback)


def kernel_short_name(name: str, limit: int = 130) -> str:
    compact = " ".join(name.split())
    return compact if len(compact) <= limit else compact[:limit - 3] + "..."


def top_records(records: List[KernelRecord], field: str, count: int, reverse: bool = True) -> List[Dict[str, Any]]:
    filtered = [record for record in records if getattr(record, field) is not None]
    filtered.sort(key=lambda record: getattr(record, field), reverse=reverse)
    return [{"kernel": kernel_short_name(record.kernel_name), "value": getattr(record, field)} for record in filtered[:count]]


def detect_tensor_core(record: KernelRecord) -> bool:
    if record.tensor_pipe_pct is not None and record.tensor_pipe_pct > 0.05:
        return True
    name = record.kernel_name.lower()
    return any(token in name for token in ("tensorop", "wmma", "mma_", "hmma", "imma"))


def likely_memory_bound(record: KernelRecord) -> bool:
    dram = record.dram_throughput_pct
    memory = record.memory_throughput_pct
    sm = record.sm_throughput_pct
    return (dram is not None and dram >= 60.0) or (memory is not None and sm is not None and memory >= 60.0 and memory > sm * 1.2)


def likely_compute_bound(record: KernelRecord) -> bool:
    memory = record.memory_throughput_pct
    sm = record.sm_throughput_pct
    return memory is not None and sm is not None and sm >= 60.0 and sm > memory * 1.2


def model_conclusions(summary: ModelSummary) -> List[str]:
    conclusions = []
    if summary.spilling_kernels == 0:
        conclusions.append("No local-memory register spilling was detected.")
    else:
        conclusions.append(f"Register spilling was detected in {summary.spilling_kernels} kernel launch(es).")
    if summary.average_occupancy_pct is not None:
        if summary.average_occupancy_pct < 35:
            conclusions.append("Average achieved occupancy is low; inspect register, shared-memory, and block-size limits.")
        elif summary.average_occupancy_pct >= 70:
            conclusions.append("Average achieved occupancy is high.")
        else:
            conclusions.append("Average achieved occupancy is moderate.")
    if summary.average_l2_hit_rate_pct is not None:
        if summary.average_l2_hit_rate_pct >= 80:
            conclusions.append("L2 cache locality is strong.")
        elif summary.average_l2_hit_rate_pct < 40:
            conclusions.append("L2 hit rate is low, so a larger fraction of traffic reaches device memory.")
    if summary.likely_memory_bound_kernels:
        conclusions.append(f"{summary.likely_memory_bound_kernels} kernel launch(es) appear memory-throughput limited.")
    if summary.likely_compute_bound_kernels:
        conclusions.append(f"{summary.likely_compute_bound_kernels} kernel launch(es) appear compute-throughput limited.")
    if summary.tensor_core_kernels:
        conclusions.append(f"Tensor Core activity was detected in {summary.tensor_core_kernels} kernel launch(es).")
    if summary.shared_memory_kernels == 0:
        conclusions.append("No explicit per-block shared-memory allocation was reported.")
    return conclusions


def summarize_model(path: Path, records: List[KernelRecord]) -> ModelSummary:
    durations = values(records, "duration_us")
    registers = values(records, "registers_per_thread")
    occupancy = values(records, "occupancy_pct")
    shared = values(records, "shared_mem_kb")
    spilling = sum(1 for record in records if (record.local_spill_requests or 0) > 0)
    shared_spilling = sum(1 for record in records if (record.shared_spill_requests or 0) > 0)
    shared_memory = sum(1 for record in records if (record.shared_mem_kb or 0) > 0)
    tensor_cores = sum(1 for record in records if detect_tensor_core(record))
    memory_bound = sum(1 for record in records if likely_memory_bound(record))
    compute_bound = sum(1 for record in records if likely_compute_bound(record))
    summary = ModelSummary(
        model=path.stem,
        csv_file=str(path),
        kernels=len(records),
        unique_kernels=len({record.kernel_name for record in records}),
        total_gpu_time_us=sum(durations) if durations else None,
        average_kernel_time_us=mean_or_none(durations),
        maximum_kernel_time_us=max_or_none(durations),
        average_registers_per_thread=weighted_mean(records, "registers_per_thread"),
        maximum_registers_per_thread=max_or_none(registers),
        average_occupancy_pct=weighted_mean(records, "occupancy_pct"),
        minimum_occupancy_pct=min_or_none(occupancy),
        average_shared_mem_kb=weighted_mean(records, "shared_mem_kb"),
        maximum_shared_mem_kb=max_or_none(shared),
        average_dram_throughput_pct=weighted_mean(records, "dram_throughput_pct"),
        average_dram_bandwidth_gbs=weighted_mean(records, "dram_bandwidth_gbs"),
        average_l1_hit_rate_pct=weighted_mean(records, "l1_hit_rate_pct"),
        average_l2_hit_rate_pct=weighted_mean(records, "l2_hit_rate_pct"),
        average_tensor_pipe_pct=weighted_mean(records, "tensor_pipe_pct"),
        spilling_kernels=spilling,
        shared_spilling_kernels=shared_spilling,
        shared_memory_kernels=shared_memory,
        tensor_core_kernels=tensor_cores,
        likely_memory_bound_kernels=memory_bound,
        likely_compute_bound_kernels=compute_bound,
        top_slowest_kernels=top_records(records, "duration_us", 10),
        top_register_kernels=top_records(records, "registers_per_thread", 10),
        lowest_occupancy_kernels=top_records(records, "occupancy_pct", 10, reverse=False),
        conclusions=[],
    )
    summary.conclusions = model_conclusions(summary)
    return summary


def backend_conclusions(summary: BackendSummary) -> List[str]:
    conclusions = []
    if summary.spilling_kernels == 0:
        conclusions.append("No local-memory register spilling was detected in this backend.")
    else:
        conclusions.append(f"Register spilling was detected in {summary.spilling_kernels} kernel launch(es).")
    if summary.average_occupancy_pct is not None:
        if summary.average_occupancy_pct < 35:
            conclusions.append("Time-weighted occupancy is low.")
        elif summary.average_occupancy_pct >= 70:
            conclusions.append("Time-weighted occupancy is high.")
        else:
            conclusions.append("Time-weighted occupancy is moderate.")
    if summary.tensor_core_kernels:
        conclusions.append(f"Tensor Core activity was detected in {summary.tensor_core_kernels} kernel launch(es).")
    if summary.likely_memory_bound_kernels:
        conclusions.append(f"{summary.likely_memory_bound_kernels} kernel launch(es) appear memory-throughput limited.")
    if summary.likely_compute_bound_kernels:
        conclusions.append(f"{summary.likely_compute_bound_kernels} kernel launch(es) appear compute-throughput limited.")
    return conclusions


def summarize_backend(backend: str, model_data: List[Tuple[ModelSummary, List[KernelRecord]]]) -> BackendSummary:
    models = [item[0] for item in model_data]
    records = [record for _, model_records in model_data for record in model_records]
    durations = values(records, "duration_us")
    registers = values(records, "registers_per_thread")
    summary = BackendSummary(
        backend=backend,
        models=len(models),
        kernels=len(records),
        unique_kernels=len({record.kernel_name for record in records}),
        total_gpu_time_us=sum(durations) if durations else None,
        average_kernel_time_us=mean_or_none(durations),
        average_registers_per_thread=weighted_mean(records, "registers_per_thread"),
        maximum_registers_per_thread=max_or_none(registers),
        average_occupancy_pct=weighted_mean(records, "occupancy_pct"),
        average_shared_mem_kb=weighted_mean(records, "shared_mem_kb"),
        average_dram_throughput_pct=weighted_mean(records, "dram_throughput_pct"),
        average_dram_bandwidth_gbs=weighted_mean(records, "dram_bandwidth_gbs"),
        average_l1_hit_rate_pct=weighted_mean(records, "l1_hit_rate_pct"),
        average_l2_hit_rate_pct=weighted_mean(records, "l2_hit_rate_pct"),
        average_tensor_pipe_pct=weighted_mean(records, "tensor_pipe_pct"),
        spilling_kernels=sum(model.spilling_kernels for model in models),
        shared_spilling_kernels=sum(model.shared_spilling_kernels for model in models),
        shared_memory_kernels=sum(model.shared_memory_kernels for model in models),
        tensor_core_kernels=sum(model.tensor_core_kernels for model in models),
        likely_memory_bound_kernels=sum(model.likely_memory_bound_kernels for model in models),
        likely_compute_bound_kernels=sum(model.likely_compute_bound_kernels for model in models),
        model_summaries=models,
        conclusions=[],
    )
    summary.conclusions = backend_conclusions(summary)
    return summary


def fmt(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.{digits}f}{suffix}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(root: Path, summaries: List[BackendSummary], warnings: List[str]) -> Path:
    output = root / "profile_summary.md"
    lines = ["# Nsight Compute profile summary", "", f"Source directory: `{root}`", ""]
    if warnings:
        lines.extend(["## Parse warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    if summaries:
        rows = []
        for summary in summaries:
            rows.append([
                summary.backend,
                str(summary.models),
                str(summary.kernels),
                fmt(summary.total_gpu_time_us, 2),
                fmt(summary.average_registers_per_thread, 2),
                fmt(summary.average_occupancy_pct, 2, "%"),
                fmt(summary.average_l2_hit_rate_pct, 2, "%"),
                str(summary.spilling_kernels),
                str(summary.tensor_core_kernels),
            ])
        lines.extend([
            "## Backend comparison",
            "",
            markdown_table(
                ["Backend", "Models", "Kernel launches", "Total GPU time (us)", "Registers/thread", "Occupancy", "L2 hit rate", "Spilling launches", "Tensor Core launches"],
                rows,
            ),
            "",
        ])
    for summary in summaries:
        lines.extend([
            f"## {summary.backend.upper()}",
            "",
            markdown_table(
                ["Metric", "Value"],
                [
                    ["Models", str(summary.models)],
                    ["Kernel launches", str(summary.kernels)],
                    ["Unique kernel names", str(summary.unique_kernels)],
                    ["Total GPU time", fmt(summary.total_gpu_time_us, 2, " us")],
                    ["Average kernel duration", fmt(summary.average_kernel_time_us, 3, " us")],
                    ["Time-weighted registers/thread", fmt(summary.average_registers_per_thread)],
                    ["Maximum registers/thread", fmt(summary.maximum_registers_per_thread)],
                    ["Time-weighted occupancy", fmt(summary.average_occupancy_pct, 2, "%")],
                    ["Time-weighted shared memory/block", fmt(summary.average_shared_mem_kb, 3, " KB")],
                    ["Time-weighted DRAM throughput", fmt(summary.average_dram_throughput_pct, 2, "%")],
                    ["Time-weighted DRAM bandwidth", fmt(summary.average_dram_bandwidth_gbs, 2, " GB/s")],
                    ["Time-weighted L1 hit rate", fmt(summary.average_l1_hit_rate_pct, 2, "%")],
                    ["Time-weighted L2 hit rate", fmt(summary.average_l2_hit_rate_pct, 2, "%")],
                    ["Local spilling launches", str(summary.spilling_kernels)],
                    ["Shared spilling launches", str(summary.shared_spilling_kernels)],
                    ["Shared-memory launches", str(summary.shared_memory_kernels)],
                    ["Tensor Core launches", str(summary.tensor_core_kernels)],
                    ["Likely memory-bound launches", str(summary.likely_memory_bound_kernels)],
                    ["Likely compute-bound launches", str(summary.likely_compute_bound_kernels)],
                ],
            ),
            "",
            "### Interpretation",
            "",
        ])
        lines.extend(f"- {item}" for item in summary.conclusions)
        lines.append("")
        for model in summary.model_summaries:
            lines.extend([
                f"### {model.model}",
                "",
                markdown_table(
                    ["Metric", "Value"],
                    [
                        ["Kernel launches", str(model.kernels)],
                        ["Unique kernel names", str(model.unique_kernels)],
                        ["Total GPU time", fmt(model.total_gpu_time_us, 2, " us")],
                        ["Average kernel duration", fmt(model.average_kernel_time_us, 3, " us")],
                        ["Maximum kernel duration", fmt(model.maximum_kernel_time_us, 3, " us")],
                        ["Time-weighted registers/thread", fmt(model.average_registers_per_thread)],
                        ["Maximum registers/thread", fmt(model.maximum_registers_per_thread)],
                        ["Time-weighted occupancy", fmt(model.average_occupancy_pct, 2, "%")],
                        ["Minimum occupancy", fmt(model.minimum_occupancy_pct, 2, "%")],
                        ["Time-weighted shared memory/block", fmt(model.average_shared_mem_kb, 3, " KB")],
                        ["Time-weighted DRAM throughput", fmt(model.average_dram_throughput_pct, 2, "%")],
                        ["Time-weighted L1 hit rate", fmt(model.average_l1_hit_rate_pct, 2, "%")],
                        ["Time-weighted L2 hit rate", fmt(model.average_l2_hit_rate_pct, 2, "%")],
                        ["Spilling launches", str(model.spilling_kernels)],
                        ["Tensor Core launches", str(model.tensor_core_kernels)],
                    ],
                ),
                "",
                "#### Interpretation",
                "",
            ])
            lines.extend(f"- {item}" for item in model.conclusions)
            lines.append("")
            for title, entries, unit in (
                    ("Slowest kernel launches", model.top_slowest_kernels, "us"),
                    ("Highest register usage", model.top_register_kernels, "registers/thread"),
                    ("Lowest occupancy", model.lowest_occupancy_kernels, "%"),
            ):
                lines.extend([f"#### {title}", ""])
                if entries:
                    lines.append(markdown_table(["Kernel", "Value"], [[entry["kernel"].replace("|", "\\|"), fmt(entry["value"], 3, f" {unit}")] for entry in entries]))
                else:
                    lines.append("No data available.")
                lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_json(root: Path, summaries: List[BackendSummary], warnings: List[str]) -> Path:
    output = root / "profile_summary.json"
    payload = {"root": str(root), "warnings": warnings, "backends": [asdict(summary) for summary in summaries]}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def find_profile_csvs(backend_dir: Path) -> List[Path]:
    result = []
    for path in sorted(backend_dir.glob("*.csv")):
        lower = path.name.lower()
        if lower == "benchmark.csv" or lower.startswith("profile_summary"):
            continue
        result.append(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Nsight Compute CSV reports for SOFIE, ONNX Runtime, TensorRT, and PyTorch AOTInductor.")
    parser.add_argument("root", type=Path, help="Benchmark run directory containing sofie/, ort/, tensorrt/, and/or pytorch_aot/.")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: directory does not exist: {root}", file=sys.stderr)
        return 2

    summaries: List[BackendSummary] = []
    warnings: List[str] = []

    for backend in BACKENDS:
        backend_dir = root / backend
        if not backend_dir.is_dir():
            continue
        csv_files = find_profile_csvs(backend_dir)
        if not csv_files:
            warnings.append(f"{backend}: no Nsight Compute CSV files found.")
            continue
        model_data: List[Tuple[ModelSummary, List[KernelRecord]]] = []
        for path in csv_files:
            try:
                records = read_ncu_csv(path)
            except Exception as error:
                warnings.append(f"{backend}/{path.name}: {error}")
                continue
            if not records:
                warnings.append(f"{backend}/{path.name}: no kernel rows found.")
                continue
            model_data.append((summarize_model(path, records), records))
        if model_data:
            summaries.append(summarize_backend(backend, model_data))

    if not summaries:
        print("error: no usable Nsight Compute CSV files were found.", file=sys.stderr)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        return 1

    markdown_path = write_markdown(root, summaries, warnings)
    json_path = write_json(root, summaries, warnings)

    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())