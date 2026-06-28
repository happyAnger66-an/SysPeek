"""Rendering of benchmark results: rich tables and JSON."""

from __future__ import annotations

import json
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from syspeek.core.device import DeviceInfo
from syspeek.core.result import BenchmarkResult
from syspeek.theoretical import SpecSourceMode

_CATEGORY_TITLE = {
    "compute": "Compute (GEMM throughput)",
    "memory": "Memory bandwidth",
    "latency": "Latency",
}

_SOURCE_LABEL = {
    "auto": "auto",
    "fixed": "fixed",
}


def print_device_panel(
    console: Console,
    device: DeviceInfo,
    notes: list[str],
    spec_source: Optional[SpecSourceMode] = None,
) -> None:
    table = Table(title="Device", show_header=False, title_justify="left")
    info = device.to_dict()
    for key in (
        "name",
        "platform",
        "sm_arch",
        "compute_capability",
        "total_memory_gb",
        "multi_processor_count",
        "l2_cache_size_mb",
        "is_integrated",
    ):
        table.add_row(key, str(info[key]))
    table.add_row("torch", str(device.extra.get("torch_version", "")))
    table.add_row("cuda", str(device.extra.get("cuda_version", "")))
    if spec_source is not None:
        table.add_row("spec_source", spec_source.value)
    if "board_model" in device.extra:
        table.add_row("board_model", str(device.extra["board_model"]))
    console.print(table)
    for note in notes:
        console.print(f"[yellow]note:[/yellow] {note}")


def print_results(console: Console, results: list[BenchmarkResult]) -> None:
    by_cat: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    for cat, items in by_cat.items():
        table = Table(title=_CATEGORY_TITLE.get(cat, cat), title_justify="left")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Value", justify="right", style="bold")
        table.add_column("Unit")
        table.add_column("Theoretical", justify="right")
        table.add_column("Eff.", justify="right")
        table.add_column("Spec src", style="dim")
        table.add_column("Median ms", justify="right")
        table.add_column("Config", style="dim")

        for r in items:
            if r.error:
                table.add_row(
                    r.name,
                    "[red]ERR[/red]",
                    r.unit,
                    "-",
                    "-",
                    "-",
                    "-",
                    f"[red]{r.error}[/red]",
                )
                continue
            eff = r.efficiency
            theo = f"{r.theoretical:.1f}" if r.theoretical else "-"
            eff_s = f"{eff * 100:.1f}%" if eff is not None else "-"
            src = _format_spec_source(r)
            median = f"{r.timing.median_ms:.4f}" if r.timing else "-"
            table.add_row(
                r.name,
                f"{r.value:.2f}",
                r.unit,
                theo,
                eff_s,
                src,
                median,
                _fmt_config(r.config),
            )
        console.print(table)


def _format_spec_source(r: BenchmarkResult) -> str:
    if not r.theoretical:
        return "-"
    label = _SOURCE_LABEL.get(r.theoretical_source or "", r.theoretical_source or "-")
    if r.theoretical_detail:
        # Keep table compact; full detail is in JSON.
        short = r.theoretical_detail
        if len(short) > 36:
            short = short[:33] + "..."
        return f"{label}\n[dim]{short}[/dim]"
    return label


def _fmt_config(config: dict[str, Any]) -> str:
    parts = []
    for k, v in config.items():
        if k == "bytes":
            parts.append(f"{v / (1024 ** 2):.0f}MB")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def build_json(
    device: DeviceInfo,
    results: list[BenchmarkResult],
    spec_source: Optional[SpecSourceMode] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device": device.to_dict(),
        "results": [r.to_dict() for r in results],
    }
    if spec_source is not None:
        payload["spec_source"] = spec_source.value
    return payload


def dump_json(
    path: str,
    device: DeviceInfo,
    results: list[BenchmarkResult],
    spec_source: Optional[SpecSourceMode] = None,
) -> None:
    with open(path, "w") as f:
        json.dump(build_json(device, results, spec_source), f, indent=2)
