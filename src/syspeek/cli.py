"""Command-line interface for SysPeek."""

from __future__ import annotations

from typing import Optional

import click
from rich.console import Console

from syspeek.core.device import detect_device
from syspeek.core.registry import get_benchmarks
from syspeek.core.result import BenchmarkResult, RunContext
from syspeek.platforms import get_platform
from syspeek.reporting import dump_json, print_device_panel, print_results
from syspeek.theoretical import SpecSourceMode

console = Console()

_CATEGORIES = ["compute", "memory", "latency"]


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """SysPeek: benchmark achieved GPU FLOPS, bandwidth, and latency."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


@cli.command(name="info")
@click.option("--device", "device_index", default=0, type=int, help="CUDA device index.")
def info(device_index: int) -> None:
    """Show detected device and platform information."""
    dev = detect_device(device_index)
    platform = get_platform(dev)
    print_device_panel(console, dev, platform.notes())


@cli.command(name="list")
def list_benchmarks() -> None:
    """List available benchmarks."""
    for b in get_benchmarks():
        console.print(f"[cyan]{b.name}[/cyan] ([dim]{b.category}[/dim]) - {b.description}")


@cli.command(name="run")
@click.option("--device", "device_index", default=0, type=int, help="CUDA device index.")
@click.option(
    "--bench",
    "bench_names",
    multiple=True,
    help="Run only these benchmark(s) by name (repeatable).",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    type=click.Choice(_CATEGORIES),
    help="Run only these categories (repeatable).",
)
@click.option(
    "--dtype",
    "dtypes",
    multiple=True,
    help="Restrict compute dtypes (e.g. --dtype fp16 --dtype bf16).",
)
@click.option("--warmup", default=10, type=int, help="Warmup iterations.")
@click.option("--rep", default=50, type=int, help="Timed repetitions.")
@click.option("--gemm-size", default=8192, type=int, help="Square GEMM M=N=K.")
@click.option(
    "--gemm-fast",
    is_flag=True,
    help="FP16/BF16: use Tensor Core fast accumulation (FP16 acc, higher TFLOPS).",
)
@click.option("--transfer-mb", default=256, type=int, help="H2D/D2H total transfer size (MB).")
@click.option(
    "--transfer-streams",
    default=8,
    show_default=True,
    type=int,
    help="Parallel CUDA streams for host<->device copies.",
)
@click.option(
    "--transfer-mode",
    type=click.Choice(["multi_stream", "single", "threaded"], case_sensitive=False),
    default="multi_stream",
    show_default=True,
    help="Copy strategy: multi_stream (default), single buffer, or threaded CPU submit.",
)
@click.option(
    "--device-mem-mb",
    default=1024,
    type=int,
    help="GPU local memory (VRAM/LPDDR) test buffer size (MB).",
)
@click.option("--no-flush-l2", is_flag=True, help="Disable L2 cache flush between reps.")
@click.option(
    "--spec-source",
    type=click.Choice(["auto", "fixed", "auto-fallback"], case_sensitive=False),
    default="auto-fallback",
    show_default=True,
    help=(
        "Theoretical peak source: auto (derive from device/clocks), "
        "fixed (curated table), auto-fallback (auto then fixed)."
    ),
)
@click.option("-o", "--output", type=click.Path(), help="Write results JSON to file.")
@click.option("--json", "as_json", is_flag=True, help="Print results JSON to stdout.")
def run(
    device_index: int,
    bench_names: tuple[str, ...],
    categories: tuple[str, ...],
    dtypes: tuple[str, ...],
    warmup: int,
    rep: int,
    gemm_size: int,
    gemm_fast: bool,
    transfer_mb: int,
    transfer_streams: int,
    transfer_mode: str,
    device_mem_mb: int,
    no_flush_l2: bool,
    spec_source: str,
    output: Optional[str],
    as_json: bool,
) -> None:
    """Run benchmarks and print a report."""
    dev = detect_device(device_index)
    platform = get_platform(dev)
    torch_device = f"cuda:{device_index}"
    spec_mode = SpecSourceMode(spec_source.replace("-", "_"))

    ctx = RunContext(
        device=dev,
        torch_device=torch_device,
        warmup=warmup,
        rep=rep,
        flush_l2=not no_flush_l2,
        sizes={
            "gemm_size": gemm_size,
            "gemm_fast": gemm_fast,
            "transfer_bytes": transfer_mb * 1024 * 1024,
            "transfer_streams": transfer_streams,
            "transfer_mode": transfer_mode,
            "device_mem_bytes": device_mem_mb * 1024 * 1024,
        },
        dtypes=list(dtypes) if dtypes else None,
        spec_source=spec_mode,
    )

    benchmarks = get_benchmarks(
        names=list(bench_names) if bench_names else None,
        categories=list(categories) if categories else None,
    )
    benchmarks = [b for b in benchmarks if b.applicable(dev)]

    if not as_json:
        print_device_panel(console, dev, platform.notes(), spec_mode)

    results: list[BenchmarkResult] = []
    for b in benchmarks:
        if not as_json:
            console.print(f"[dim]running {b.name}...[/dim]")
        try:
            results.extend(b.run(ctx))
        except Exception as e:  # keep going; record the failure
            results.append(
                BenchmarkResult(
                    name=b.name,
                    category=b.category,
                    value=0.0,
                    unit="",
                    error=f"{type(e).__name__}: {e}",
                )
            )

    if as_json:
        import json as _json

        from syspeek.reporting import build_json

        click.echo(_json.dumps(build_json(dev, results, spec_mode), indent=2))
    else:
        print_results(console, results)

    if output:
        dump_json(output, dev, results, spec_mode)
        if not as_json:
            console.print(f"[green]Results written to {output}[/green]")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
