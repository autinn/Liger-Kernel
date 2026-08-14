"""Compare isolated CUDA/CuTe, CuTe DSL, and cuTile MLP3 2SM kernels."""

import argparse
import csv
import fcntl
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
BACKEND_PATHS = {
    "cutedsl": EXPERIMENT_ROOT / "backends/cutedsl.py",
    "cutile": EXPERIMENT_ROOT / "backends/cutile.py",
    "triton": EXPERIMENT_ROOT / "backends/triton.py",
}
CUDA_BINARIES = {
    "cuda_s5": EXPERIMENT_ROOT / "build/cuda_mlp3_2sm",
    "cuda_s6": EXPERIMENT_ROOT / "build/cuda_mlp3_2sm_s6",
}
LOCK_PATH = Path("/data/ssd/all-mlp-2sm.lock")

PYTHON_PROVIDERS = tuple(BACKEND_PATHS)
PROVIDERS = ("cuda_s5", "cuda_s6", *PYTHON_PROVIDERS)

TILE_M = 256
TILE_N = 256
TILE_K = 64
L2_EVICTION_BYTES = 256 << 20
DY_SEED = 0x12345678
Z_SEED = 0x9ABCDEF0


@dataclass(frozen=True)
class ModelShape:
    name: str
    tokens: int
    hidden: int
    intermediate: int
    experts: int


MODEL_SHAPES = (
    ModelShape("Qwen3-30B-A3B", 8192, 2048, 768, 8),
    ModelShape("Qwen3-235B-A22B", 8192, 4096, 1536, 8),
    ModelShape("Qwen3.5-122B-A10B", 8192, 3072, 1024, 8),
    ModelShape("Llama-4-Scout-17B-16E", 8192, 5120, 8192, 8),
    ModelShape("Mixtral-8x7B", 8192, 4096, 14336, 8),
    ModelShape("Mixtral-8x22B", 8192, 6144, 16384, 8),
)
MODEL_BY_NAME = {shape.name: shape for shape in MODEL_SHAPES}


def _load_backend(provider):
    if provider not in BACKEND_PATHS:
        raise ValueError(f"{provider} is not a Python backend")
    module_name = f"mlp3_2sm_backend_{provider}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        BACKEND_PATHS[provider],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {provider} backend from {BACKEND_PATHS[provider]}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_hashed_bf16(shape, seed, device, chunk_elements=4 * 1024 * 1024):
    import torch

    count = math.prod(shape)
    output = torch.empty(count, dtype=torch.bfloat16, device=device)
    mask = 0xFFFFFFFF
    for start in range(0, count, chunk_elements):
        end = min(start + chunk_elements, count)
        value = torch.arange(start, end, dtype=torch.int64, device=device) & mask
        value = (value ^ seed) & mask
        value = ((value ^ (value >> 16)) * 0x7FEB352D) & mask
        value = ((value ^ (value >> 15)) * 0x846CA68B) & mask
        value = (value ^ (value >> 16)) & mask
        finite = ((value & 0xFFFF).to(torch.float32) / 32768.0 - 1.0) * 0.5
        output[start:end].copy_(finite)
    return output.view(shape)


def _make_k_ranges(blocks_per_expert, device):
    import torch

    starts = []
    ends = []
    running = 0
    for blocks in blocks_per_expert:
        starts.append(running)
        running += blocks
        ends.append(running)
    return (
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(ends, dtype=torch.int32, device=device),
    )


def _make_balanced_inputs(shape, device):
    total_blocks = shape.tokens // TILE_K
    if total_blocks % shape.experts:
        raise ValueError(f"{shape.name}: K blocks do not divide across experts")
    blocks_per_expert = [total_blocks // shape.experts] * shape.experts
    dy = _make_hashed_bf16((shape.tokens, shape.hidden), DY_SEED, device)
    z = _make_hashed_bf16(
        (shape.tokens, shape.intermediate),
        Z_SEED,
        device,
    )
    starts, ends = _make_k_ranges(blocks_per_expert, device)
    return dy, z, starts, ends


def _reference(dy, z, starts, ends):
    import torch

    outputs = []
    for first_block, last_block in zip(
        starts.cpu().tolist(),
        ends.cpu().tolist(),
        strict=True,
    ):
        first = first_block * TILE_K
        last = last_block * TILE_K
        if first == last:
            outputs.append(
                torch.zeros(
                    (dy.shape[1], z.shape[1]),
                    dtype=torch.float32,
                    device=dy.device,
                )
            )
        else:
            outputs.append(dy[first:last].float().T @ z[first:last].float())
    return torch.stack(outputs)


def _correctness_case(
    provider,
    name,
    blocks_per_expert,
    hidden,
    intermediate,
    outer_split,
    k_split,
    device,
):
    import torch

    backend = _load_backend(provider)
    tokens = sum(blocks_per_expert) * TILE_K
    dy = _make_hashed_bf16((tokens, hidden), 0x2468ACEF, device)
    z = _make_hashed_bf16((tokens, intermediate), 0x13579BDF, device)
    starts, ends = _make_k_ranges(blocks_per_expert, device)
    state = backend.prepare(dy, z, starts, ends)
    state["output"].zero_()
    actual = backend.launch(state, outer_split, k_split).view(
        len(blocks_per_expert),
        hidden,
        intermediate,
    )
    torch.cuda.synchronize(device)
    expected = _reference(dy, z, starts, ends)
    difference = actual.float() - expected
    if k_split == 1:
        relative = difference.abs() / expected.abs().clamp_min(1e-3)
        mean_tolerance = 0.01
        max_tolerance = 0.05
        normalization = "elementwise"
    else:
        relative = difference.abs() / expected.abs().max().clamp_min(1e-6)
        mean_tolerance = 0.0005
        max_tolerance = math.inf
        normalization = "absmax"
    metrics = {
        "name": name,
        "normalization": normalization,
        "mean_relative": relative.mean().item(),
        "max_relative": relative.max().item(),
        "max_absolute": difference.abs().max().item(),
    }
    print(
        f"CORRECTNESS_CASE,provider={provider},"
        + ",".join(
            f"{key}={value:.9g}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items()
        ),
        flush=True,
    )
    if metrics["mean_relative"] >= mean_tolerance or metrics["max_relative"] >= max_tolerance:
        raise RuntimeError(f"{provider} correctness failed: {metrics}")
    return metrics


def _run_python_correctness(provider, device):
    cases = (
        ("small", [2, 2, 2, 2], 256, 256, 1, 1),
        ("multi-split2", [4, 4, 4, 4], 512, 512, 2, 1),
        ("empty-skew", [0, 1, 5, 2], 256, 256, 1, 1),
        ("all-but-one-empty", [0, 0, 0, 8], 256, 256, 1, 1),
        ("many-empty-prime", [0, 1, 0, 3, 0, 2, 0, 1], 256, 256, 1, 1),
        ("empty-skew-ksplit2", [0, 1, 5, 2], 256, 256, 1, 2),
        ("empty-skew-ksplit3", [0, 1, 5, 2], 256, 256, 1, 3),
        ("all-but-one-empty-ksplit3", [0, 0, 0, 8], 256, 256, 1, 3),
        ("many-empty-prime-ksplit3", [0, 1, 0, 3, 0, 2, 0, 1], 256, 256, 1, 3),
    )
    metrics = [
        _correctness_case(
            provider,
            name,
            blocks,
            hidden,
            intermediate,
            outer_split,
            k_split,
            device,
        )
        for name, blocks, hidden, intermediate, outer_split, k_split in cases
    ]
    result = {
        "passed": True,
        "cases": len(metrics),
        "max_mean_relative": max(item["mean_relative"] for item in metrics),
        "max_relative": max(item["max_relative"] for item in metrics),
        "max_absolute": max(item["max_absolute"] for item in metrics),
    }
    print(
        f"CORRECTNESS,provider={provider},"
        + ",".join(
            f"{key}={value:.9g}" if isinstance(value, float) else f"{key}={int(value)}" for key, value in result.items()
        ),
        flush=True,
    )
    return result


def _divisors(value):
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def _time_python_split(
    backend,
    state,
    outer_split,
    eviction,
    warmups,
    samples,
    device,
):
    import torch

    state["output"].zero_()
    backend.launch(state, outer_split, k_split=1)
    torch.cuda.synchronize(device)

    for iteration in range(warmups):
        state["output"].zero_()
        eviction.fill_(iteration)
        backend.launch(state, outer_split, k_split=1)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    timings = []
    for iteration in range(samples):
        state["output"].zero_()
        eviction.fill_(0x10000 + iteration)
        start_event.record()
        backend.launch(state, outer_split, k_split=1)
        stop_event.record()
        stop_event.synchronize()
        timings.append(start_event.elapsed_time(stop_event))
    return statistics.median(timings)


def _run_python_benchmark(provider, shape, args, device):
    import torch

    backend = _load_backend(provider)
    dy, z, starts, ends = _make_balanced_inputs(shape, device)
    state = backend.prepare(dy, z, starts, ends)
    eviction = torch.empty(
        L2_EVICTION_BYTES // torch.tensor([], dtype=torch.int32).element_size(),
        dtype=torch.int32,
        device=device,
    )

    best = None
    for outer_split in _divisors(shape.hidden // TILE_M):
        latency_ms = _time_python_split(
            backend,
            state,
            outer_split,
            eviction,
            args.warmups,
            args.samples,
            device,
        )
        tflops = 2.0 * shape.tokens * shape.hidden * shape.intermediate / (latency_ms * 1e9)
        cells = shape.experts * (shape.intermediate // TILE_N) * outer_split
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        logical_blocks = max(1, min(sm_count // 2, cells))
        current = {
            "provider": provider,
            "model": shape.name,
            "latency_ms": latency_ms,
            "tflops": tflops,
            "split": outer_split,
            "logical_blocks": logical_blocks,
        }
        print(
            "SWEEP,"
            + ",".join(
                f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in current.items()
            ),
            flush=True,
        )
        if best is None or current["tflops"] > best["tflops"]:
            best = current

    print(
        "RESULT,"
        + ",".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in best.items()
        ),
        flush=True,
    )
    return best


def _worker_main(args):
    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    if args.correctness_only:
        _run_python_correctness(args.worker, device)
        return
    if not args.model or len(args.model) != 1:
        raise ValueError(f"{args.worker} worker requires exactly one --model")
    _run_python_benchmark(
        args.worker,
        MODEL_BY_NAME[args.model[0]],
        args,
        device,
    )


def _provider_environment(args):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    return environment


def _run_process(command, environment, cwd=EXPERIMENT_ROOT):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    return result.stdout


def _parse_key_values(line, start=1):
    result = {}
    for item in line.split(",")[start:]:
        key, value = item.split("=", 1)
        result[key] = value
    return result


def _run_cuda_correctness(provider, args):
    environment = _provider_environment(args)
    output = _run_process(
        [str(CUDA_BINARIES[provider]), "--correctness-only"],
        environment,
    )
    cases = [line for line in output.splitlines() if line.startswith("CORRECTNESS,")]
    if not cases:
        raise RuntimeError("CUDA backend emitted no correctness cases")
    return {"passed": True, "cases": len(cases)}


def _run_python_correctness_process(provider, args):
    output = _run_process(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            provider,
            "--correctness-only",
        ],
        _provider_environment(args),
    )
    lines = [line for line in output.splitlines() if line.startswith("CORRECTNESS,")]
    if len(lines) != 1:
        raise RuntimeError(f"{provider} emitted {len(lines)} aggregate correctness lines")
    fields = _parse_key_values(lines[0])
    return {
        "passed": fields["passed"] == "1",
        "cases": int(fields["cases"]),
        "max_mean_relative": float(fields["max_mean_relative"]),
        "max_relative": float(fields["max_relative"]),
        "max_absolute": float(fields["max_absolute"]),
    }


def _run_cuda_benchmark(provider, shape, args):
    environment = _provider_environment(args)
    environment.update(
        {
            "MLP3_SHAPE": shape.name,
            "MLP3_SKIP_CORRECTNESS": "1",
            "MLP3_WARMUP": str(args.warmups),
            "MLP3_ITERS": str(args.samples),
            "MLP3_FIXED_1SM_SPLIT": "1",
        }
    )
    output = _run_process([str(CUDA_BINARIES[provider])], environment)
    prefix = f"RESULT,{shape.name},"
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(f"CUDA backend emitted {len(lines)} result lines for {shape.name}")
    fields = _parse_key_values(lines[0], start=2)
    return {
        "provider": provider,
        "model": shape.name,
        "latency_ms": float(fields["2sm_ms"]),
        "tflops": float(fields["2sm_tflops"]),
        "split": int(fields["2sm_split"]),
    }


def _run_python_benchmark_process(provider, shape, args):
    output = _run_process(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            provider,
            "--model",
            shape.name,
            "--warmups",
            str(args.warmups),
            "--samples",
            str(args.samples),
        ],
        _provider_environment(args),
    )
    lines = [line for line in output.splitlines() if line.startswith("RESULT,")]
    if len(lines) != 1:
        raise RuntimeError(f"{provider} emitted {len(lines)} result lines for {shape.name}")
    fields = _parse_key_values(lines[0])
    return {
        "provider": provider,
        "model": shape.name,
        "latency_ms": float(fields["latency_ms"]),
        "tflops": float(fields["tflops"]),
        "split": int(fields["split"]),
        "logical_blocks": int(fields["logical_blocks"]),
    }


def _run_provider(provider, shape, args):
    if provider in CUDA_BINARIES:
        return _run_cuda_benchmark(provider, shape, args)
    if provider in PYTHON_PROVIDERS:
        return _run_python_benchmark_process(provider, shape, args)
    raise ValueError(provider)


def _geomean(values):
    values = tuple(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _aggregate(shape, rounds):
    result = asdict(shape)
    for provider in PROVIDERS:
        provider_rounds = [item[provider] for item in rounds]
        result[f"{provider}_latency_ms"] = statistics.median(item["latency_ms"] for item in provider_rounds)
        result[f"{provider}_tflops"] = statistics.median(item["tflops"] for item in provider_rounds)
        result[f"{provider}_split"] = statistics.mode(item["split"] for item in provider_rounds)
    for provider in PYTHON_PROVIDERS:
        result[f"{provider}_over_cuda_s5"] = result["cuda_s5_latency_ms"] / result[f"{provider}_latency_ms"]
        result[f"{provider}_over_cuda_s6"] = result["cuda_s6_latency_ms"] / result[f"{provider}_latency_ms"]
    result["cutedsl_over_cutile"] = result["cutile_latency_ms"] / result["cutedsl_latency_ms"]
    return result


def _write_csv(path, results):
    fieldnames = (
        "name",
        "tokens",
        "hidden",
        "intermediate",
        "experts",
        "cuda_s5_latency_ms",
        "cuda_s5_tflops",
        "cuda_s5_split",
        "cuda_s6_latency_ms",
        "cuda_s6_tflops",
        "cuda_s6_split",
        "cutedsl_latency_ms",
        "cutedsl_tflops",
        "cutedsl_split",
        "cutile_latency_ms",
        "cutile_tflops",
        "cutile_split",
        "cutedsl_over_cuda_s5",
        "cutedsl_over_cuda_s6",
        "cutile_over_cuda_s5",
        "cutile_over_cuda_s6",
        "cutedsl_over_cutile",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)


def _print_summary(results):
    print(
        "\nMODEL,CUDA_S5_MS,CUDA_S6_MS,CUTEDSL_MS,CUTILE_MS,"
        "CUDA_S5_TFLOPS,CUDA_S6_TFLOPS,CUTEDSL_TFLOPS,CUTILE_TFLOPS,"
        "CUTEDSL_OVER_S5,CUTEDSL_OVER_S6,CUTILE_OVER_S5,CUTILE_OVER_S6,"
        "CUTEDSL_OVER_CUTILE",
        flush=True,
    )
    for result in results:
        print(
            f"{result['name']},"
            f"{result['cuda_s5_latency_ms']:.6f},"
            f"{result['cuda_s6_latency_ms']:.6f},"
            f"{result['cutedsl_latency_ms']:.6f},"
            f"{result['cutile_latency_ms']:.6f},"
            f"{result['cuda_s5_tflops']:.3f},"
            f"{result['cuda_s6_tflops']:.3f},"
            f"{result['cutedsl_tflops']:.3f},"
            f"{result['cutile_tflops']:.3f},"
            f"{result['cutedsl_over_cuda_s5']:.6f},"
            f"{result['cutedsl_over_cuda_s6']:.6f},"
            f"{result['cutile_over_cuda_s5']:.6f},"
            f"{result['cutile_over_cuda_s6']:.6f},"
            f"{result['cutedsl_over_cutile']:.6f}",
            flush=True,
        )
    speedups = {
        "cutedsl_over_cuda_s5": _geomean(item["cutedsl_over_cuda_s5"] for item in results),
        "cutedsl_over_cuda_s6": _geomean(item["cutedsl_over_cuda_s6"] for item in results),
        "cutile_over_cuda_s5": _geomean(item["cutile_over_cuda_s5"] for item in results),
        "cutile_over_cuda_s6": _geomean(item["cutile_over_cuda_s6"] for item in results),
        "cutedsl_over_cutile": _geomean(item["cutedsl_over_cutile"] for item in results),
    }
    print(
        "GEOMEAN," + ",".join(f"{key.upper()}={value:.6f}" for key, value in speedups.items()),
        flush=True,
    )
    return speedups


def _orchestrator_main(args):
    missing_binaries = [str(binary) for binary in CUDA_BINARIES.values() if not binary.is_file()]
    if missing_binaries:
        raise FileNotFoundError(
            f"CUDA backends are not built: {missing_binaries}; run cmake -S . -B build && cmake --build build -j2"
        )
    selected = tuple(MODEL_BY_NAME[name] for name in args.model) if args.model else MODEL_SHAPES

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        correctness = {provider: _run_cuda_correctness(provider, args) for provider in CUDA_BINARIES}
        correctness.update({provider: _run_python_correctness_process(provider, args) for provider in PYTHON_PROVIDERS})
        if args.correctness_only:
            return

        raw_rounds = {}
        results = []
        for shape in selected:
            shape_rounds = []
            for round_index in range(args.rounds):
                start = round_index % len(PROVIDERS)
                order = PROVIDERS[start:] + PROVIDERS[:start]
                current = {"round": round_index + 1, "order": order}
                for provider in order:
                    current[provider] = _run_provider(provider, shape, args)
                shape_rounds.append(current)
            raw_rounds[shape.name] = shape_rounds
            results.append(_aggregate(shape, shape_rounds))

    geomeans = _print_summary(results)
    _write_csv(args.csv_output, results)
    args.raw_output.write_text(
        json.dumps(
            {
                "protocol": {
                    "rounds": args.rounds,
                    "samples_per_split_per_round": args.samples,
                    "warmups_per_split_per_round": args.warmups,
                    "l2_eviction_bytes": L2_EVICTION_BYTES,
                    "provider_order": "rotates each round",
                    "timing": "CUDA events around one kernel launch",
                    "output": "BF16 reduction add",
                    "tile": [TILE_M, TILE_N, TILE_K],
                    "cuda_configurations": {
                        "cuda_s5": {
                            "stages": 5,
                            "epilogue_n": 64,
                            "accumulator_stages": 2,
                        },
                        "cuda_s6": {
                            "stages": 6,
                            "epilogue_n": 64,
                            "accumulator_stages": 2,
                        },
                    },
                    "cutile_configuration": {
                        "num_ctas": 2,
                        "occupancy": 1,
                        "load_latency": 10,
                    },
                    "cutedsl_configuration": {
                        "cta_group": 2,
                        "cluster_shape": [2, 1],
                        "accumulator_stages": 2,
                        "epilogue_warpgroups": 2,
                    },
                },
                "correctness": correctness,
                "rounds": raw_rounds,
                "results": results,
                "geomeans": geomeans,
            },
            indent=2,
        )
        + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=PYTHON_PROVIDERS)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", action="append", choices=MODEL_BY_NAME)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=EXPERIMENT_ROOT / "results.csv",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=EXPERIMENT_ROOT / "results_raw.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        _worker_main(args)
    else:
        _orchestrator_main(args)


if __name__ == "__main__":
    main()
