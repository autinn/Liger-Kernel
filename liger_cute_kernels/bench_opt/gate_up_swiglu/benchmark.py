"""Benchmark every gate/up SwiGLU implementation in this experiment."""

import argparse
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
BACKENDS_DIR = EXPERIMENT_ROOT / "backends"
CUDA_BINARY = EXPERIMENT_ROOT / "build/cuda_cpp"
LOCK_PATH = Path("/data/ssd/all-mlp-2sm.lock")

PYTHONPATH = os.pathsep.join(
    (
        str(EXPERIMENT_ROOT),
        "/tmp/cutlass_dsl_452",
        "/tmp/cutlass_dsl_452/nvidia_cutlass_dsl/python_packages",
        "/tmp/swiglu_pkgs",
    )
)
TILEIRAS_BIN = "/home/jobuser/.local/lib/python3.12/site-packages/nvidia/cu13/bin"

PROVIDERS = (
    "cuda_cpp",
    "cutedsl",
    "cutile",
    "triton",
    "cublas_triton",
    "cublas_cutedsl",
)
PYTHON_PROVIDERS = tuple(provider for provider in PROVIDERS if provider != "cuda_cpp")
BASELINE_PROVIDER = "cuda_cpp"

TILE_M = 128
INPUT_SEED = 0x12345678
GATE_SEED = 0x9ABCDEF0
UP_SEED = 0x31415926


@dataclass(frozen=True)
class ModelShape:
    name: str
    m: int
    h: int
    i: int
    e: int


MODEL_SHAPES = (
    ModelShape("Qwen3-30B-A3B", 8192, 2048, 768, 8),
    ModelShape("Qwen3-235B-A22B", 8192, 4096, 1536, 8),
    ModelShape("Qwen3.5-122B-A10B", 8192, 3072, 1024, 8),
    ModelShape("Llama-4-Scout-17B-16E", 8192, 5120, 8192, 8),
    ModelShape("Mixtral-8x7B", 8192, 4096, 14336, 8),
    ModelShape("Mixtral-8x22B", 8192, 6144, 16384, 8),
)
MODEL_BY_NAME = {shape.name: shape for shape in MODEL_SHAPES}


def _make_hashed_bf16(shape, seed, device, chunk_elements=4 * 1024 * 1024):
    import torch

    count = math.prod(shape)
    output = torch.empty(count, dtype=torch.bfloat16, device=device)
    mask = 0xFFFFFFFF
    for start in range(0, count, chunk_elements):
        end = min(start + chunk_elements, count)
        index = torch.arange(start, end, dtype=torch.int64, device=device)
        value = (index ^ (index >> 32) ^ seed) & mask
        value = ((value ^ (value >> 16)) * 0x7FEB352D) & mask
        value = ((value ^ (value >> 15)) * 0x846CA68B) & mask
        value = (value ^ (value >> 16)) & mask
        finite = ((value & 0xFFFF).to(torch.float32) / 65535.0 - 0.5) * 0.5
        output[start:end].copy_(finite)
    return output.view(shape)


def _make_inputs(shape, device):
    import torch

    x = _make_hashed_bf16((shape.m, shape.h), INPUT_SEED, device)
    gate = _make_hashed_bf16((shape.e, shape.i, shape.h), GATE_SEED, device)
    up = _make_hashed_bf16((shape.e, shape.i, shape.h), UP_SEED, device)
    num_m_tiles = math.ceil(shape.m / TILE_M)
    rows = torch.arange(num_m_tiles, dtype=torch.int64, device=device) * TILE_M
    expert_ids = torch.clamp(rows * shape.e // shape.m, max=shape.e - 1).to(torch.int32)
    return x, gate, up, expert_ids


def _reference(x, gate, up, expert_ids):
    import torch
    import torch.nn.functional as functional

    output = torch.empty(
        (x.shape[0], gate.shape[1]),
        dtype=torch.float32,
        device=x.device,
    )
    for m_tile, expert in enumerate(expert_ids.tolist()):
        row_start = m_tile * TILE_M
        row_end = min(row_start + TILE_M, x.shape[0])
        x_block = x[row_start:row_end].float()
        gate_block = x_block @ gate[expert].float().T
        up_block = x_block @ up[expert].float().T
        output[row_start:row_end] = functional.silu(gate_block) * up_block
    return output


def _load_backend(provider):
    if provider not in PYTHON_PROVIDERS:
        raise ValueError(f"{provider} is not a Python backend")
    module_name = f"gate_up_swiglu_backend_{provider}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = BACKENDS_DIR / f"{provider}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load backend {provider} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _capture_graph(launch, device):
    import torch

    launch()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize(device)
    return graph


def _warm(graph, warmup_ms, launch_warmups, device):
    import torch

    start = time.perf_counter()
    while (time.perf_counter() - start) * 1000.0 < warmup_ms:
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize(device)
    for _ in range(launch_warmups):
        graph.replay()
    torch.cuda.synchronize(device)


def _sample(graph, samples):
    import torch

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(samples):
        start_event.record()
        graph.replay()
        stop_event.record()
        stop_event.synchronize()
        timings.append(start_event.elapsed_time(stop_event))
    return timings


def _run_python_correctness(provider, device):
    import torch

    backend = _load_backend(provider)
    shape = ModelShape("correctness", 768, 512, 256, 3)
    x, gate, up, expert_ids = _make_inputs(shape, device)
    state = backend.prepare(x, gate, up, expert_ids)
    actual = backend.launch(state).view(shape.m, shape.i)
    torch.cuda.synchronize(device)
    expected = _reference(x, gate, up, expert_ids)
    difference = actual.float() - expected
    relative_frobenius = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected)
    relative = difference.abs() / expected.abs().clamp_min(1e-3)
    metrics = {
        "relative_frobenius": relative_frobenius.item(),
        "mean_relative": relative.mean().item(),
        "max_relative": relative.max().item(),
        "max_absolute": difference.abs().max().item(),
    }
    print(
        f"CORRECTNESS,provider={provider}," + ",".join(f"{key}={value:.9g}" for key, value in metrics.items()),
        flush=True,
    )
    if metrics["relative_frobenius"] >= 0.05 or metrics["mean_relative"] >= 0.01:
        raise RuntimeError(f"{provider} correctness gate failed: {metrics}")


def _run_python_benchmark(provider, shape, args, device):
    import torch

    backend = _load_backend(provider)
    x, gate, up, expert_ids = _make_inputs(shape, device)
    state = backend.prepare(x, gate, up, expert_ids)
    launch = lambda: backend.launch(state)
    graph = _capture_graph(launch, device)
    graph.replay()
    torch.cuda.synchronize(device)

    round_medians = []
    for round_index in range(args.worker_rounds):
        _warm(graph, args.warmup_ms, args.launch_warmups, device)
        timings = _sample(graph, args.samples)
        round_median = statistics.median(timings)
        round_medians.append(round_median)
        print(
            f"ROUND,provider={provider},model={shape.name},round={round_index + 1},median_ms={round_median:.6f}",
            flush=True,
        )

    latency_ms = statistics.median(round_medians)
    tflops = 4.0 * shape.m * shape.h * shape.i / (latency_ms * 1e9)
    print(
        f"RESULT,provider={provider},model={shape.name},"
        f"latency_ms={latency_ms:.6f},tflops={tflops:.4f},"
        f"round_min_ms={min(round_medians):.6f},"
        f"round_max_ms={max(round_medians):.6f}",
        flush=True,
    )


def _worker_main(args):
    import torch

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    if args.correctness_only:
        _run_python_correctness(args.worker, device)
        return
    model_name = args.model[0] if isinstance(args.model, list) else args.model
    _run_python_benchmark(args.worker, MODEL_BY_NAME[model_name], args, device)


def _provider_environment(provider, device):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    if provider != "cuda_cpp":
        environment["PYTHONPATH"] = PYTHONPATH
    if provider == "cutile":
        environment["PATH"] = TILEIRAS_BIN + os.pathsep + environment.get("PATH", "")
    return environment


def _provider_command(provider, args, model=None, correctness=False):
    common = [
        "--samples",
        str(args.samples),
        "--warmup-ms",
        str(args.warmup_ms),
        "--launch-warmups",
        str(args.launch_warmups),
    ]
    if provider == "cuda_cpp":
        command = [str(CUDA_BINARY)]
        if correctness:
            command.append("--correctness-only")
        else:
            command.extend(["--model", model, "--rounds", "1", *common])
        return command

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        provider,
        "--device",
        "0",
    ]
    if correctness:
        command.append("--correctness-only")
    else:
        command.extend(
            [
                "--model",
                model,
                "--worker-rounds",
                "1",
                *common,
            ]
        )
    return command


def _parse_fields(line):
    fields = {}
    for item in line.strip().split(",")[1:]:
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def _run_provider(provider, args, model=None, correctness=False):
    command = _provider_command(provider, args, model=model, correctness=correctness)
    result = subprocess.run(
        command,
        cwd=EXPERIMENT_ROOT,
        env=_provider_environment(provider, args.device),
        check=True,
        text=True,
        capture_output=True,
    )
    print(result.stdout, end="", flush=True)
    prefix = "CORRECTNESS," if correctness else "RESULT,"
    lines = [line for line in result.stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(f"{provider} emitted {len(lines)} {prefix.rstrip(',')} lines; stderr:\n{result.stderr}")
    fields = _parse_fields(lines[0])
    fields["command"] = command
    return fields


def _geomean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _aggregate(shape, providers, round_results):
    result = {
        "model": shape.name,
        "M": shape.m,
        "H": shape.h,
        "I": shape.i,
        "E": shape.e,
    }
    for provider in providers:
        latencies = [float(item[provider]["latency_ms"]) for item in round_results]
        latency = statistics.median(latencies)
        tflops = 4.0 * shape.m * shape.h * shape.i / (latency * 1e9)
        result[f"{provider}_latency_ms"] = f"{latency:.6f}"
        result[f"{provider}_tflops"] = f"{tflops:.4f}"
        result[f"{provider}_round_min_ms"] = min(latencies)
        result[f"{provider}_round_max_ms"] = max(latencies)
        result[f"{provider}_round_medians_ms"] = latencies

    baseline_tflops = float(result[f"{BASELINE_PROVIDER}_tflops"])
    for provider in providers:
        result[f"{provider}_over_{BASELINE_PROVIDER}"] = f"{float(result[f'{provider}_tflops']) / baseline_tflops:.6f}"
    return result


def _write_csv(path, providers, results):
    fieldnames = ["model", "M", "H", "I", "E"]
    for provider in providers:
        fieldnames.extend(
            (
                f"{provider}_latency_ms",
                f"{provider}_tflops",
                f"{provider}_over_{BASELINE_PROVIDER}",
            )
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})


def _print_summary(providers, results):
    print("\nMODEL," + ",".join(f"{provider.upper()}_TFLOPS" for provider in providers))
    for result in results:
        print(result["model"] + "," + ",".join(result[f"{provider}_tflops"] for provider in providers))

    geomeans = {
        provider: _geomean([float(result[f"{provider}_tflops"]) for result in results]) for provider in providers
    }
    baseline = geomeans[BASELINE_PROVIDER]
    print("GEOMEAN," + ",".join(f"{geomeans[provider]:.4f}" for provider in providers))
    print(
        f"SPEEDUP_VS_{BASELINE_PROVIDER.upper()},"
        + ",".join(f"{geomeans[provider] / baseline:.6f}" for provider in providers)
    )
    return {
        provider: {
            "tflops": geomeans[provider],
            f"over_{BASELINE_PROVIDER}": geomeans[provider] / baseline,
        }
        for provider in providers
    }


def _orchestrator_main(args):
    providers = tuple(args.providers)
    if BASELINE_PROVIDER not in providers:
        raise ValueError(f"{BASELINE_PROVIDER} must be included as the baseline")
    if not CUDA_BINARY.is_file():
        raise FileNotFoundError(
            f"CUDA backend is not built: {CUDA_BINARY}; run cmake -S . -B build && cmake --build build"
        )

    selected_shapes = tuple(MODEL_BY_NAME[name] for name in args.model) if args.model else MODEL_SHAPES
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        correctness = {provider: _run_provider(provider, args, correctness=True) for provider in providers}

        raw_rounds = {}
        results = []
        for shape in selected_shapes:
            model_rounds = []
            for round_index in range(args.rounds):
                start = round_index % len(providers)
                order = providers[start:] + providers[:start]
                current = {"round": round_index + 1, "order": order}
                for provider in order:
                    current[provider] = _run_provider(
                        provider,
                        args,
                        model=shape.name,
                    )
                model_rounds.append(current)
            raw_rounds[shape.name] = model_rounds
            results.append(_aggregate(shape, providers, model_rounds))

    geomean = _print_summary(providers, results)
    _write_csv(args.csv_output, providers, results)
    args.raw_output.write_text(
        json.dumps(
            {
                "protocol": {
                    "rounds": args.rounds,
                    "samples_per_round": args.samples,
                    "warmup_ms_per_round": args.warmup_ms,
                    "launch_warmups_per_round": args.launch_warmups,
                    "provider_order": "rotates each round",
                    "timing": "CUDA events around one CUDA graph replay",
                },
                "providers": providers,
                "correctness": correctness,
                "rounds": raw_rounds,
                "results": results,
                "geomean": geomean,
            },
            indent=2,
        )
        + "\n"
    )


def _read(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(errors="replace")


def _count(pattern, text):
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resource_values(text, function_pattern):
    match = re.search(
        rf"Function {function_pattern}.*?:\n"
        r"\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        text,
    )
    if not match:
        raise RuntimeError(f"resource entry not found: {function_pattern}")
    return {
        "registers": int(match.group(1)),
        "stack_bytes": int(match.group(2)),
        "static_shared_bytes": int(match.group(3)),
        "local_bytes": int(match.group(4)),
    }


def _sass_metrics(text):
    return {
        "utchmma_1cta_static": _count(r"\bUTCHMMA\b(?!\.2CTA)", text),
        "utchmma_2cta_static": _count(r"\bUTCHMMA\.2CTA\b", text),
        "utma_load_static": _count(r"\bUTMALDG\b", text),
        "utma_load_2cta_static": _count(r"\bUTMALDG\.[^\s]*2CTA\b", text),
        "utma_store_static": _count(r"\bUTMASTG\b", text),
        "mufu_exp2_static": _count(r"\bMUFU\.EX2\b", text),
        "mufu_rcp_static": _count(r"\bMUFU\.RCP\b", text),
        "local_load_store_static": _count(
            r"(?:^|\s)(?:LDL|STL)(?:\.|\s)",
            text,
        ),
    }


def _ptx_metrics(text):
    return {
        "tcgen05_mma_1cta_static": _count(
            r"tcgen05\.mma\.cta_group::1",
            text,
        ),
        "tcgen05_mma_2cta_static": _count(
            r"tcgen05\.mma\.cta_group::2",
            text,
        ),
        "rcp_approx_static": _count(r"rcp\.approx", text),
        "exp2_approx_static": _count(r"ex2\.approx", text),
    }


def _codegen_main(args):
    artifacts = args.artifact_dir
    tileir_paths = sorted((artifacts / "cutile_dump").glob("*.tileirbc"))
    if len(tileir_paths) != 1:
        raise RuntimeError(f"expected one cuTile bytecode file, found {tileir_paths}")

    result = {
        "comparison": {
            "operation": "Z = SiLU(X @ W_gate[e].T) * (X @ W_up[e].T)",
            "tile": [128, 128, 64],
            "cta_mode": "1CTA",
            "math": "approximate reciprocal",
        },
        "providers": {},
    }
    resource_patterns = {
        "cuda": r"_ZN19gate_up_swiglu_cuda12fused_kernel",
        "cutedsl": r"kernel_cutlass_kernel_.*FusedSwigluGateUpPersistentKernel.*",
        "cutile": r"cutile_gate_up_swiglu_kernel_.*",
    }
    for provider in ("cuda", "cutedsl", "cutile"):
        sass_path = artifacts / f"{provider}.sass"
        cubin_path = artifacts / f"{provider}.cubin"
        resources_path = artifacts / f"{provider}_resources.txt"
        result["providers"][provider] = {
            "sass": _sass_metrics(_read(sass_path)),
            "resources": _resource_values(
                _read(resources_path),
                resource_patterns[provider],
            ),
            "artifacts": {
                "sass": {
                    "path": str(sass_path),
                    "sha256": _sha256(sass_path),
                },
                "cubin": {
                    "path": str(cubin_path),
                    "sha256": _sha256(cubin_path),
                },
            },
        }

    for provider in ("cuda", "cutedsl"):
        ptx_path = artifacts / f"{provider}.ptx"
        result["providers"][provider]["ptx"] = _ptx_metrics(_read(ptx_path))
        result["providers"][provider]["artifacts"]["ptx"] = {
            "path": str(ptx_path),
            "sha256": _sha256(ptx_path),
        }
    result["providers"]["cutile"]["ptx"] = {
        "available": False,
        "reason": "TileIRAS compiles TileIR bytecode directly to cubin",
    }
    result["providers"]["cutile"]["tileir_bytecode"] = {
        "path": str(tileir_paths[0]),
        "sha256": _sha256(tileir_paths[0]),
        "bytes": tileir_paths[0].stat().st_size,
    }

    failures = []
    for provider, provider_result in result["providers"].items():
        sass = provider_result["sass"]
        if sass["utchmma_1cta_static"] < 8:
            failures.append(f"{provider}: fewer than eight 1CTA MMA sites")
        if sass["utchmma_2cta_static"] != 0:
            failures.append(f"{provider}: unexpected 2CTA MMA sites")
        if sass["utma_load_static"] < 3:
            failures.append(f"{provider}: fewer than three TMA loads")
        if sass["utma_store_static"] < 1:
            failures.append(f"{provider}: missing TMA output store")
        if sass["mufu_exp2_static"] < 1 or sass["mufu_rcp_static"] < 1:
            failures.append(f"{provider}: missing fast sigmoid instructions")
        if sass["local_load_store_static"] != 0:
            failures.append(f"{provider}: local-memory instructions present")
        resources = provider_result["resources"]
        if resources["stack_bytes"] != 0 or resources["local_bytes"] != 0:
            failures.append(f"{provider}: stack/local allocation present")
    for provider in ("cuda", "cutedsl"):
        ptx = result["providers"][provider]["ptx"]
        if ptx["tcgen05_mma_1cta_static"] != 8:
            failures.append(f"{provider}: PTX must contain eight 1CTA MMAs")
        if ptx["tcgen05_mma_2cta_static"] != 0:
            failures.append(f"{provider}: PTX unexpectedly contains 2CTA MMAs")
        if ptx["rcp_approx_static"] < 1 or ptx["exp2_approx_static"] < 1:
            failures.append(f"{provider}: PTX is not fast math")

    result["logic_check"] = {
        "passed": not failures,
        "failures": failures,
    }
    args.codegen_output.write_text(json.dumps(result, indent=2) + "\n")
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=PYTHON_PROVIDERS)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--worker-rounds", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup-ms", type=float, default=200.0)
    parser.add_argument("--launch-warmups", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", action="append", choices=MODEL_BY_NAME)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
    )
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
    parser.add_argument("--check-codegen", action="store_true")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts",
    )
    parser.add_argument(
        "--codegen-output",
        type=Path,
        default=EXPERIMENT_ROOT / "codegen.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        _worker_main(args)
    elif args.check_codegen:
        _codegen_main(args)
    else:
        _orchestrator_main(args)


if __name__ == "__main__":
    main()
