"""Capture and verify native Blackwell code for every MLP3 2SM provider."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys

from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
BACKEND_PATHS = {
    "cutedsl": EXPERIMENT_ROOT / "backends/cutedsl.py",
    "cutile": EXPERIMENT_ROOT / "backends/cutile.py",
}
CUDA_BINARIES = {
    "cuda_s5": EXPERIMENT_ROOT / "build/cuda_mlp3_2sm",
    "cuda_s6": EXPERIMENT_ROOT / "build/cuda_mlp3_2sm_s6",
}
LOCK_PATH = Path("/data/ssd/all-mlp-2sm.lock")


def _load_backend(provider):
    spec = importlib.util.spec_from_file_location(
        f"mlp3_{provider}_codegen_backend",
        BACKEND_PATHS[provider],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {provider} backend from {BACKEND_PATHS[provider]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker(provider):
    import torch

    backend = _load_backend(provider)
    device = torch.device("cuda:0")
    dy = torch.zeros((512, 256), dtype=torch.bfloat16, device=device)
    z = torch.zeros((512, 256), dtype=torch.bfloat16, device=device)
    starts = torch.tensor((0, 2, 4, 6), dtype=torch.int32, device=device)
    ends = torch.tensor((2, 4, 6, 8), dtype=torch.int32, device=device)
    state = backend.prepare(dy, z, starts, ends)
    state["output"].zero_()
    backend.launch(state, outer_split=1, k_split=1)
    torch.cuda.synchronize(device)
    print(f"CODEGEN_PROBE,provider={provider},passed=1")
    if hasattr(backend, "compile_metadata"):
        print("CODEGEN_METADATA," + json.dumps(backend.compile_metadata(), sort_keys=True))


def _run(command, *, environment=None):
    result = subprocess.run(
        command,
        cwd=EXPERIMENT_ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.stdout


def _clean_files(directory, patterns):
    directory.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in directory.glob(pattern):
            if not path.is_file():
                raise RuntimeError(f"refusing to remove non-file artifact: {path}")
            path.unlink()


def _extract_single_cached_cubin(cache_db, output_path):
    connection = sqlite3.connect(cache_db)
    try:
        rows = connection.execute("SELECT key, blob FROM cache ORDER BY key").fetchall()
    finally:
        connection.close()
    if len(rows) != 1:
        raise RuntimeError(f"expected one cuTile cache entry, found {len(rows)}")
    key, blob = rows[0]
    output_path.write_bytes(blob)
    return key


def _dump(cuobjdump, source, sass_path, resources_path):
    sass = subprocess.run(
        [cuobjdump, "--dump-sass", str(source)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    resources = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(source)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    sass_path.write_text(sass)
    resources_path.write_text(resources)


def _count(pattern, text):
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def _sass_metrics(text):
    return {
        "utchmma_1cta_static": _count(r"\bUTCHMMA\b(?!\.2CTA)", text),
        "utchmma_2cta_static": _count(r"\bUTCHMMA\.2CTA\b", text),
        "utma_load_2cta_static": _count(r"\bUTMALDG\.[^\s]*2CTA\b", text),
        "utma_reduce_add_static": _count(r"\bUTMAREDG\.[^\s]*\.ADD\b", text),
        "cga_barrier_arrive_static": _count(r"\bUCGABAR_ARV\b", text),
        "cga_barrier_wait_static": _count(r"\bUCGABAR_WAIT\b", text),
        "local_load_store_static": _count(r"(?:^|\s)(?:LDL|STL)(?:\.|\s)", text),
    }


def _ptx_metrics(text):
    return {
        "tcgen05_mma_2cta_static": _count(
            r"tcgen05\.mma\.cta_group::2",
            text,
        ),
        "tma_load_2cta_static": _count(
            r"cp\.async\.bulk\.tensor\.[^\n;]*cta_group::2",
            text,
        ),
        "tma_reduce_add_static": _count(
            r"cp\.reduce\.async\.bulk\.tensor\.[^\n;]*\.add\.",
            text,
        ),
    }


def _function_sass(text, marker):
    lines = text.splitlines(keepends=True)
    function_starts = [index for index, line in enumerate(lines) if "Function :" in line]
    matching = [index for index in function_starts if marker in lines[index]]
    if len(matching) != 1:
        raise RuntimeError(f"expected one SASS function containing {marker!r}, found {len(matching)}")
    start = matching[0]
    later_starts = [index for index in function_starts if index > start]
    end = later_starts[0] if later_starts else len(lines)
    return "".join(lines[start:end])


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path):
    return {
        "path": str(path.relative_to(EXPERIMENT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _capture(args):
    missing = [str(path) for path in CUDA_BINARIES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing CUDA binaries: {missing}")
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        candidate = Path("/usr/local/cuda/bin/cuobjdump")
        if not candidate.is_file():
            raise FileNotFoundError("cuobjdump")
        cuobjdump = str(candidate)

    artifacts = args.artifact_dir
    cache_dir = artifacts / "cutile_cache"
    dump_dir = artifacts / "cutile_dump"
    cutedsl_dump_dir = artifacts / "cutedsl_dump"
    _clean_files(cache_dir, ("cache.db", "cache.db-shm", "cache.db-wal"))
    _clean_files(dump_dir, ("*.tileirbc",))
    _clean_files(cutedsl_dump_dir, ("*.cubin", "*.ptx", "*.sass", "*.mlir"))
    artifacts.mkdir(parents=True, exist_ok=True)
    for name in (
        "cutile.cubin",
        "cutile.sass",
        "cutile_resources.txt",
        "cutedsl.cubin",
        "cutedsl.ptx",
        "cutedsl.sass",
        "cutedsl_resources.txt",
        "cuda_s5.sass",
        "cuda_s5_resources.txt",
        "cuda_s6.sass",
        "cuda_s6_resources.txt",
    ):
        path = artifacts / name
        if path.is_file():
            path.unlink()

    cutile_environment = os.environ.copy()
    cutile_environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    cutile_environment["CUDA_TILE_CACHE_DIR"] = str(cache_dir)
    cutile_environment["CUDA_TILE_DUMP_BYTECODE"] = str(dump_dir)
    cutedsl_environment = os.environ.copy()
    cutedsl_environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    cutedsl_environment["CUTE_DSL_NO_CACHE"] = "1"
    cutedsl_environment["CUTE_DSL_KEEP"] = "ptx,cubin"
    cutedsl_environment["CUTE_DSL_DUMP_DIR"] = str(cutedsl_dump_dir)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        _run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "cutile",
            ],
            environment=cutile_environment,
        )
        cutedsl_output = _run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "cutedsl",
            ],
            environment=cutedsl_environment,
        )
    metadata_lines = [line for line in cutedsl_output.splitlines() if line.startswith("CODEGEN_METADATA,")]
    if len(metadata_lines) != 1:
        raise RuntimeError(f"expected one CuTe DSL metadata line, found {metadata_lines}")
    cutedsl_metadata = json.loads(metadata_lines[0].split(",", 1)[1])

    cutile_cubin = artifacts / "cutile.cubin"
    cache_key = _extract_single_cached_cubin(cache_dir / "cache.db", cutile_cubin)
    cutile_sass = artifacts / "cutile.sass"
    cutile_resources = artifacts / "cutile_resources.txt"
    _dump(cuobjdump, cutile_cubin, cutile_sass, cutile_resources)
    cutedsl_cubins = list(cutedsl_dump_dir.glob("*.cubin"))
    cutedsl_ptx_files = list(cutedsl_dump_dir.glob("*.ptx"))
    if len(cutedsl_cubins) != 1 or len(cutedsl_ptx_files) != 1:
        raise RuntimeError(f"expected one CuTe DSL cubin and PTX artifact, found {cutedsl_cubins}, {cutedsl_ptx_files}")
    cutedsl_cubin = artifacts / "cutedsl.cubin"
    cutedsl_ptx = artifacts / "cutedsl.ptx"
    shutil.copyfile(cutedsl_cubins[0], cutedsl_cubin)
    shutil.copyfile(cutedsl_ptx_files[0], cutedsl_ptx)
    cutedsl_sass = artifacts / "cutedsl.sass"
    cutedsl_resources = artifacts / "cutedsl_resources.txt"
    _dump(
        cuobjdump,
        cutedsl_cubin,
        cutedsl_sass,
        cutedsl_resources,
    )
    cuda_artifacts = {}
    for provider, binary in CUDA_BINARIES.items():
        sass_path = artifacts / f"{provider}.sass"
        resources_path = artifacts / f"{provider}_resources.txt"
        _dump(cuobjdump, binary, sass_path, resources_path)
        cuda_artifacts[provider] = (sass_path, resources_path)

    tileir_paths = list(dump_dir.glob("*.tileirbc"))
    if len(tileir_paths) != 1:
        raise RuntimeError(f"expected one TileIR bytecode artifact, found {tileir_paths}")

    providers = {}
    for provider, (sass_path, resources_path) in cuda_artifacts.items():
        providers[provider] = {
            "sass": _sass_metrics(
                _function_sass(
                    sass_path.read_text(errors="replace"),
                    "mlp3_two_sm_kernel",
                )
            ),
            "artifacts": {
                "sass": _artifact(sass_path),
                "resources": _artifact(resources_path),
            },
        }
    providers["cutile"] = {
        "sass": _sass_metrics(
            _function_sass(
                cutile_sass.read_text(errors="replace"),
                "cutile_mlp3_2sm_kernel",
            )
        ),
        "cache_key": cache_key,
        "artifacts": {
            "cubin": _artifact(cutile_cubin),
            "sass": _artifact(cutile_sass),
            "resources": _artifact(cutile_resources),
            "tileir_bytecode": _artifact(tileir_paths[0]),
        },
    }
    providers["cutedsl"] = {
        "configuration": cutedsl_metadata,
        "sass": _sass_metrics(
            _function_sass(
                cutedsl_sass.read_text(errors="replace"),
                "Mlp3TwoSmPersistentKernel",
            )
        ),
        "ptx": _ptx_metrics(cutedsl_ptx.read_text(errors="replace")),
        "artifacts": {
            "cubin": _artifact(cutedsl_cubin),
            "ptx": _artifact(cutedsl_ptx),
            "sass": _artifact(cutedsl_sass),
            "resources": _artifact(cutedsl_resources),
        },
    }

    failures = []
    for provider, data in providers.items():
        metrics = data["sass"]
        if metrics["utchmma_2cta_static"] < 1:
            failures.append(f"{provider}: missing 2CTA MMA")
        if metrics["utma_load_2cta_static"] < 1:
            failures.append(f"{provider}: missing 2CTA TMA load")
        if metrics["utma_reduce_add_static"] < 1:
            failures.append(f"{provider}: missing TMA reduction-add store")
        if metrics["cga_barrier_arrive_static"] < 1:
            failures.append(f"{provider}: missing CGA barrier arrive")
        if metrics["cga_barrier_wait_static"] < 1:
            failures.append(f"{provider}: missing CGA barrier wait")
        if metrics["local_load_store_static"] != 0:
            failures.append(f"{provider}: local-memory instructions present")
    if providers["cutile"]["sass"]["utchmma_1cta_static"] != 0:
        failures.append("cutile: unexpected 1CTA MMA")
    if providers["cutedsl"]["sass"]["utchmma_1cta_static"] != 0:
        failures.append("cutedsl: unexpected 1CTA MMA")
    for key, value in providers["cutedsl"]["ptx"].items():
        if value < 1:
            failures.append(f"cutedsl: missing PTX evidence for {key}")

    result = {
        "operation": "dA[expert] = dY[expert].T @ Z[expert]",
        "logical_tile": [256, 256, 64],
        "cta_mode": "2CTA",
        "providers": providers,
        "logic_check": {
            "passed": not failures,
            "failures": failures,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError("; ".join(failures))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=BACKEND_PATHS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "codegen.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        _capture(args)


if __name__ == "__main__":
    main()
