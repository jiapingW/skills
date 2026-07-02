#!/usr/bin/env python3
"""Scan AI infra repositories for common performance review targets."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "third_party",
    "vendor",
    "wandb",
}

DEFAULT_SUFFIXES = {
    ".py",
    ".pyi",
    ".cu",
    ".cuh",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
}


@dataclass(frozen=True)
class Pattern:
    name: str
    severity: str
    regex: re.Pattern[str]
    note: str


PATTERNS = [
    Pattern(
        "device_to_host_sync",
        "high",
        re.compile(r"\.(item|tolist|numpy)\s*\("),
        "Device-to-host sync or transfer in a hot path can serialize GPU work.",
    ),
    Pattern(
        "explicit_cpu_transfer",
        "high",
        re.compile(r"\.cpu\s*\("),
        "CPU transfer in model, scheduler, or metric code can dominate latency.",
    ),
    Pattern(
        "cuda_synchronize",
        "medium",
        re.compile(r"torch\.cuda\.synchronize\s*\(|cudaDeviceSynchronize\s*\("),
        "Synchronization is valid at measurement boundaries but suspicious inside hot paths.",
    ),
    Pattern(
        "host_wall_clock_timing",
        "medium",
        re.compile(r"\b(time\.time|time\.perf_counter)\s*\("),
        "Host timers around async GPU work need synchronization or CUDA events.",
    ),
    Pattern(
        "distributed_barrier",
        "medium",
        re.compile(r"\b(dist|torch\.distributed)\.barrier\s*\("),
        "Barriers can hide rank skew and create cluster-wide stalls.",
    ),
    Pattern(
        "collective",
        "info",
        re.compile(r"\b(all_reduce|all_gather|reduce_scatter|all_to_all|broadcast)\s*\("),
        "Collective call: check overlap, tensor size, frequency, and rank balance.",
    ),
    Pattern(
        "contiguous_copy",
        "medium",
        re.compile(r"\.contiguous\s*\("),
        "Contiguous conversions can be hidden full-tensor copies.",
    ),
    Pattern(
        "empty_cache_hotpath",
        "medium",
        re.compile(r"torch\.cuda\.empty_cache\s*\("),
        "empty_cache in a loop often masks allocator issues and hurts performance.",
    ),
    Pattern(
        "retain_graph",
        "medium",
        re.compile(r"retain_graph\s*=\s*True"),
        "retain_graph=True can keep activations alive and inflate memory.",
    ),
    Pattern(
        "find_unused_parameters",
        "medium",
        re.compile(r"find_unused_parameters\s*=\s*True"),
        "DDP unused-parameter detection adds overhead and can indicate graph issues.",
    ),
]


def iter_files(root: Path, suffixes: set[str], skip_dirs: set[str]):
    for path in root.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.name == Path(__file__).name:
            continue
        if path.is_file() and path.suffix in suffixes:
            yield path


def scan_file(path: Path, root: Path):
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return

    rel = path.relative_to(root)
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        for pattern in PATTERNS:
            if pattern.regex.search(line):
                yield {
                    "file": str(rel),
                    "line": lineno,
                    "severity": pattern.severity,
                    "kind": pattern.name,
                    "code": stripped[:180],
                    "note": pattern.note,
                }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan for common AI infra performance review targets."
    )
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan")
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=[],
        help="Additional file suffix to scan, for example --suffix .cu",
    )
    parser.add_argument(
        "--skip-dir",
        action="append",
        default=[],
        help="Additional directory name to skip",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    suffixes = set(DEFAULT_SUFFIXES)
    suffixes.update(args.suffix)
    skip_dirs = set(DEFAULT_SKIP_DIRS)
    skip_dirs.update(args.skip_dir)

    findings = []
    for path in iter_files(root, suffixes, skip_dirs):
        findings.extend(scan_file(path, root) or [])

    severity_rank = {"high": 0, "medium": 1, "info": 2}
    findings.sort(key=lambda x: (severity_rank.get(x["severity"], 9), x["file"], x["line"]))

    if args.json:
        print(json.dumps(findings, indent=2, sort_keys=True))
        return 0

    if not findings:
        print("No common AI infra performance review targets found.")
        return 0

    for item in findings:
        print(
            f"{item['severity'].upper():6} {item['file']}:{item['line']} "
            f"{item['kind']} - {item['note']}"
        )
        print(f"       {item['code']}")

    print(
        f"\nFound {len(findings)} review targets. Treat these as leads, not proof of bugs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
