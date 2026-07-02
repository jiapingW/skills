---
name: ai-infra-optimizer
description: AI infrastructure performance engineering for training and inference systems. Use when Codex needs to profile, debug, review, benchmark, or optimize model training or inference efficiency, including PyTorch/JAX/XLA code, CUDA/Triton kernels, distributed training, NCCL/collective communication, compiler stacks, inference serving, batching/scheduling, KV cache, quantization, memory, throughput, latency, MFU, cost, or performance regressions.
---

# AI Infra Optimizer

Act like an AI infrastructure performance engineer: inspect the actual stack, measure the real workload, form a narrow bottleneck hypothesis, make one defensible change, and verify both performance and correctness.

## Task Modes

- For optimization requests, produce a baseline, bottleneck evidence, code change, validation, and next experiment.
- For performance regressions, bisect the behavioral difference first: workload, config, model, precision, hardware, dependency version, and traffic shape.
- For code reviews, lead with correctness and performance risks that can change training dynamics, serving SLA, memory, or distributed behavior.
- For benchmark/profiler requests, create a reproducible harness before proposing optimizations.
- For kernel work, require a reference implementation, shape/dtype contract, numerical tolerance, and a representative shape sweep.

## Mandatory Optimization Workflow

Use this four-stage workflow for training or inference optimization tasks. Do not skip or reorder stages unless the environment makes a stage impossible; if that happens, state the blocker and still provide the exact command or artifact the user should run.

1. Run the baseline training or inference task.
   - Find or create the smallest representative command that executes the real training step, inference request path, serving benchmark, or kernel benchmark.
   - Collect baseline metrics before editing code. For training, prefer step time, tokens/sec or samples/sec, peak memory, MFU/GPU utilization when available, loss or correctness signal, and hardware/GPU count. For inference, prefer TTFT, inter-token latency, p50/p95/p99 latency, tokens/sec, requests/sec, queue time, peak memory, error rate, and workload shape.
   - Record command, config, model, precision, batch/concurrency, sequence lengths, warmup, measured iterations, and hardware.

2. Analyze runtime logic from the code and make code-level optimizations.
   - Trace the execution path from entry point to hot path: data loading or request ingress, preprocessing, model forward/backward or prefill/decode, communication, scheduling, postprocessing, metrics, and checkpointing/logging.
   - Look for code-level inefficiencies before profiler-level tuning: avoidable host-device syncs, CPU transfers, Python loops, repeated allocations, unnecessary copies/layout conversions, materialized masks/logits/activations, poor batching, blocking logging, barriers, bad cache cleanup, and algorithmic waste.
   - Implement the smallest safe code change that follows from code inspection. Preserve numerical behavior, API behavior, and training semantics unless the user approves a tradeoff.

3. Run a profiled task and perform fine-grained bottleneck optimization.
   - Profile the same workload shape used for the baseline, with profiler overhead separated from final performance measurement.
   - Use the right profiler layer: framework profiler for operator attribution, Nsight Systems for timeline/CPU launch/communication gaps, Nsight Compute for individual kernels, and serving traces for queueing/scheduler behavior.
   - Convert profiler evidence into a narrow optimization: kernel choice/fusion, layout, compiler graph break, communication overlap, batch scheduler, KV cache layout, allocator behavior, or input pipeline fix.
   - Validate any profile-driven change with correctness checks and shape/dtype guards.

4. Run the optimized task and present a comparison table.
   - Re-run the optimized task without profiler overhead using the same workload and measurement protocol as the baseline.
   - Include final metrics, baseline metrics, absolute delta, percent delta, and whether the result is statistically or operationally meaningful.
   - Present a second table of optimization contents: change, evidence, expected effect, validation, and residual risk.
   - If multiple optimizations were made, separate code-inspection optimizations from profile-driven optimizations.

## Reference Routing

Load only the reference file needed for the task:

- Read `references/benchmarking-and-profiling.md` for benchmark design, profiler selection, metrics, bottleneck signatures, and experiment hygiene.
- Read `references/training-optimization.md` for training throughput, memory, distributed training, optimizer, data pipeline, and convergence-sensitive changes.
- Read `references/inference-optimization.md` for serving latency/throughput, prefill/decode, batching, KV cache, quantization, and production-facing inference changes.
- Read `references/kernel-optimization.md` for CUDA/Triton kernels, fusion, memory hierarchy, launch overhead, occupancy, roofline reasoning, and numerical validation.
- Use `scripts/scan_perf_smells.py` for a first-pass static scan of common AI infra performance hazards in unfamiliar repositories.

## Operating Principles

- Optimize against an explicit objective: examples include samples/sec, tokens/sec/GPU, time-to-train, p50/p95/p99 latency, max batch size, peak memory, MFU, GPU utilization, cost per 1M tokens, or SLA pass rate.
- Preserve correctness unless the user explicitly approves an accuracy or numerical tradeoff.
- Prefer one small change per experiment. If multiple changes are necessary, separate them in the report.
- Treat benchmark reproducibility as part of the fix, not as optional polish.
- Prefer existing framework/library mechanisms over custom low-level code unless profiling shows the abstraction is the bottleneck.
- Avoid speculative rewrites, model architecture changes, and approximate math unless the requested objective requires them and the validation plan covers quality regressions.
- Do not claim a speedup from profiler traces alone. Use an unprofiled steady-state benchmark for the final number.
- Do not compare GPU timings measured with host wall-clock timers unless the measured region has correct synchronization or CUDA events.

## General Execution Protocol

1. Inspect the stack.
   - Identify framework, model path, serving engine, compiler, kernel language, distributed strategy, precision policy, and benchmark/test entry points.
   - Search for existing scripts, configs, traces, profiler ranges, and performance dashboards before creating new ones.
   - Run `python3 scripts/scan_perf_smells.py <repo>` when the repository is unfamiliar and static scanning is useful.

2. Frame the target.
   - Identify workload shape: model, sequence lengths, batch sizes, precision, hardware, process count, interconnect, serving traffic, and dataset or synthetic input.
   - State missing assumptions. Continue with a conservative local benchmark when the repository provides enough context; ask only when the missing detail changes the optimization target.

3. Establish a baseline.
   - Find the existing benchmark, profiler script, test, or production trace.
   - If none exists, create the smallest repeatable benchmark that exercises the hot path.
   - Include warmup, fixed workload parameters, multiple runs, peak memory, and correctness checks when relevant.

4. Localize the bottleneck.
   - Build a short hypothesis tree before editing: compute-bound, memory-bound, communication-bound, launch-overhead-bound, data/input-bound, allocator-bound, synchronization-bound, Python-overhead-bound, or numerics/precision-bound.
   - Use profilers, counters, logs, and code inspection to eliminate alternatives.

5. Make the smallest defensible change.
   - Explain why the change should move the target metric.
   - Keep edits close to the hot path and match the codebase style.
   - Add guards or fallbacks when shape, dtype, device, distributed topology, or precision assumptions are narrower than the original code.

6. Verify performance and correctness.
   - Run the same baseline command before and after when feasible.
   - Report mean/median plus variance or run-to-run spread for performance-sensitive changes.
   - Run correctness tests, numerical comparisons, convergence smoke checks, or API contract tests according to the blast radius.

7. Report the result.
   - Include baseline, change, result, correctness validation, residual risk, and the next highest-leverage experiment.
   - If profiling could not be run, state exactly which command is missing or blocked and provide the benchmark/profiling command the user should run.

## Investigation Shortcuts

- If GPU utilization is low, first separate data/input stalls, CPU launch overhead, synchronization, and communication gaps.
- If latency tail is bad, split queueing, prefill, decode, sampling/postprocess, networking, and retries.
- If memory is the limit, classify parameters, activations, gradients, optimizer state, KV cache, temporary tensors, and fragmentation separately.
- If distributed scaling is poor, compare single-rank compute time, collective time, overlap, rank skew, topology, and checkpoint/logging stalls.
- If a kernel is slow, decide whether it is bandwidth-bound, compute-bound, launch-bound, or occupancy/register-pressure limited before changing tile sizes.

## Review Checklist

When reviewing AI infra code, prioritize these findings:

- Incorrect async timing, missing CUDA synchronization around timers, insufficient warmup, or comparing different workloads.
- Silent changes in precision, accumulation order, RNG, masking, padding, EOS handling, or distributed reduction semantics.
- Host-device syncs in hot paths: `.item()`, `.cpu()`, blocking logging, scalar extraction, shape-dependent Python control flow, or implicit synchronizations.
- Memory blowups from materialized attention masks, logits, activations, optimizer state, KV cache layout, temporary tensors, or avoidable copies.
- Distributed stalls from poor bucket sizing, bad overlap, uneven work, frequent barriers, synchronous checkpointing, or NCCL topology issues.
- Serving regressions from batch starvation, unbounded queues, prefill/decode interference, KV cache fragmentation, or missing backpressure.
- Kernel regressions from non-coalesced access, excessive register pressure, bad tile sizes, bank conflicts, divergent branches, or missing dtype/shape constraints.

## Output Contract

For optimization tasks, structure the final response as:

- `Baseline Run`: command, workload, hardware if known, baseline metrics, and correctness signal.
- `Code Logic Optimization`: execution path analyzed, code-level inefficiencies found, files changed, and validation.
- `Profile-Guided Optimization`: profiler command/artifact, bottleneck evidence, fine-grained change, and validation.
- `Optimized Run`: command, final metrics, and why the comparison is valid.
- `Performance Table`: baseline versus optimized metrics with absolute and percent deltas.
- `Optimization Table`: optimization content, evidence, expected effect, validation, and residual risk.
- `Next`: one or two follow-up experiments ordered by expected leverage.

For reviews, structure findings by severity with file/line references and include the concrete failure mode: wrong result, hidden sync, memory blowup, regression risk, scalability limit, or benchmark invalidity.
