# Benchmarking And Profiling

## Table Of Contents

- Repo reconnaissance
- End-to-end optimization runbook
- Measurement hygiene
- Metric selection
- Profiler selection
- Profiler recipes
- Bottleneck signatures
- Benchmark templates
- Reporting format

## Repo Reconnaissance

Use a quick static pass before opening many files:

```bash
rg -n "torch\\.profiler|jax\\.profiler|nvtx|nsys|ncu|benchmark|tokens/sec|samples/sec|cuda\\.synchronize|\\.item\\(|\\.cpu\\(|dist\\.barrier|all_reduce|all_gather" .
python3 scripts/scan_perf_smells.py .
```

Then identify:

- Benchmark entry points and workload configs.
- Profiler ranges or tracing utilities.
- Framework and compiler switches.
- Precision policy and deterministic settings.
- Distributed launcher and rank/world-size assumptions.
- Serving load generator or production trace format.

## End-To-End Optimization Runbook

Use this as the default execution sequence for training and inference optimization:

1. Baseline run.
   - Execute the representative training or inference task before editing code.
   - Save command, config, workload shape, hardware, warmup/measured iterations, metrics, and correctness signal.

2. Code-logic optimization.
   - Trace the runtime path from entry point to hot path.
   - Optimize obvious code-level waste first: syncs, CPU transfers, repeated allocations, copies, Python loops, padding waste, blocking logs, barriers, scheduler mistakes, or cache cleanup.
   - Run correctness checks after the code-level change.

3. Profile-guided optimization.
   - Run profiler on the same workload shape.
   - Attribute time to operators, kernels, communication, CPU launch gaps, input stalls, queueing, or allocator behavior.
   - Implement a fine-grained optimization only when profiler evidence supports it.

4. Optimized run and report.
   - Re-run without profiler overhead using the same measurement protocol as the baseline.
   - Present metric deltas and optimization contents in tables.

Performance comparison table:

| Metric | Baseline | Optimized | Abs Delta | Delta % | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| step time or latency | 0.000 | 0.000 | 0.000 | 0.0% | same workload |
| tokens/sec or req/sec | 0.000 | 0.000 | 0.000 | 0.0% | higher is better |
| peak memory | 0.000 | 0.000 | 0.000 | 0.0% | lower is better |

Optimization contents table:

| Stage | Change | Evidence | Expected Effect | Validation | Risk |
| --- | --- | --- | --- | --- | --- |
| Code logic | change summary | code path evidence | fewer syncs/copies/etc. | test/metric | residual risk |
| Profile-guided | change summary | profiler evidence | less kernel/comm/queue time | test/metric | residual risk |

## Measurement Hygiene

- Pin workload shape: model revision, batch size, sequence length distribution, precision, tensor parallelism, pipeline parallelism, data parallelism, beam/sampling settings, and hardware.
- Separate cold start, warmup, steady state, and teardown.
- Synchronize GPU timing explicitly when measuring from host code.
- Avoid mixing profiler overhead with throughput results. Run profiled and unprofiled measurements separately.
- Prefer several short steady-state runs over one heroic run. Report variance when differences are small.
- Use synthetic data only when the production input pipeline is not part of the target.
- Keep clocks, power settings, and GPU sharing in mind. Note them when results are noisy.
- Compare identical correctness and quality settings: precision, quantization, dropout, sampling, caching, padding, masking, and deterministic flags.
- Do not use profiler runtimes as final speedup numbers. Profilers change scheduling and add overhead.
- For small deltas below 3-5%, require more repetitions or stronger profiler evidence before presenting the change as a win.
- Record exact command, git diff, dependency versions when relevant, environment variables, GPU model, GPU count, and driver/runtime if available.

## Metric Selection

- Training: tokens/sec, samples/sec, step time, MFU, TFLOP/s, peak memory, time-to-loss, checkpoint time, data wait time, communication time.
- Inference: p50/p95/p99 latency, time-to-first-token, inter-token latency, tokens/sec, requests/sec, queue time, batch occupancy, KV cache utilization, SLA pass rate.
- Kernels: runtime, achieved bandwidth, achieved FLOP/s, occupancy, register count, shared memory use, spills, memory transactions, launch count.
- Cost: GPU-hours, cost per 1M training tokens, cost per 1M generated tokens, energy if available.

## Profiler Selection

- Use framework profilers first for operator-level attribution: `torch.profiler`, TensorBoard traces, XLA/JAX profiles, or framework-native tracing.
- Use Nsight Systems for timeline questions: CPU launch overhead, GPU gaps, communication overlap, data loader stalls, stream synchronization, and NCCL timelines.
- Use Nsight Compute for one kernel at a time: memory coalescing, occupancy, stalls, register pressure, shared memory behavior, and achieved roofline position.
- Use NVTX ranges around model phases: data load, forward, loss, backward, optimizer, all-reduce, checkpoint, prefill, decode, sampling, and postprocess.
- Use production traces or request logs for serving queueing, batching, SLA, and traffic-shape problems.

## Profiler Recipes

Use these as starting points and adapt to the repo:

```bash
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o trace --force-overwrite true <command>
ncu --set full --target-processes all -o kernel_report <command>
```

For PyTorch timing inside a benchmark:

- Use CUDA events for GPU regions or synchronize immediately before and after host timing.
- Reset peak memory before measured steps with `torch.cuda.reset_peak_memory_stats()`.
- Report `torch.cuda.max_memory_allocated()` and, when fragmentation matters, reserved memory.

For distributed runs:

- Capture per-rank timing when possible.
- Avoid timing only rank 0 unless the metric is intentionally rank-0 work.
- Add barriers only around measurement boundaries; barriers inside hot paths can change the workload.

## Bottleneck Signatures

- Compute-bound: high tensor-core utilization, high arithmetic intensity, little idle time, low memory stall impact.
- Memory-bandwidth-bound: high HBM throughput, low arithmetic intensity, runtime dominated by reads/writes or tensor materialization.
- Launch-overhead-bound: many tiny kernels, GPU gaps, high CPU framework overhead, poor fusion.
- Communication-bound: NCCL/all-reduce/all-gather dominates, poor overlap, rank skew, topology-sensitive variance.
- Input-bound: GPU idle before forward, high data loader time, CPU decode/augmentation/tokenization in the critical path.
- Synchronization-bound: frequent barriers, host reads from device tensors, logging or metrics forcing `.item()` in the hot path.
- Allocator-bound: repeated large allocations, fragmentation, cache misses, unpredictable peak memory.
- Scheduler-bound serving: high queue time, low batch occupancy, poor request admission, starvation between long and short requests.

## Benchmark Templates

For training changes, define:

- Exact command and config.
- Number of warmup steps and measured steps.
- Batch size, gradient accumulation, sequence length, model size, precision, number of GPUs.
- Metrics: step time, tokens/sec, peak memory, loss equality or tolerance.

For inference changes, define:

- Prompt length distribution, output length distribution, concurrency, request rate, sampling parameters, and cache state.
- Metrics: TTFT, ITL, end-to-end latency percentiles, tokens/sec, queue time, error rate, memory.

For kernel changes, define:

- Shape sweep, dtype sweep, device, warmup, iterations, reference implementation, tolerance, and profiler command.

Benchmark failure modes to catch:

- Different sequence length distribution after a data or packing change.
- Dropout, sampling, or RNG differences hiding correctness regressions.
- A batch-size increase counted as throughput improvement without memory/SLA comparison.
- Benchmark includes first-time compilation, CUDA graph capture, autotuning, or cache warmup in steady-state timing.
- Metric is averaged across failed, timed-out, or cancelled requests.

## Reporting Format

Use a compact table when possible:

| Case | Before | After | Delta | Notes |
| --- | ---: | ---: | ---: | --- |
| step time | 0.000s | 0.000s | 0.0% | workload details |

State when measurement is not reliable, especially for small deltas, shared hardware, profiler overhead, or mismatched inputs.
