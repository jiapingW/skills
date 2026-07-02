# Inference Optimization

## Table Of Contents

- First-pass triage
- Decision tree
- Prefill and decode
- Batching and scheduling
- KV cache
- Quantization and accuracy
- Serving system
- Correctness and rollout

## First-Pass Triage

- Separate model compute from request queueing, tokenization, detokenization, network time, postprocessing, and logging.
- For LLMs, split prefill from decode. Use TTFT for prefill-facing latency and inter-token latency for decode.
- Decide whether the goal is lower tail latency, higher throughput, lower cost, larger context, more concurrency, or smaller memory.
- Capture prompt length distribution, output length distribution, concurrency, request rate, sampling params, model revision, precision, GPU count, and cache warm state.

## Decision Tree

- If TTFT is high, inspect queue time, tokenization, prefill batching, attention kernel, context length, prefix cache hit rate, and tensor-parallel collectives.
- If inter-token latency is high, inspect decode batch occupancy, KV cache layout, memory bandwidth, sampling/logits processing, launch overhead, and synchronization.
- If throughput is low but latency is acceptable, inspect batching policy, admission control, max batch tokens, CPU scheduler overhead, and GPU occupancy.
- If p99 is bad but average is good, inspect queueing, long-prompt interference, memory pressure, request cancellation, retries, and lock contention.
- If OOM or fragmentation appears under load, inspect KV cache allocator, eviction, max context policy, beam count, and per-request cleanup.

## Prefill And Decode

- Prefill is often compute-heavy and benefits from efficient attention, larger batches, and tensor-core-friendly shapes.
- Decode is often memory-bandwidth and launch-overhead sensitive because batch-size and token-per-step behavior differ from prefill.
- Avoid evaluating only average tokens/sec when the SLA depends on TTFT or p99 latency.
- Check sampling, logits processors, beam search, and stop-token handling for CPU or synchronization overhead.
- Benchmark prefill-only, decode-only, and mixed traffic separately when changing scheduler or cache code.

## Batching And Scheduling

- Inspect continuous batching, max batch tokens, admission control, queue timeout, fairness, and starvation behavior.
- Avoid increasing throughput by hiding unacceptable queue latency.
- Use traffic-shaped benchmarks with prompt/output length distributions, concurrency, and request rate.
- Separate offline batch inference from online serving; they optimize for different objectives.
- Track batch occupancy over time, not only aggregate requests/sec.
- Preserve cancellation and timeout cleanup; leaked cache blocks turn into delayed production failures.

## KV Cache

- Track KV memory per request as a function of layers, heads, head dimension, dtype, context length, and beam count.
- Watch for fragmentation, eviction policy, prefix sharing, paged attention metadata overhead, and cache compaction cost.
- Validate cache correctness across prefill/decode boundaries, sequence packing, prefix cache hits, and request cancellation.
- Consider lower-precision KV cache only with quality checks and model/task-specific tolerance.
- Estimate memory as `2 * layers * kv_heads * head_dim * tokens * dtype_bytes * batch * beams`, plus allocator and metadata overhead.
- For MQA/GQA models, use KV heads rather than query heads in memory estimates.
- Validate cache indexing after prefix sharing, chunked prefill, speculative decoding, and paged/block allocation changes.

## Quantization And Accuracy

- Treat quantization as a quality/performance tradeoff, not a free optimization.
- Specify weight, activation, and KV cache precision separately.
- Validate with task metrics, golden prompts, perplexity where appropriate, and regression tests for edge prompts.
- Watch for calibration data mismatch, outlier channels, small-batch latency regressions, and unsupported kernels.
- Distinguish memory savings from latency wins; quantized paths can be slower for small batches or unsupported shapes.

## Serving System

- Look for CPU bottlenecks in tokenization, detokenization, JSON handling, scheduler locks, logging, metrics, and networking.
- Add backpressure for overload rather than allowing unbounded queue growth.
- Measure memory headroom, request cancellation cleanup, timeout behavior, and error handling under load.
- For multi-GPU serving, check tensor-parallel collectives, rank skew, load balancing, and placement.
- Instrument queue time, scheduler time, model time, postprocess time, and response write time separately.
- Use open-loop load tests for capacity/SLA work and closed-loop tests only when modeling client-perceived behavior.

## Correctness And Rollout

- Preserve API contract, streaming behavior, EOS handling, sampling determinism when seeded, and error semantics.
- Compare outputs on golden prompts before and after performance changes.
- For approximate changes, report expected quality risk and provide an A/B or canary plan when production traffic is involved.
