# Training Optimization

## Table Of Contents

- First-pass triage
- Decision tree
- Data pipeline
- Forward/backward compute
- Memory
- Distributed training
- Optimizer and precision
- Compiler and graph capture
- Checkpointing
- Correctness and convergence

## First-Pass Triage

- Start from the training step breakdown: data wait, forward, loss, backward, optimizer, communication, checkpointing, logging.
- Measure tokens/sec or samples/sec alongside peak memory and loss behavior.
- Identify whether the target is per-step speed, larger batch/sequence length, time-to-train, cost, or stability.
- Record global batch size, microbatch size, gradient accumulation, sequence length policy, precision, number of GPUs, and parallelism strategy.

## Decision Tree

- If GPU has large idle gaps before forward, inspect data loading, tokenization, host-to-device copies, and CPU scheduling first.
- If forward/backward dominates and tensor cores are underused, inspect dtype, dimensions, layout, attention implementation, and compiler/fusion settings.
- If backward or optimizer dominates, inspect gradient accumulation, fused optimizer availability, sharding, foreach paths, and unnecessary gradient materialization.
- If peak memory blocks batch size, separate activation memory from optimizer state, parameters, gradients, temporary tensors, and fragmentation.
- If multi-GPU scaling is poor, compare single-GPU step time, collective time, overlap, rank skew, and checkpoint/logging barriers.
- If loss changes after a performance patch, inspect precision, reduction order, masking, padding, sequence packing, RNG, dropout, and optimizer semantics before accepting the speedup.

## Data Pipeline

- Look for GPU idle time before forward.
- Check tokenization, decode, augmentation, collation, padding, host-to-device copies, worker count, pinned memory, prefetching, and caching.
- Avoid optimizing model code if the GPU is starved by input.
- Prefer bucketing/packing sequence lengths when padding waste dominates.
- For language models, measure real tokens/sec, not just samples/sec, when sequence length varies.
- For packing changes, validate labels, attention masks, position IDs, EOS handling, and loss normalization.

## Forward/Backward Compute

- Prefer built-in optimized attention, fused norms, fused optimizer paths, compiled graphs, and library kernels before custom code.
- Check tensor-core eligibility: dtype, layout, alignment, matmul dimensions, autocast, and accumulation dtype.
- Remove avoidable tensor materialization, transposes, contiguous calls, masking expansions, and duplicate computation.
- Watch for Python loops over layers, tokens, heads, experts, or microbatches in hot paths.
- Verify that flash/SDPA attention selection actually happens for the target dtype, mask type, head dimension, dropout, and causal setting.
- Check whether activation checkpointing boundaries reduce memory without recomputing expensive non-deterministic or communication-heavy work.

## Memory

- Separate activation memory, optimizer state, gradients, parameters, temporary tensors, and fragmentation.
- Consider activation checkpointing when memory blocks batch size or sequence length, but measure recompute overhead.
- Prefer in-place or fused operations only when autograd semantics remain correct.
- Inspect attention masks, logits, intermediate hidden states, and saved activations for accidental full-size materialization.
- For long context training, prioritize sequence packing, flash attention variants, activation recomputation policy, and parallelism strategy.
- Beware hidden copies from dtype conversions, layout conversions, tensor slicing followed by contiguous operations, and CPU offload boundaries.
- Treat `empty_cache()` in the training loop as a smell unless there is a measured fragmentation problem and no better allocator fix.

## Distributed Training

- Map the parallelism strategy: data, tensor, pipeline, sequence, expert, sharding, ZeRO/FSDP, or hybrid.
- Check rank balance, microbatch schedule, gradient accumulation, bucket sizes, overlap of compute/communication, and collective frequency.
- Use timeline profiling to distinguish NCCL time from idle gaps caused by late ranks.
- Treat barriers, synchronous logging, validation, and checkpoint writes as possible cluster-wide stalls.
- Verify topology assumptions: NVLink, PCIe, InfiniBand, RoCE, NUMA, NIC binding, and process placement.
- For FSDP/ZeRO, inspect all-gather timing, reduce-scatter timing, prefetch policy, bucket sizes, parameter freezing, and state-dict checkpoint mode.
- For pipeline parallelism, inspect bubble fraction, microbatch count, inter-stage imbalance, activation transfer, and schedule.
- For MoE, inspect expert imbalance, token dispatch, all-to-all time, capacity factor, dropped tokens, and routing correctness.

## Optimizer And Precision

- Validate mixed precision settings, loss scaling, gradient clipping, accumulation dtype, and optimizer state dtype.
- Use fused or foreach optimizer implementations when available and correct.
- Consider optimizer state sharding or lower-precision optimizer states when memory is the limit.
- Never change precision policy without checking loss parity or convergence-sensitive metrics.
- Keep master weights, gradient scaling, clipping order, and weight decay semantics explicit when replacing optimizers.

## Compiler And Graph Capture

- Use `torch.compile`, CUDA graphs, XLA compilation, or framework-native graph capture when workload shapes are stable.
- Guard dynamic-shape paths and fall back cleanly when graph assumptions are not met.
- Measure compile time separately from steady-state runtime.
- Watch for graph breaks caused by Python side effects, shape-dependent control flow, tensor-to-scalar reads, logging, and unsupported ops.
- Check recompilation counts for dynamic shapes before concluding a compiler path helps.

## Checkpointing

- Measure checkpoint save/load time and whether it blocks training.
- Prefer async or distributed checkpointing only when correctness, restartability, and storage limits are validated.
- Include optimizer, scheduler, RNG, dataloader state, and distributed metadata when restart fidelity matters.

## Correctness And Convergence

- Compare loss for a small fixed batch before and after.
- Check gradient norms or parameter deltas for sensitive training-loop changes.
- Run a short convergence smoke test when changing precision, optimizer, loss, masking, sequence packing, or distributed reductions.
- Preserve RNG behavior when reproducibility is part of the contract.
