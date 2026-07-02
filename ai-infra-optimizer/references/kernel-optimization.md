# Kernel Optimization

## Table Of Contents

- Decide whether custom kernels are justified
- Kernel contract
- Roofline reasoning
- CUDA and Triton checklist
- Fusion
- Autotuning
- Numerical validation

## Decide Whether Custom Kernels Are Justified

- Use library kernels, framework fused ops, or compiler fusion first unless profiling identifies a specific unresolved bottleneck.
- Custom kernels are most justified for repeated hot shapes, avoidable tensor materialization, unusual layouts, or operations not well covered by vendor libraries.
- Keep a reference implementation for correctness and shape coverage.

## Kernel Contract

Before editing or adding a kernel, define:

- Supported shapes, dtypes, strides/layouts, devices, and alignment assumptions.
- Forward-only or forward/backward coverage.
- Numerical tolerance versus the reference.
- Fallback path for unsupported shapes.
- Representative shape sweep and adversarial edge cases.
- Whether the kernel is intended for training, inference prefill, inference decode, or offline batch inference.

## Roofline Reasoning

- Estimate arithmetic intensity before optimizing.
- If memory-bound, reduce bytes moved, improve locality, coalesce access, and avoid intermediate writes.
- If compute-bound, improve tensor-core use, tile sizes, instruction mix, and occupancy without increasing memory traffic enough to lose the gain.
- If launch-bound, fuse operations, use persistent kernels where appropriate, or graph-capture stable sequences.
- Do not chase occupancy alone. A lower-occupancy kernel can win if it reduces memory traffic or uses tensor cores better.

## CUDA And Triton Checklist

- Check coalesced global memory access and alignment.
- Avoid unnecessary `contiguous()` calls by supporting the layout or moving layout conversion outside the hot path.
- Manage register pressure, shared memory use, spills, and occupancy together.
- Watch for divergent branches, bank conflicts, non-vectorized memory ops, and scalar loops.
- Verify dtype support and accumulation dtype explicitly.
- Handle edge shapes, masks, strides, and non-power-of-two dimensions deliberately.
- In Triton, inspect generated code when performance is surprising and keep autotune spaces bounded.
- For reductions, check accumulation order, determinism requirements, atomics, and precision.
- For attention-like kernels, validate masks, causal boundaries, sequence lengths, block tables, and partial tiles.
- For decode kernels, optimize for small token-step behavior instead of assuming prefill-like shapes.

## Fusion

- Fuse when it removes materialization or launch overhead without making the kernel too register-heavy.
- Do not fuse across boundaries that harm reuse, complicate autograd, or hide useful library kernels.
- Validate backward pass separately when changing training kernels.
- Keep fusion benefits honest by measuring both launch count and memory traffic, not only single-kernel runtime.

## Autotuning

- Sweep representative shapes, not just the current benchmark shape.
- Cache or bound autotuning so startup latency is acceptable for the workload.
- Report shape coverage and fallback behavior.
- Keep compile/autotune time separate from steady-state runtime in serving contexts.

## Numerical Validation

- Compare against a simple reference over dtype and shape sweeps.
- Use tolerances appropriate for dtype, operation order, and accumulation strategy.
- Include adversarial shapes: small, large, non-contiguous, odd sizes, boundary masks, empty or single-token cases where relevant.
- Check gradients for training kernels with finite differences or framework gradcheck when feasible.
- Test NaN/Inf propagation and masked-out values when the operation has masking or normalization semantics.
