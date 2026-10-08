# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103077<br>
**Allocated GPU Node:** Tesla T4<br>
**CUDA Compute Capability:** 7.5<br>
**Official Verification Token:** C76A500563EE23A488D7

## Task 1: Warp divergence benchmark (N = 2^20, 1,000 iterations/element, 256 threads/block, mean of 10 trials)

| Kernel | Mean time (ms) | Std (ms) | Slowdown vs A |
|---|---|---|---|
| A: Uniform | 20.9102 | 5.3690 | 1.00x |
| B: Full divergence (interleaved) | 95.3065 | 3.1497 | 4.56x |
| C: Warp-aligned branching | 46.7801 | 0.0261 | 2.24x |

**Analysis.** Kernel A is the baseline: every thread in a warp follows the same instruction stream.
Kernel B (`idx % 2`) splits every warp into two groups, so the hardware runs Path 1 and Path 2 one after
the other with half of the lanes masked each time. Measured: B took 4.56x the time of A.
Kernel C branches on `idx // 32`, which is constant inside a warp, so every warp follows a single path and nothing is
serialized. Measured: C took 2.24x the time of A. Kernels B and C do the same total work; only the
mapping of branches to warps differs.

## Task 2: 1D stencil with halo replication
`TASK 2 PASSED: MAX DELTA = 5.960e-08` (N = 100,007). Edge threads clamp the missing neighbour to `d_in[0]` / `d_in[N-1]`,
matching `np.pad(mode='edge')`.

## Task 3: Grid-stride loop
All 2^24 = 16,777,216 elements equal the factor using only 64 x 256 = 16,384 threads. Each thread handles
elements `start, start + stride, ...`.

## Task 4: Sobel-X
2048 x 2048 image, blocks of 16x16, grid 128 x 128. Border pixels are 0.0; interior pixels use the Kx kernel.

## Verification
`python verify_submission.py` passes all checks. Token: **C76A500563EE23A488D7**

