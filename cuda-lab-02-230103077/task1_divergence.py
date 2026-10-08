import time
import numpy as np
from numba import cuda

N = 2 ** 20
ITERS = 1000
TPB = 256
TRIALS = 10


@cuda.jit
def kernel_a_uniform(y):
    idx = cuda.grid(1)
    if idx < y.shape[0]:
        v = y[idx]
        for _ in range(ITERS):
            v = v * 1.0001 + 0.0001
        y[idx] = v


@cuda.jit
def kernel_b_interleaved(y):
    idx = cuda.grid(1)
    if idx < y.shape[0]:
        v = y[idx]
        if idx % 2 == 0:
            for _ in range(ITERS):          
                v = v * 1.0001 + 0.0001
        else:
            for _ in range(ITERS):         
                v = (v - 0.0001) / 1.0001
        y[idx] = v


@cuda.jit
def kernel_c_warp_aligned(y):
    idx = cuda.grid(1)
    if idx < y.shape[0]:
        v = y[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(ITERS):          
                v = v * 1.0001 + 0.0001
        else:
            for _ in range(ITERS):          
                v = (v - 0.0001) / 1.0001
        y[idx] = v


def bench(kernel, d_y):
    bpg = (N + TPB - 1) // TPB
    kernel[bpg, TPB](d_y)                   
    cuda.synchronize()
    times = []
    for _ in range(TRIALS):
        cuda.synchronize()
        t0 = time.perf_counter()
        kernel[bpg, TPB](d_y)
        cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def main():
    np.random.seed(0)
    h_y = np.random.rand(N).astype(np.float32)
    d_y = cuda.to_device(h_y)              
    results = []
    for name, k in [("A: Uniform", kernel_a_uniform),
                    ("B: Full divergence (interleaved)", kernel_b_interleaved),
                    ("C: Warp-aligned branching", kernel_c_warp_aligned)]:
        d_y.copy_to_device(h_y)
        results.append((name, *bench(k, d_y)))
    base = results[0][1]
    lines = ["| Kernel | Mean time (ms) | Std (ms) | Slowdown vs A |",
             "|---|---|---|---|"]
    for name, m, s in results:
        lines.append(f"| {name} | {m:.4f} | {s:.4f} | {m / base:.2f}x |")
    table = "\n".join(lines)
    print(table)
    with open("task1_results.md", "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
