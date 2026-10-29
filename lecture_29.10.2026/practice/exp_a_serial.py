"""Experiment A — The Stubborn Serial Tail.

Parallel work is clean, but the master process sleeps 0.40 s before dispatch.
Expectation: Karp-Flatt e(p) stays ~constant across p (fixed Amdahl serial path).
"""
import time
import math
from multiprocessing import Pool

REPEATS = 3  


def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))


def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    t0 = time.perf_counter()
    time.sleep(0.40)  
    with Pool(processes=workers) as pool:
        pool.map(heavy_math, [chunk_size] * total_chunks)
    return time.perf_counter() - t0


def karp_flatt(t1, tp, p):
    return ((p * tp / t1) - 1.0) / (p - 1.0)


def best_of(workers):
    return min(run_experiment(workers) for _ in range(REPEATS))


if __name__ == "__main__":
    import os
    print(f"Execution Cores: {os.cpu_count()}")
    runtimes = {p: best_of(p) for p in (1, 2, 4)}
    t1 = runtimes[1]
    print(f"P=1 | Runtime={t1:.3f}s | serial fraction 0.40/T1 = {0.40 / t1:.3f}")
    for p in (2, 4):
        tp = runtimes[p]
        sp = t1 / tp
        e = karp_flatt(t1, tp, p)
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | KarpFlatt e={e:.3f}")
    ratio = karp_flatt(t1, runtimes[4], 4) / karp_flatt(t1, runtimes[2], 2)
    print(f"Rubric check: e(4)/e(2) = {ratio:.2f}  (need 0.80..1.25)")
