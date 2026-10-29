"""Experiment C — The Privatization Fix.

Each worker accumulates into a private local variable (zero shared state);
results are combined with ONE reduction at the end.
Expectation: near-linear speedup, e(p) collapses toward ~0.
"""
import time
from multiprocessing import Pool

REPEATS = 3


def private_worker(iterations):
    local_sum = 0  
    for _ in range(iterations):
        local_sum += 1
    return local_sum


def run_privatized(p, total_increments=20_000_000):
    increments_per_worker = total_increments // p
    tasks = [increments_per_worker] * p
    t0 = time.perf_counter()
    with Pool(processes=p) as pool:
        results = pool.map(private_worker, tasks)
    total = sum(results)  
    t1 = time.perf_counter()
    assert total == total_increments
    return t1 - t0


def karp_flatt(t1, tp, p):
    return ((p * tp / t1) - 1.0) / (p - 1.0)


if __name__ == "__main__":
    import os
    print(f"Execution Cores: {os.cpu_count()}")
    runtimes = {p: min(run_privatized(p) for _ in range(REPEATS)) for p in (1, 2, 4)}
    t1 = runtimes[1]
    print(f"P=1 | Runtime={t1:.3f}s")
    for p in (2, 4):
        tp = runtimes[p]
        sp = t1 / tp
        e = karp_flatt(t1, tp, p)
        eff = sp / p * 100
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | Efficiency={eff:.0f}% | KarpFlatt e={e:.3f}")
    sp4 = t1 / runtimes[4]
    print(f"Rubric check: T4 < T1 -> {runtimes[4] < t1}; E4 = {sp4/4*100:.0f}% (need > 70%)")
