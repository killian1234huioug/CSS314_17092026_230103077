"""Experiment B — The Contention Traffic Jam.

Every worker increments ONE shared integer under ONE lock on every iteration.
Expectation: negative speedup (T4 > T1) and a surging Karp-Flatt e(p).
"""
import time
from multiprocessing import Process, Value, Lock

REPEATS = 3


def contended_worker(shared_counter, lock, iterations):
    for _ in range(iterations):
        with lock: 
            shared_counter.value += 1


def run_contended(p, total_increments=2_000_000):
    increments_per_worker = total_increments // p
    counter = Value('i', 0, lock=False)  
    lock = Lock()
    procs = [
        Process(target=contended_worker, args=(counter, lock, increments_per_worker))
        for _ in range(p)
    ]
    t0 = time.perf_counter()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    t1 = time.perf_counter()
    assert counter.value == total_increments
    return t1 - t0


def karp_flatt(t1, tp, p):
    return ((p * tp / t1) - 1.0) / (p - 1.0)


if __name__ == "__main__":
    import os
    print(f"Execution Cores: {os.cpu_count()}")
    cores = [1, 2, 4]
    runtimes = {p: min(run_contended(p) for _ in range(REPEATS)) for p in cores}
    t1 = runtimes[1]
    print(f"P=1 | Runtime={t1:.3f}s")
    for p in (2, 4):
        tp = runtimes[p]
        sp = t1 / tp
        e = karp_flatt(t1, tp, p)
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | KarpFlatt e={e:.3f}")
    print(f"Rubric check: T4 > T1 -> {runtimes[4] > t1}")
