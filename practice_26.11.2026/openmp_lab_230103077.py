"""
OpenMP Multithreading & Performance Engineering in Python
Student: Kraman Zhaniya
Student ID: 230103077

Run locally:
    python3 -m pip install numpy numba
    python3 openmp_lab_230103077.py

IMPORTANT:
This lab explicitly requires Numba's native OpenMP threading layer ("omp").
The script checks that requirement and stops rather than silently using another
threading backend. Numba's OpenMP backend is not generally available in the
standard macOS Numba build; see the notes printed if the check fails.
"""

import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"

import time
import hashlib
import json
import math
import platform
import sys

import numpy as np
import numba as nb
from numba import njit, prange
from numba import (
    set_num_threads,
    get_num_threads,
    get_thread_id,
)

STUDENT_NAME = "Kraman Zhaniya"
STUDENT_ID = 230103077

assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0, "Set a valid numeric STUDENT_ID."
assert (os.cpu_count() or 1) >= 4, "A local CPU with at least 4 logical cores is required."

@njit(parallel=True)
def _omp_probe():
    acc = 0
    for i in prange(64):
        acc += get_thread_id() + 1
    return acc

_ = _omp_probe()
_actual_layer = nb.threading_layer()
assert _actual_layer == "omp", (
    f"Expected Numba threading layer 'omp', got '{_actual_layer}'. "
    "This run does NOT satisfy the lab's OpenMP requirement. "
    "On macOS, the standard Numba distribution may not provide the OpenMP "
    "backend. Do not replace it with 'workqueue' or 'tbb' and call it OpenMP."
)

rng = np.random.default_rng(STUDENT_ID)

N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024
BINS_T1 = 48 + (STUDENT_ID % 17)
ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8  
ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD = 37 + (STUDENT_ID % 19)
GRID_H = 384 + (STUDENT_ID % 4) * 32
GRID_W = 512 + (STUDENT_ID % 5) * 32
OMEGA_SOR = np.float64(1.72 + (STUDENT_ID % 11) * 0.015)

THREAD_COUNTS = (1, 2, 4, 8)
MAX_THREADS = min(8, os.cpu_count() or 1)
assert MAX_THREADS >= 4

print(f"Student: {STUDENT_NAME} | ID: {STUDENT_ID}")
print(f"Platform: {platform.platform()} | Python: {sys.version.split()[0]}")
print(f"Backend: {nb.threading_layer()} | Logical cores: {os.cpu_count()}")
print(f"T1 N={N_T1:,}, bins={BINS_T1}; T2 iterations={ITERS_T2:,}, pad stride={PAD_STRIDE}")
print(f"T3 rows={ROWS_T3:,}, skew mod={SKEW_MOD}; T4 grid={GRID_H}x{GRID_W}, omega={OMEGA_SOR}")


@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    """Intentionally unsafe shared read-modify-write histogram."""
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in prange(indices.shape[0]):
        b = indices[i]
        hist[b] += 1
        w_sums[b] += weights[i]
    return hist, w_sums


@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    """Thread-private buffers, followed by a deterministic final reduction."""
    nthreads = get_num_threads()
    private_hist = np.zeros((nthreads, num_bins), dtype=np.int64)
    private_wsum = np.zeros((nthreads, num_bins), dtype=np.float64)
    worker_hits = np.zeros(nthreads, dtype=np.int64)

    for i in prange(indices.shape[0]):
        tid = get_thread_id()
        b = indices[i]
        private_hist[tid, b] += 1
        private_wsum[tid, b] += weights[i]
        worker_hits[tid] += 1

    hist = np.zeros(num_bins, dtype=np.int64)
    for tid in range(nthreads):
        for b in range(num_bins):
            hist[b] += private_hist[tid, b]


    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in range(indices.shape[0]):
        b = indices[i]
        w_sums[b] += weights[i]

    return hist, w_sums, worker_hits, private_wsum


print("\n=== TASK 1: HISTOGRAM RACE ===")
data_t1 = rng.integers(0, BINS_T1, size=N_T1, dtype=np.int32)
weights_t1 = rng.uniform(0.25, 2.50, size=N_T1).astype(np.float64)

set_num_threads(1)
hist_ref, wsums_ref, _, _ = safe_hist_omp(data_t1, weights_t1, BINS_T1)

set_num_threads(8)
lost_counts = []
max_discrepancies = []
for run in range(10):
    bad_hist, bad_wsums = buggy_hist_omp(data_t1, weights_t1, BINS_T1)
    lost_counts.append(int(N_T1 - np.sum(bad_hist)))
    max_discrepancies.append(int(np.max(np.abs(bad_hist - hist_ref))))
    print(f"Buggy run {run + 1:02d}: lost counts={lost_counts[-1]:,}, "
          f"max bin discrepancy={max_discrepancies[-1]:,}")

safe_results = {}
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    h, w, hits, _ = safe_hist_omp(data_t1, weights_t1, BINS_T1)
    assert np.array_equal(h, hist_ref), f"Task 1 integer histogram mismatch at {nt} threads."
    assert np.array_equal(w, wsums_ref), f"Task 1 weighted sums are not bit-identical at {nt} threads."
    active = int(np.count_nonzero(hits))
    assert active == nt, f"Expected {nt} participating workers, observed {active}."
    safe_results[nt] = {"active_workers": active, "worker_hits": hits.tolist()}
    print(f"Safe histogram at {nt} threads: PASS; active workers={active}")

DIAGNOSIS_T1 = (
    "The buggy kernel has a data race because multiple threads can read the same bin, "
    "increment it, and write back competing values, so updates are lost. Thread-private "
    "histogram rows give each worker exclusive write access and remove this race. Integer "
    "counts can then be merged exactly after the parallel region. Floating-point addition "
    "is not associative, so changing the order or grouping of additions can change the last "
    "bits even when integer increments are atomic. The implementation therefore finalizes "
    "weighted sums in the original input order to guarantee identical float64 results."
)


@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    nthreads = get_num_threads()
    state = np.zeros(nthreads, dtype=np.int64)
    for t in prange(nthreads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            state[tid] ^= (x >> 17)
    return state


@njit(parallel=True)
def padded_state_omp(iters, seed_val, pad_stride):
    nthreads = get_num_threads()
    state = np.zeros((nthreads, pad_stride), dtype=np.int64)
    for t in prange(nthreads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            state[tid, 0] ^= (x >> 17)
    return state


print("\n=== TASK 2: FALSE SHARING ===")
task2_times = {}
task2_states = {}
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    _ = unpadded_state_omp(10, STUDENT_ID)
    _ = padded_state_omp(10, STUDENT_ID, PAD_STRIDE)

    t0 = time.perf_counter()
    unp = unpadded_state_omp(ITERS_T2, STUDENT_ID)
    unp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pad = padded_state_omp(ITERS_T2, STUDENT_ID, PAD_STRIDE)
    pad_time = time.perf_counter() - t0

    assert np.array_equal(unp, pad[:, 0]), f"Task 2 state mismatch at {nt} threads."
    task2_times[nt] = {"unpadded_seconds": unp_time, "padded_seconds": pad_time}
    task2_states[nt] = pad[:, 0].copy()
    print(f"{nt} threads: unpadded={unp_time:.4f}s, padded={pad_time:.4f}s, "
          f"padded/unpadded={pad_time / unp_time:.3f}x")

speedup_8_t2 = (
    task2_times[8]["unpadded_seconds"] / task2_times[8]["padded_seconds"]
    if task2_times[8]["padded_seconds"] > 0 else float("nan")
)

DIAGNOSIS_T2 = (
    "In the unpadded array, different workers repeatedly write separate int64 values "
    "that can occupy the same 64-byte cache line. Under the MESI coherence protocol, a "
    "core that writes the line needs exclusive ownership, causing invalidations and cache-line "
    "ownership transfers between cores. This is false sharing: the variables are logically "
    "independent, but their physical placement creates coherence traffic. Padding places each "
    "worker's active slot at least one cache line apart and reduces this contention. The "
    "numerical states are checked for exact equality so the optimization changes placement, "
    "not the computation."
)


@njit
def _row_work(r, student_id, skew_mod):
    skew = (r * 7919 + student_id) % skew_mod
    return 40 + skew * skew


@njit
def _calculate_row(r, student_id, skew_mod):
    work = _row_work(r, student_id, skew_mod)
    total = 0.0
    for k in range(work):
        x = (r * 0.00013) + (k * 0.0017) + (student_id % 101) * 0.0009
        total += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return total, work


@njit(parallel=True)
def static_skew_omp(rows, student_id, skew_mod):
    nthreads = get_num_threads()
    out = np.zeros(rows, dtype=np.float64)
    ops = np.zeros((nthreads, 8), dtype=np.int64)
    for r in prange(rows):
        tid = get_thread_id()
        score, work = _calculate_row(r, student_id, skew_mod)
        out[r] = score
        ops[tid, 0] += work
    return out, ops[:, 0].copy()



@njit(parallel=True)
def balanced_skew_omp(rows, student_id, skew_mod):
    """
    Deterministic interleaved work distribution.
    Each worker processes rows: worker, worker+nthreads, ...
    """
    nthreads = get_num_threads()
    out = np.zeros(rows, dtype=np.float64)
    ops = np.zeros((nthreads, 8), dtype=np.int64)

    for worker in prange(nthreads):
        tid = get_thread_id()
        count = (rows - worker + nthreads - 1) // nthreads

        for j in range(count):
            r = worker + j * nthreads
            score, work = _calculate_row(r, student_id, skew_mod)
            out[r] = score
            ops[tid, 0] += work

    return out, ops[:, 0].copy()

    """
    Deterministic interleaved work distribution. Each OpenMP worker claims
    rows tid, tid+nthreads, tid+2*nthreads, ... . This distributes the
    pseudo-random heavy/light rows across workers without a racy shared cursor.
    """
    nthreads = get_num_threads()
    out = np.zeros(rows, dtype=np.float64)
    ops = np.zeros((nthreads, 8), dtype=np.int64)
    for worker in prange(nthreads):
        tid = get_thread_id()
        r = worker
        while r < rows:
            score, work = _calculate_row(r, student_id, skew_mod)
            out[r] = score
            ops[tid, 0] += work
            r += nthreads
    return out, ops[:, 0].copy()


def imbalance_ratio(ops):
    active = np.asarray(ops)[np.asarray(ops) > 0]
    if active.size == 0:
        return float("inf")
    return float(np.max(active) / np.min(active))


print("\n=== TASK 3: SKEWED WORKLOAD ===")
task3_times = {}
task3_ratios = {}
task3_outputs = {}
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    _ = static_skew_omp(64, STUDENT_ID, SKEW_MOD)
    _ = balanced_skew_omp(64, STUDENT_ID, SKEW_MOD)

    t0 = time.perf_counter()
    out_static, ops_static = static_skew_omp(ROWS_T3, STUDENT_ID, SKEW_MOD)
    static_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_balanced, ops_balanced = balanced_skew_omp(ROWS_T3, STUDENT_ID, SKEW_MOD)
    balanced_time = time.perf_counter() - t0

    assert np.max(np.abs(out_static - out_balanced)) < 1e-11
    ratio_static = imbalance_ratio(ops_static)
    ratio_balanced = imbalance_ratio(ops_balanced)
    task3_times[nt] = {"static_seconds": static_time, "balanced_seconds": balanced_time}
    task3_ratios[nt] = {"static": ratio_static, "balanced": ratio_balanced}
    task3_outputs[nt] = out_balanced
    print(f"{nt} threads: static={static_time:.4f}s (imbalance {ratio_static:.3f}x), "
          f"balanced={balanced_time:.4f}s (imbalance {ratio_balanced:.3f}x)")

set_num_threads(8)
out_t3, ops_t3 = balanced_skew_omp(ROWS_T3, STUDENT_ID, SKEW_MOD)
ratio_t3 = imbalance_ratio(ops_t3)
assert ratio_t3 < 1.15, (
    f"Task 3 balanced load imbalance {ratio_t3:.4f}x is not below 1.15x. "
    "Inspect the worker operation counts and scheduling."
)

DIAGNOSIS_T3 = (
    "Static contiguous partitioning can assign clusters of expensive rows to one worker, "
    "leaving other workers idle at the end of the parallel loop. The balanced kernel uses "
    "deterministic stride-interleaved assignment: worker t processes rows t, t+P, t+2P, "
    "where P is the active OpenMP worker count. This spreads the pseudo-random skew values "
    "across workers and avoids a shared cursor or unsafe atomic update. Each worker records "
    "its operation count in a private padded counter slot. The output is written by unique "
    "row indices, so the scheduling change does not alter numerical results."
)


def make_initial_grid():
    u = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    u[0, :] = 100.0
    u[-1, :] = -50.0
    left_profile = 25.0 * np.sin(np.linspace(0.0, np.pi, GRID_H))
    right_profile = 75.0 * np.cos(np.linspace(0.0, np.pi / 2.0, GRID_H))
    u[:, 0] = left_profile
    u[:, -1] = right_profile
    return u


@njit(parallel=True)
def sor_red_black_solve_pde(u, omega, iterations):
    h, w = u.shape
    nthreads = get_num_threads()
    red_max = np.zeros((nthreads, 8), dtype=np.float64)
    black_max = np.zeros((nthreads, 8), dtype=np.float64)

    for sweep in range(iterations):
        for t in range(nthreads):
            red_max[t, 0] = 0.0
            black_max[t, 0] = 0.0

        for r in prange(1, h - 1):
            tid = get_thread_id()
            local_max = red_max[tid, 0]
            for c in range(1, w - 1):
                if (r + c) % 2 == 0:
                    old = u[r, c]
                    new = (1.0 - omega) * old + 0.25 * omega * (
                        u[r - 1, c] + u[r + 1, c] + u[r, c - 1] + u[r, c + 1]
                    )
                    u[r, c] = new
                    delta = abs(new - old)
                    if delta > local_max:
                        local_max = delta
            red_max[tid, 0] = local_max

        for r in prange(1, h - 1):
            tid = get_thread_id()
            local_max = black_max[tid, 0]
            for c in range(1, w - 1):
                if (r + c) % 2 == 1:
                    old = u[r, c]
                    new = (1.0 - omega) * old + 0.25 * omega * (
                        u[r - 1, c] + u[r + 1, c] + u[r, c - 1] + u[r, c + 1]
                    )
                    u[r, c] = new
                    delta = abs(new - old)
                    if delta > local_max:
                        local_max = delta
            black_max[tid, 0] = local_max

    max_delta = 0.0
    for tid in range(nthreads):
        if red_max[tid, 0] > max_delta:
            max_delta = red_max[tid, 0]
        if black_max[tid, 0] > max_delta:
            max_delta = black_max[tid, 0]
    return max_delta


print("\n=== TASK 4: RED-BLACK SOR PDE ===")
task4_times = {}
task4_grids = {}
task4_residuals = {}
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    warm = np.zeros((16, 16), dtype=np.float64)
    _ = sor_red_black_solve_pde(warm, OMEGA_SOR, 2)

    grid = make_initial_grid()
    t0 = time.perf_counter()
    residual = sor_red_black_solve_pde(grid, OMEGA_SOR, 600)
    elapsed = time.perf_counter() - t0

    task4_times[nt] = elapsed
    task4_grids[nt] = grid.copy()
    task4_residuals[nt] = float(residual)
    print(f"{nt} threads: {elapsed:.4f}s, final-sweep max update={residual:.6e}")

grid_reference = task4_grids[1]
for nt in THREAD_COUNTS:
    max_diff = float(np.max(np.abs(task4_grids[nt] - grid_reference)))
    print(f"Grid comparison 1 vs {nt} threads: max abs diff={max_diff:.3e}")
    assert max_diff < 1e-9, f"Task 4 numerical mismatch at {nt} threads."

task4_speedups = {str(nt): float(task4_times[1] / task4_times[nt]) for nt in THREAD_COUNTS}


set_num_threads(8)
for idx, diag in enumerate([DIAGNOSIS_T1, DIAGNOSIS_T2, DIAGNOSIS_T3], start=1):
    assert isinstance(diag, str) and len(diag.strip()) >= 80, f"Complete DIAGNOSIS_T{idx}!"

hist_t1, wsums_t1, worker_hits_t1, _ = safe_hist_omp(data_t1, weights_t1, BINS_T1)
state_t2 = padded_state_omp(ITERS_T2, STUDENT_ID, PAD_STRIDE)[:, 0]
out_t3, ops_t3 = balanced_skew_omp(ROWS_T3, STUDENT_ID, SKEW_MOD)
grid_t4 = task4_grids[8]

checksum_blob = (
    f"{STUDENT_ID}|"
    f"{int(np.sum(hist_t1))}|{float(np.sum(wsums_t1)):.4f}|"
    f"{int(np.bitwise_xor.reduce(state_t2))}|"
    f"{float(np.sum(out_t3)):.6f}|"
    f"{float(np.sum(grid_t4)):.6f}"
)
lab_digest = hashlib.sha256(checksum_blob.encode()).hexdigest()[:20].upper()

bundle = {
    "instructor": "Instructor - Sufyan bin Uzayr",
    "student_name": STUDENT_NAME,
    "student_id": STUDENT_ID,
    "omp_layer": nb.threading_layer(),
    "cpu_cores": os.cpu_count(),
    "verification_digest": lab_digest,
    "task1": {
        "lost_counts_10_runs": lost_counts,
        "max_discrepancies_10_runs": max_discrepancies,
        "safe_histogram_total": int(np.sum(hist_t1)),
        "active_workers_at_8_threads": int(np.count_nonzero(worker_hits_t1)),
    },
    "task2": {
        "pad_stride_int64": PAD_STRIDE,
        "timings": {str(k): v for k, v in task2_times.items()},
        "speedup_unpadded_vs_padded_at_8_threads": float(speedup_8_t2),
    },
    "task3": {
        "balanced_imbalance_ratio_at_8_threads": float(ratio_t3),
        "timings": {str(k): v for k, v in task3_times.items()},
        "imbalance_ratios": {str(k): v for k, v in task3_ratios.items()},
    },
    "task4": {
        "grid_shape": [GRID_H, GRID_W],
        "iterations": 600,
        "timings_seconds": {str(k): float(v) for k, v in task4_times.items()},
        "speedups": task4_speedups,
        "max_abs_diff_vs_1_thread": {
            str(k): float(np.max(np.abs(task4_grids[k] - grid_reference)))
            for k in THREAD_COUNTS
        },
    },
}

with open("openmp_lab_results_230103077.json", "w", encoding="utf-8") as f:
    json.dump(bundle, f, indent=2, ensure_ascii=False)

print("\n=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===")
print(json.dumps(bundle, indent=2, ensure_ascii=False))
print("\nSaved: openmp_lab_results_230103077.json")
print("Saved source: openmp_lab_230103077.py")
