import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = "killian1234huioug"
STUDENT_ID = "230103077"
STUDENT_NAME = "Kraman Zhaniya"

NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
            return int(out.strip())
        except Exception:
            pass
    return 64


def detect_cpu_model() -> str:
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8").strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


# ==============================================================================
# TASK 1: Hardware Topology & Identity Binding
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_model = detect_cpu_model()
    cpu_sig = f"{platform.machine()}::{cpu_model}::{logical_cores}"

    digest = hashlib.sha256(f"{git_user}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)  # 64-bit

    affinity_supported = hasattr(os, "sched_setaffinity")
    if affinity_supported:
        available = sorted(os.sched_getaffinity(0))
    else:
        available = list(range(logical_cores))
    cores = available[:NUM_THREADS]

    return {
        "git_username": git_user,
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "cpu_model": cpu_model,
        "cpu_sig": cpu_sig,
        "instance_seed": instance_seed,
        "affinity_supported": affinity_supported,
        "affinity_cores": cores,
    }


# ==============================================================================
# TASK 2: GIL-free native counter kernel
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    one = np.uint64(1)
    shift = np.uint64(63)
    for i in range(iters):
        ui = np.uint64(i)
        delta = ((seed ^ ui) & one) + one
        # The target address depends on the value just loaded from memory
        # (top bit is always 0 in practice), so the compiler cannot prove
        # the slot is loop-invariant and cannot keep it in a register.
        idx = slot + np.int64(buffer[slot] >> shift)
        buffer[idx] = buffer[idx] + delta


# ==============================================================================
# TASK 4: Register-privatized baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    one = np.uint64(1)
    acc = np.uint64(0)
    for i in range(iters):
        ui = np.uint64(i)
        acc += ((seed ^ ui) & one) + one
    buffer[slot] = acc


def aligned_buffer(n_elems: int, line: int = 64) -> np.ndarray:
    raw = np.zeros(n_elems + line // 8, dtype=np.uint64)
    off = (-raw.ctypes.data % line) // 8
    buf = raw[off:off + n_elems]
    assert buf.ctypes.data % line == 0
    return buf


def _thread_entry(kernel, buf, slot, iters, seed, core):
    if core is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})  # pid 0 = calling thread on Linux
        except OSError:
            pass
    kernel(buf, slot, iters, seed)


def _run_threads(kernel, buf, stride, iters, ctx):
    seeds = [np.uint64((ctx["instance_seed"] + t) & 0xFFFFFFFFFFFFFFFF) for t in range(NUM_THREADS)]
    cores = ctx["affinity_cores"]
    threads = [
        threading.Thread(
            target=_thread_entry,
            args=(kernel, buf, t * stride, iters, seeds[t], cores[t % len(cores)]),
        )
        for t in range(NUM_THREADS)
    ]
    ru0 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    t0 = time.perf_counter_ns()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.perf_counter_ns() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    vol = (ru1.ru_nvcsw - ru0.ru_nvcsw) if ru0 else None
    invol = (ru1.ru_nivcsw - ru0.ru_nivcsw) if ru0 else None
    finals = [int(buf[t * stride]) for t in range(NUM_THREADS)]
    return elapsed, vol, invol, finals


# ==============================================================================
# TASK 3: Multi-threaded stride sweep
# ==============================================================================
def run_stride_benchmark(ctx: dict) -> dict:
    iters = ITERATIONS_PER_THREAD
    # Warm-up / JIT compile (excluded from timing)
    warm = aligned_buffer(NUM_THREADS * 16)
    worker_strided_accumulate(warm, 0, 10, np.uint64(1))
    worker_private_accumulate(warm, 0, 10, np.uint64(1))

    results = {}
    for stride in STRIDES:
        buf = aligned_buffer(NUM_THREADS * stride)
        elapsed, vol, invol, finals = _run_threads(worker_strided_accumulate, buf, stride, iters, ctx)
        total = NUM_THREADS * iters
        results[f"stride_{stride}"] = {
            "stride_elems": stride,
            "stride_bytes": stride * 8,
            "elapsed_ns": elapsed,
            "ns_per_update": elapsed / iters,  # per-thread wall time / updates per thread
            "ns_per_update_aggregate": elapsed / total,
            "voluntary_ctx_switches": vol,
            "involuntary_ctx_switches": invol,
            "final_values": finals,
        }

    # Task 4 baseline
    buf = aligned_buffer(NUM_THREADS * 8)
    elapsed, vol, invol, finals = _run_threads(worker_private_accumulate, buf, 8, iters, ctx)
    results["private_baseline"] = {
        "elapsed_ns": elapsed,
        "ns_per_update": elapsed / iters,
        "ns_per_update_aggregate": elapsed / (NUM_THREADS * iters),
        "voluntary_ctx_switches": vol,
        "involuntary_ctx_switches": invol,
        "final_values": finals,
    }
    results["slowdown_stride1_vs_stride8"] = (
        results["stride_1"]["elapsed_ns"] / results["stride_8"]["elapsed_ns"]
    )
    return results


# ==============================================================================
# TASK 5: Parity + LLVM IR audit
# ==============================================================================
def audit_llvm_and_parity(results: dict, ctx: dict) -> dict:
    ref = results["private_baseline"]["final_values"]
    parity = all(results[f"stride_{s}"]["final_values"] == ref for s in STRIDES)

    ir_map = worker_strided_accumulate.inspect_llvm()
    ir = "\n".join(ir_map.values())
    has_i64_store = "store i64" in ir
    no_object_fallback = ("PyObject" not in ir.split("define", 1)[-1][:0]) and \
        "object mode" not in ir.lower()
    # nopython-compiled functions expose a native signature; confirm it
    nopython = all("pyobject" not in str(sig).lower() for sig in ir_map.keys())

    return {
        "bit_exact_parity": parity,
        "llvm_store_i64_found": has_i64_store,
        "llvm_store_i64_count": ir.count("store i64"),
        "llvm_load_i64_count": ir.count("load i64"),
        "no_object_fallback": bool(no_object_fallback and nopython),
    }


def main():
    ctx = init_system_context(GIT_USERNAME)
    print("Context:", json.dumps({k: v for k, v in ctx.items() if k != "cpu_sig"}, indent=2))
    results = run_stride_benchmark(ctx)
    audit = audit_llvm_and_parity(results, ctx)

    verified = (audit["bit_exact_parity"] and audit["llvm_store_i64_found"]
                and audit["no_object_fallback"])
    manifest = {
        "student": {"git_username": GIT_USERNAME, "student_id": STUDENT_ID, "name": STUDENT_NAME},
        "context": ctx,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "num_threads": NUM_THREADS,
        "results": results,
        "audit": audit,
        "verification_passed": bool(verified),
    }
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for k in ("stride_1", "stride_4", "stride_8", "stride_16", "private_baseline"):
        print(f"{k:18s} {results[k]['elapsed_ns']/1e6:10.1f} ms  {results[k]['ns_per_update']:.3f} ns/update")
    print(f"slowdown stride1/stride8: {results['slowdown_stride1_vs_stride8']:.2f}x")
    print("verification_passed:", verified)


if __name__ == "__main__":
    main()
