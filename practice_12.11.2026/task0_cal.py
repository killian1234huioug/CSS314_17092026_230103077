
import multiprocessing as mp
import platform
import subprocess
import time

from harness import emit_verify_tag


STUDENT_ID = 230103077
ITERATIONS = 1_000_000


def get_sysctl(key):
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Not available"


def main():

    print("=" * 55)
    print("PHASE 0: HARDWARE TOPOLOGY & TELEMETRY")
    print("=" * 55)

    # Hardware information
    cpu_model = get_sysctl("machdep.cpu.brand_string")
    if cpu_model == "Not available":
        cpu_model = platform.processor() or platform.machine()

    physical_cores = get_sysctl("hw.physicalcpu")
    logical_cores = mp.cpu_count()

    cache_line = get_sysctl("hw.cachelinesize")
    l1_cache = get_sysctl("hw.l1dcachesize")
    l2_cache = get_sysctl("hw.l2cachesize")
    l3_cache = get_sysctl("hw.l3cachesize")

    print("\n--- HARDWARE INFORMATION ---")
    print("CPU Model:", cpu_model)
    print("Physical Cores:", physical_cores)
    print("Logical Threads:", logical_cores)
    print("Cache Line Size:", cache_line, "bytes")
    print("L1d Cache:", l1_cache, "bytes")
    print("L2 Cache:", l2_cache, "bytes")
    print("L3 Cache:", l3_cache, "bytes")

    # Timer latency benchmark
    print("\n--- TIMER LATENCY BENCHMARK ---")

    start = time.perf_counter_ns()

    for _ in range(ITERATIONS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_ns = total_ns / ITERATIONS
    total_ms = total_ns / 1_000_000

    print("Iterations:", ITERATIONS)
    print(f"Total time: {total_ms:.4f} ms")
    print(f"Average timer latency: {average_ns:.4f} ns")

    # Verification signature
    print("\n--- VERIFICATION ---")

    tag = emit_verify_tag(
        "PHASE0",
        STUDENT_ID,
        average_ns,
        total_ms
    )

    print(tag)

    print("\nPHASE 0 COMPLETED")


if __name__ == "__main__":
    mp.freeze_support()
    main()
