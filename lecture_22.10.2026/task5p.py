
import os
import time
import hashlib
import statistics
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103077
NUM_TASKS = 10_000
HASHES_PER_TASK = 500
PAYLOAD_SIZE = 1024
REPEATS = 3

CORES = os.cpu_count() or 1
POOL_SIZES = sorted(set([
    1,
    max(1, CORES // 2),
    CORES,
    CORES * 2,
    CORES * 4,
    CORES * 16,
    256,
]))

BASE_PAYLOAD = bytes((i % 256 for i in range(PAYLOAD_SIZE)))


def cpu_task(task_id):
    """Hash a 1 KB payload 500 times."""
    payload = (
        BASE_PAYLOAD[:-4] + task_id.to_bytes(4, byteorder="little")
    )

    digest = b""
    for _ in range(HASHES_PER_TASK):
        digest = hashlib.sha256(payload).digest()
        payload = payload[:-32] + digest

    return digest


def run_batch(pool_size):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        results = list(executor.map(cpu_task, range(NUM_TASKS)))

    elapsed = time.perf_counter() - start

    assert len(results) == NUM_TASKS
    assert all(len(result) == 32 for result in results)

    return elapsed


def main():
    print("=== TASK 5: THREAD OVERSUBSCRIPTION THRESHOLD ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPU cores: {CORES}")
    print(f"Tasks: {NUM_TASKS:,}")
    print(f"SHA-256 operations per task: {HASHES_PER_TASK}")
    print(f"Payload size: {PAYLOAD_SIZE} bytes")
    print(f"Timing: median of {REPEATS} runs per pool size")
    print(f"Pool sizes: {POOL_SIZES}")
    print()

    results = []

    for pool_size in POOL_SIZES:
        print(f"Testing pool size {pool_size}...", flush=True)
        run_times = [
            run_batch(pool_size)
            for _ in range(REPEATS)
        ]
        elapsed = statistics.median(run_times)
        results.append((pool_size, elapsed))

    baseline = next(t for p, t in results if p == 1)


    first_slowdown = None
    for i in range(1, len(results)):
        if results[i][1] > results[i - 1][1]:
            first_slowdown = results[i][0]
            break

    print()
    print("Pool Size | Execution Time (ms) | Speedup vs 1 | Status")
    print("-" * 68)

    for i, (pool_size, elapsed) in enumerate(results):
        speedup = baseline / elapsed if elapsed > 0 else float("nan")

        if pool_size == 1:
            status = "BASELINE"
        elif pool_size == first_slowdown:
            status = "FIRST SLOWDOWN"
        elif i > 0 and elapsed > results[i - 1][1]:
            status = "SLOWER"
        else:
            status = "NO NEW SLOWDOWN"

        print(
            f"{pool_size:>9} | {elapsed * 1000:>19.3f} | "
            f"{speedup:>12.3f}x | {status}"
        )

    print()
    if first_slowdown is not None:
        print(
            f"First observed degradation: pool size {first_slowdown}. "
            "Its median time exceeded the preceding tested pool size."
        )
        print(
            "Possible causes include thread scheduling, context switching, "
            "and Python GIL contention; timing alone cannot isolate one cause."
        )
    else:
        print(
            "No first slowdown was detected between adjacent tested "
            "pool sizes in this run."
        )

    print("Verification: PASSED (all tasks completed)")
    print("Status: COMPLETED")


if __name__ == "__main__":
    main()
