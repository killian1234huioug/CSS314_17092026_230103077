
import os
import time
import threading

STUDENT_ID = "230103077"
NUM_THREADS = 4
INCREMENTS_PER_THREAD = 2_000_000
EXPECTED = NUM_THREADS * INCREMENTS_PER_THREAD

shared_counter = 0
locked_counter = 0
lock = threading.Lock()


def unsafe_worker():
    global shared_counter

    for i in range(INCREMENTS_PER_THREAD):
        value = shared_counter

        if i % 1000 == 0:
            time.sleep(0)

        shared_counter = value + 1


def safe_worker():
    global locked_counter

    for _ in range(INCREMENTS_PER_THREAD):
        with lock:
            locked_counter += 1


def run_test(worker):
    threads = [
        threading.Thread(target=worker)
        for _ in range(NUM_THREADS)
    ]

    start = time.perf_counter()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return time.perf_counter() - start


def main():
    global shared_counter, locked_counter

    print("=== TASK 6: SYNCHRONIZATION COST & DATA LOSS ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {os.uname().sysname}")
    print(f"Parallel threads: {NUM_THREADS}")
    print(f"Increments per thread: {INCREMENTS_PER_THREAD:,}")
    print()

    shared_counter = 0
    unsafe_time = run_test(unsafe_worker)
    actual = shared_counter
    lost = EXPECTED - actual
    lost_percentage = lost / EXPECTED * 100

    print("--- Version A: Without Lock ---")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {actual:,}")
    print(f"Lost updates:         {lost:,}")
    print(f"Lost updates (%):     {lost_percentage:.4f}%")
    print(f"Wall-clock time:      {unsafe_time:.3f} seconds")
    print()

    locked_counter = 0
    safe_time = run_test(safe_worker)

    print("--- Version B: With Lock ---")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {locked_counter:,}")
    print(f"Lost updates:         {EXPECTED - locked_counter:,}")
    print(f"Wall-clock time:      {safe_time:.3f} seconds")
    print()

    penalty = safe_time - unsafe_time
    percentage = (
        (safe_time - unsafe_time) / unsafe_time * 100
        if unsafe_time > 0 else 0
    )

    print("--- Synchronization Cost ---")
    print(f"Time penalty: {penalty:+.3f} seconds")
    print(f"Relative time change: {percentage:+.2f}%")
    print(
        "Verification: "
        + ("PASSED" if locked_counter == EXPECTED else "FAILED")
    )
    print("Status: COMPLETED")


if __name__ == "__main__":
    main()
