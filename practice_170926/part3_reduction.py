import random
import threading
import time


def worker(iterations, results, index):
    local_hits = 0

    for _ in range(iterations):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1:
            local_hits += 1

    results[index] = local_hits


def run_test(thread_count):
    total = 100_000_000
    iterations_per_thread = total // thread_count

    results = [0] * thread_count
    threads = []

    start = time.perf_counter()

    for i in range(thread_count):
        t = threading.Thread(
            target=worker,
            args=(iterations_per_thread, results, i)
        )

        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_hits = sum(results)

    end = time.perf_counter()

    runtime = (end - start) * 1000
    pi = 4 * total_hits / total

    return runtime, pi


thread_counts = [1, 2, 4, 8, 16, 32]

times = {}

for t in thread_counts:
    runtime, pi = run_test(t)

    times[t] = runtime

    print(
        f"{t} threads: "
        f"{runtime:.2f} ms, "
        f"pi = {pi:.6f}"
    )


baseline = times[1]

print("\nRESULTS")
print("-" * 50)

for t in thread_counts:
    speedup = baseline / times[t]
    efficiency = speedup / t * 100

    print(
        f"{t:2} threads | "
        f"{times[t]:8.2f} ms | "
        f"{speedup:5.2f}x | "
        f"{efficiency:6.1f}%"
    )