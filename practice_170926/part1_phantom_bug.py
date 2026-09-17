import random
import threading

totalHits = 0


def worker(iterations):
    global totalHits

    for _ in range(iterations):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1:
            totalHits += 1


total = 50_000_000
threads_count = 4
iterations_per_thread = total // threads_count

for run in range(1, 6):
    totalHits = 0
    threads = []

    for _ in range(threads_count):
        t = threading.Thread(
            target=worker,
            args=(iterations_per_thread,)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    pi = 4 * totalHits / total

    print(f"Run {run}: hits = {totalHits}, pi = {pi}")
