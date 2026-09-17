import random
import threading
import time

totalHits = 0
lock = threading.Lock()


def worker(iterations):
    global totalHits

    for _ in range(iterations):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1:
            with lock:
                totalHits += 1


total = 5_000_000
threads_count = 4
iterations_per_thread = total // threads_count

start = time.perf_counter()

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

end = time.perf_counter()

pi = 4 * totalHits / total

print("Pi =", pi)
print("Time:", (end - start) * 1000, "ms")