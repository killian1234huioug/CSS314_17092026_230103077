
"""
Task 22-23: Empirical Weak-Scaling Benchmark
Uses Python multiprocessing.
"""

import argparse
import csv
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def numerical_work(arguments):
    """Perform a fixed amount of numerical work."""
    worker_id, work_per_worker = arguments

    result = 0.0

    for i in range(1, work_per_worker + 1):
        x = i + worker_id
        result += math.sin(x * 0.001) ** 2
        result += math.cos(x * 0.001) ** 2

    return result


def run_benchmark(workers, work_per_worker):
    """Run the same workload per worker."""

    tasks = [
        (worker_id, work_per_worker)
        for worker_id in range(workers)
    ]

    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(numerical_work, tasks))

    elapsed = time.perf_counter() - start

    checksum = sum(results)

    return elapsed, checksum


def main():
    parser = argparse.ArgumentParser(
        description="Empirical weak-scaling benchmark"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() or 1
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5
    )

    parser.add_argument(
        "--work-per-worker",
        type=int,
        default=150000
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results.csv"
    )

    args = parser.parse_args()

    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if args.work_per_worker < 1:
        parser.error("--work-per-worker must be at least 1")

    max_workers = min(args.max_workers, os.cpu_count() or 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    print("EMPIRICAL WEAK-SCALING BENCHMARK")
    print("--------------------------------")
    print(f"Available CPUs: {os.cpu_count()}")
    print(f"Workers tested: 1 to {max_workers}")
    print(f"Repeats: {args.repeats}")
    print(f"Work per worker: {args.work_per_worker:,}")
    print()

    for workers in range(1, max_workers + 1):

        runtimes = []

        print(f"Testing {workers} worker(s)...")

        for run_number in range(1, args.repeats + 1):

            runtime, checksum = run_benchmark(
                workers,
                args.work_per_worker
            )

            runtimes.append(runtime)

            rows.append({
                "workers": workers,
                "run": run_number,
                "work_per_worker": args.work_per_worker,
                "total_work": workers * args.work_per_worker,
                "runtime_seconds": runtime,
                "checksum": checksum
            })

            print(
                f"  Run {run_number}: "
                f"{runtime:.6f} seconds"
            )

        mean_time = statistics.mean(runtimes)

        std_time = (
            statistics.stdev(runtimes)
            if len(runtimes) > 1
            else 0.0
        )

        print(
            f"  Mean: {mean_time:.6f}s | "
            f"Std: {std_time:.6f}s\n"
        )

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "workers",
                "run",
                "work_per_worker",
                "total_work",
                "runtime_seconds",
                "checksum"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("--------------------------------")
    print(f"Results saved to: {output_path}")
    print(f"Total measurements: {len(rows)}")


if __name__ == "__main__":
    main()
