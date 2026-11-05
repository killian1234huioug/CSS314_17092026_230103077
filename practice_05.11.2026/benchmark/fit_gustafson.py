
"""
Task 23: Experimental Data Collection & Curve Fitting

Reads benchmark results and estimates the observed serial fraction
using least-squares regression against Gustafson's Law.
"""

import csv
import math
import statistics
from pathlib import Path


INPUT_FILE = Path("benchmarks/results.csv")


def read_results(filename):
    """Read raw benchmark measurements from CSV."""

    measurements = {}

    with filename.open("r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            workers = int(row["workers"])
            runtime = float(row["runtime_seconds"])

            measurements.setdefault(workers, []).append(runtime)

    return measurements


def calculate_statistics(measurements):
    """Calculate mean, standard deviation and standard error."""

    summary = {}

    for workers, runtimes in sorted(measurements.items()):

        mean_time = statistics.mean(runtimes)

        std_time = (
            statistics.stdev(runtimes)
            if len(runtimes) > 1
            else 0.0
        )

        standard_error = std_time / math.sqrt(len(runtimes))

        summary[workers] = {
            "runs": len(runtimes),
            "mean": mean_time,
            "std": std_time,
            "se": standard_error
        }

    return summary


def calculate_speedups(summary):
    """Calculate empirical speedup relative to one worker."""

    if 1 not in summary:
        raise ValueError("Results must contain measurements for 1 worker.")

    baseline = summary[1]["mean"]

    speedups = {}

    for workers, values in summary.items():
        speedups[workers] = baseline / values["mean"]

    return speedups


def estimate_alpha(speedups):
    """
    Estimate alpha using least-squares regression.

    Gustafson:
        S(P) = P - alpha * (P - 1)

    Rearranged:
        P - S(P) = alpha * (P - 1)
    """

    numerator = 0.0
    denominator = 0.0

    for workers, speedup in speedups.items():

        if workers == 1:
            continue

        x = workers - 1
        y = workers - speedup

        numerator += x * y
        denominator += x ** 2

    if denominator == 0:
        raise ValueError(
            "At least two different worker counts are required."
        )

    alpha = numerator / denominator

    return max(0.0, min(1.0, alpha))


def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        print("Run weak_scale_bench.py first.")
        return

    measurements = read_results(INPUT_FILE)

    if len(measurements) < 5:
        print("WARNING: At least 5 distinct worker counts are required.")

    summary = calculate_statistics(measurements)
    speedups = calculate_speedups(summary)
    alpha = estimate_alpha(speedups)

    print("\nEXPERIMENTAL RESULTS")
    print("=" * 75)

    print(
        f"{'Workers':<10}"
        f"{'Runs':<8}"
        f"{'Mean (s)':<15}"
        f"{'Std Dev':<15}"
        f"{'Speedup':<12}"
    )

    print("-" * 75)

    for workers, values in summary.items():

        print(
            f"{workers:<10}"
            f"{values['runs']:<8}"
            f"{values['mean']:<15.6f}"
            f"{values['std']:<15.6f}"
            f"{speedups[workers]:<12.4f}"
        )

    print("-" * 75)

    print("\nGUSTAFSON REGRESSION")
    print(f"Observed serial fraction (alpha): {alpha:.6f}")
    print(f"Observed parallel fraction: {1 - alpha:.6f}")

    print("\nTHEORETICAL GUSTAFSON SPEEDUP")

    for workers in sorted(summary):

        theoretical = workers - alpha * (workers - 1)

        print(
            f"P = {workers}: "
            f"S(P) = {theoretical:.4f}"
        )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
