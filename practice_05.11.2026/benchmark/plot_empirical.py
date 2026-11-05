
"""
Task 24: Empirical vs Theoretical Scaling Plot

Creates a publication-quality comparison of:
1. Ideal linear speedup
2. Gustafson's Law
3. Empirical weak-scaling results with error bars
4. Amdahl's Law
"""

import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_FILE = Path("benchmarks/results.csv")
OUTPUT_FILE = Path("plots/scaling_comparison.png")


def load_data():
    """Load benchmark runtimes from CSV."""

    measurements = {}

    with INPUT_FILE.open("r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            workers = int(row["workers"])
            runtime = float(row["runtime_seconds"])

            measurements.setdefault(workers, []).append(runtime)

    return measurements


def calculate_alpha(speedups):
    """Estimate serial fraction using least-squares regression."""

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
        raise ValueError("Not enough worker configurations.")

    return max(0.0, min(1.0, numerator / denominator))


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "benchmarks/results.csv not found. "
            "Run the weak scaling benchmark first."
        )

    measurements = load_data()

    if len(measurements) < 2:
        raise ValueError("At least two worker counts are required.")

    workers = sorted(measurements)

    if 1 not in measurements:
        raise ValueError("A baseline measurement with 1 worker is required.")

    means = {}
    standard_errors = {}

    for p, runtimes in measurements.items():

        means[p] = statistics.mean(runtimes)

        std = (
            statistics.stdev(runtimes)
            if len(runtimes) > 1
            else 0.0
        )

        standard_errors[p] = std / math.sqrt(len(runtimes))

    baseline = means[1]

    speedups = {
        p: baseline / means[p]
        for p in workers
    }

    # Estimate alpha from experimental measurements.
    alpha = calculate_alpha(speedups)

    # Propagate runtime standard errors into speedup errors.
    baseline_se = standard_errors[1]

    speedup_errors = {}

    for p in workers:

        if p == 1:
            speedup_errors[p] = 0.0
            continue

        relative_error = math.sqrt(
            (baseline_se / baseline) ** 2
            + (standard_errors[p] / means[p]) ** 2
        )

        speedup_errors[p] = speedups[p] * relative_error

    max_p = max(workers)

    theoretical_p = list(range(1, max_p + 1))

    ideal = theoretical_p

    gustafson = [
        p - alpha * (p - 1)
        for p in theoretical_p
    ]

    amdahl = [
        1.0 / (alpha + (1.0 - alpha) / p)
        for p in theoretical_p
    ]

    # Create output directory.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Create publication-quality plot.
    plt.figure(figsize=(11, 7))

    plt.plot(
        theoretical_p,
        ideal,
        linestyle="--",
        linewidth=2,
        label="Ideal Linear Speedup"
    )

    plt.plot(
        theoretical_p,
        gustafson,
        linewidth=2,
        label=f"Gustafson's Law (alpha={alpha:.4f})"
    )

    plt.plot(
        theoretical_p,
        amdahl,
        linewidth=2,
        label=f"Amdahl's Law (alpha={alpha:.4f})"
    )

    plt.errorbar(
        workers,
        [speedups[p] for p in workers],
        yerr=[speedup_errors[p] for p in workers],
        fmt="o",
        markersize=7,
        capsize=5,
        linewidth=1.5,
        label="Measured Empirical Speedup"
    )

    plt.title(
        "Empirical vs Theoretical Scaling",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("Number of Processes", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)

    plt.xticks(workers)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("TASK 24 PLOT GENERATED")
    print("----------------------")
    print(f"Observed alpha: {alpha:.6f}")
    print(f"Maximum workers: {max_p}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
