
import os
import platform
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

# Student information
STUDENT_ID = 230103077
ARRAY_SIZE = 20_000_000

def get_cpu_model():
    """Get the CPU model on macOS or other platforms."""
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    return platform.processor() or platform.machine()

def main():
    print("=== TASK 1: HARDWARE PROBE & SERIAL BASELINE ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {get_cpu_model()}")
    print(f"Logical Cores: {os.cpu_count()}")

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    print(f"Array Size: {ARRAY_SIZE:,}")
    print(f"Data Type: {data.dtype}")
    print("Execution Mode: Serial (1 thread)")

    with threadpool_limits(limits=1):
        start_time = time.perf_counter()

        total_sum = np.sum(data, dtype=np.float64)
        standard_deviation = np.std(data, dtype=np.float64)

        end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000

    print(f"Mathematical Sum: {total_sum:.10f}")
    print(f"Standard Deviation: {standard_deviation:.10f}")
    print(f"Execution Time: {elapsed_ms:.3f} ms")
    print("Status: COMPLETED")

if __name__ == "__main__":
    main()
