
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import statistics

from harness import emit_verify_tag


STUDENT_ID = 230103077

TOTAL_STEPS = 16_000_000
SHM_SIZE = 4096
UINT64_SIZE = 8

STRIDES = [1, 2, 4, 8, 16, 32]
PROCESS_COUNTS = [1, 2, 4, 8]

WARMUP = 1
RUNS = 7

MASK = 0xFFFFFFFFFFFFFFFF
MULTIPLIER = 6364136223846793005


# Worker function must be at module level for multiprocessing spawn.
def fs_worker(shm_name, slot_idx, iters):

    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]

    for i in range(iters):
        acc = (
            acc * MULTIPLIER + (i | 1)
        ) & MASK

        arr[slot_idx] = acc

    del arr
    shm.close()


def run_trial(shm, processes, stride):

    # Reset shared memory before each trial.
    ctypes.memset(
        ctypes.addressof(
            ctypes.c_char.from_buffer(shm.buf)
        ),
        0,
        SHM_SIZE
    )

    iters = TOTAL_STEPS // processes

    workers = []

    start = time.perf_counter_ns()

    for worker_id in range(processes):

        slot_idx = worker_id * stride

        p = mp.Process(
            target=fs_worker,
            args=(shm.name, slot_idx, iters)
        )

        workers.append(p)
        p.start()

    for p in workers:
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(
                f"Worker failed with exit code {p.exitcode}"
            )

    end = time.perf_counter_ns()

    return (end - start) / 1e6


def measure(shm, processes, stride):

    for _ in range(WARMUP):
        run_trial(shm, processes, stride)

    samples = []

    for _ in range(RUNS):
        samples.append(
            run_trial(shm, processes, stride)
        )

    samples.sort()

    median = statistics.median(samples)

    q1 = samples[RUNS // 4]
    q3 = samples[(3 * RUNS) // 4]

    iqr = q3 - q1

    throughput = TOTAL_STEPS / median / 1000

    return {
        "median": median,
        "iqr": iqr,
        "throughput": throughput
    }


def print_result(label, p, stride, result):

    print(
        f"{label:12s} "
        f"p={p:<2d} "
        f"stride={stride:<2d} "
        f"median={result['median']:10.3f} ms "
        f"IQR={result['iqr']:9.3f} ms "
        f"throughput={result['throughput']:9.3f} Mops/s"
    )


def main():

    print("=" * 75)
    print("PHASE 1: FALSE SHARING & KARP-FLATT DIAGNOSTICS")
    print("=" * 75)

    logical_cores = mp.cpu_count()
    physical_cores = 8

    print("Logical cores:", logical_cores)
    print("Physical cores:", physical_cores)
    print("Total operations:", TOTAL_STEPS)

    shm = shared_memory.SharedMemory(
        create=True,
        size=SHM_SIZE
    )

    stride_results = {}
    scaling_results = {}

    try:

        # -------------------------------------------------
        # PART 1: STRIDE SWEEP
        # -------------------------------------------------

        print("\n--- PART 1: STRIDE SWEEP (p = 8) ---")

        for stride in STRIDES:

            result = measure(
                shm,
                physical_cores,
                stride
            )

            stride_results[stride] = result

            print_result(
                "Stride Sweep",
                physical_cores,
                stride,
                result
            )

        # -------------------------------------------------
        # PART 2: PROCESS SCALING
        # -------------------------------------------------

        print("\n--- PART 2: PROCESS SCALING ---")

        for stride in [1, 16]:

            print(f"\nResults for stride = {stride}")

            scaling_results[stride] = {}

            for p in PROCESS_COUNTS:

                result = measure(
                    shm,
                    p,
                    stride
                )

                scaling_results[stride][p] = result

                print_result(
                    "Scaling",
                    p,
                    stride,
                    result
                )

        # -------------------------------------------------
        # PART 3: SPEEDUP AND EFFICIENCY
        # -------------------------------------------------

        print("\n--- PART 3: SPEEDUP, EFFICIENCY & KARР-FLATT ---")

        for stride in [1, 16]:

            baseline = scaling_results[stride][1]["median"]

            print(f"\nStride = {stride}")
            print(
                f"{'p':<8}{'Time (ms)':<15}"
                f"{'Speedup':<15}{'Efficiency':<15}"
                f"{'Karp-Flatt':<15}"
            )

            for p in PROCESS_COUNTS:

                tp = scaling_results[stride][p]["median"]

                speedup = baseline / tp
                efficiency = speedup / p

                if p >= 2:
                    serial_fraction = (
                        (1 / speedup) - (1 / p)
                    ) / (1 - (1 / p))
                else:
                    serial_fraction = None

                if serial_fraction is None:
                    serial_text = "N/A"
                else:
                    serial_text = f"{serial_fraction:.6f}"

                print(
                    f"{p:<8}"
                    f"{tp:<15.3f}"
                    f"{speedup:<15.4f}"
                    f"{efficiency:<15.4f}"
                    f"{serial_text:<15}"
                )

        # -------------------------------------------------
        # PART 4: VERIFICATION
        # -------------------------------------------------

        print("\n--- PART 4: VERIFICATION ---")

        median_stride1 = scaling_results[1][physical_cores]["median"]
        median_stride16 = scaling_results[16][physical_cores]["median"]

        tag = emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            median_stride1,
            median_stride16
        )

        print(tag)

        print("\nPHASE 1 COMPLETED")

    finally:

        shm.close()
        shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
