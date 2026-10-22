
import os
import time
import statistics
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 230103077
ARRAY_SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 3


def ipc_worker(conn):
    """Receive a pickled NumPy array through a pipe."""
    try:
        conn.send("READY")
        while True:
            array = conn.recv()
            if array is None:
                break

            checksum = float(np.sum(array, dtype=np.float64))
            conn.send(checksum)
    finally:
        conn.close()


def shared_memory_worker(conn):
    """Access an array stored in shared memory without copying it."""
    try:
        conn.send("READY")
        while True:
            message = conn.recv()
            if message is None:
                break

            shm_name, size = message
            shm = shared_memory.SharedMemory(name=shm_name)
            try:
                array = np.ndarray(
                    (size,), dtype=np.float64, buffer=shm.buf
                )
                checksum = float(np.sum(array, dtype=np.float64))
                conn.send(checksum)
            finally:
                del array
                shm.close()
    finally:
        conn.close()


def benchmark_ipc(ctx, array, expected_checksum):
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=ipc_worker, args=(child_conn,))
    process.start()
    child_conn.close()

    try:
        parent_conn.recv()  
        times = []

        for _ in range(REPEATS):
            start = time.perf_counter()
            parent_conn.send(array)
            checksum = parent_conn.recv()
            times.append(time.perf_counter() - start)

            if not np.isclose(checksum, expected_checksum, rtol=1e-12):
                raise RuntimeError("IPC checksum verification failed")

        return statistics.median(times) * 1000
    finally:
        parent_conn.send(None)
        process.join()
        parent_conn.close()


def benchmark_shared_memory(ctx, array, expected_checksum):
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    shared_array = np.ndarray(
        array.shape, dtype=array.dtype, buffer=shm.buf
    )
    shared_array[:] = array

    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=shared_memory_worker, args=(child_conn,))
    process.start()
    child_conn.close()

    try:
        parent_conn.recv()  
        times = []

        for _ in range(REPEATS):
            start = time.perf_counter()
            parent_conn.send((shm.name, len(array)))
            checksum = parent_conn.recv()
            times.append(time.perf_counter() - start)

            if not np.isclose(checksum, expected_checksum, rtol=1e-12):
                raise RuntimeError("Shared memory verification failed")

        return statistics.median(times) * 1000
    finally:
        parent_conn.send(None)
        process.join()
        parent_conn.close()
        del shared_array
        shm.close()
        shm.unlink()


def main():
    ctx = mp.get_context("spawn")

    print("=== TASK 4: HOST IPC / SHARED MEMORY TRANSFER PROFILE ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {os.uname().sysname}")
    print("Mode: CPU-only multiprocessing")
    print(f"Timing: median of {REPEATS} transfers")
    print("Times include transfer/handoff, receiver access, and acknowledgement.")
    print("Process startup and initial shared-memory population are excluded.")
    print()

    print(
        f"{'Array Size':>12} {'Array MB':>10} "
        f"{'IPC (ms)':>12} {'Shared Mem (ms)':>17} {'IPC/Shared':>12}"
    )
    print("-" * 69)

    for n in ARRAY_SIZES:
        rng = np.random.default_rng(STUDENT_ID + n)
        array = rng.random(n, dtype=np.float64)
        expected_checksum = float(np.sum(array, dtype=np.float64))

        ipc_ms = benchmark_ipc(ctx, array, expected_checksum)
        shared_ms = benchmark_shared_memory(
            ctx, array, expected_checksum
        )

        ratio = ipc_ms / shared_ms if shared_ms > 0 else float("inf")

        print(
            f"{n:>12,} {array.nbytes / 1_000_000:>10.1f} "
            f"{ipc_ms:>12.3f} {shared_ms:>17.3f} {ratio:>11.2f}x"
        )

    print()
    print("Verification: PASSED (receiver checksums match)")
    print("Status: COMPLETED")


if __name__ == "__main__":
    main()
