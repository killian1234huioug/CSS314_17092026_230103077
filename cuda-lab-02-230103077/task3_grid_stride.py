import numpy as np
from numba import cuda


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    N = h_arr.shape[0]
    d_arr = cuda.to_device(h_arr)
    threads_per_block = 256
    blocks_per_grid = 64            
    grid_stride_scale_kernel[blocks_per_grid, threads_per_block](d_arr, np.float32(factor), N)
    cuda.synchronize()
    return d_arr.copy_to_host()


def main():
    N = 2 ** 24
    factor = 3.5
    h = np.ones(N, dtype=np.float32)
    res = run_grid_stride(h, factor)
    assert res.shape[0] == N and np.all(res == np.float32(factor))
    print(f"TASK 3 PASSED: all {N:,} elements == {factor} "
          f"(launched {64 * 256:,} threads)")


if __name__ == "__main__":
    main()
