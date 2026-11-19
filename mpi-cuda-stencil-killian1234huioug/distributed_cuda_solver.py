#!/usr/bin/env python3
"""
Distributed 2-D anisotropic reaction-diffusion stencil solver.
Run with: mpirun --allow-run-as-root --oversubscribe -np 4 python3 distributed_cuda_solver.py YOUR_GITHUB_USERNAME
Requires: Open MPI, mpi4py, NumPy, Numba, CUDA GPU.
"""
import os, sys, json, math, hashlib, subprocess, time, re
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

NX = NY = 512
NY_LOCAL = NX_LOCAL = 256
PAD = 258
STEPS = 400
LAMBDA_DAMP = 0.15
TILE = 16
BLOCK = (TILE, TILE)
GRID = (16, 16)
DX = 1.0 / 513.0
DY = DX

def gpu_uuid():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True
    )
    return out.strip().splitlines()[0].strip()

def derive_seed(username, uuid):
    digest = hashlib.sha256(f"{username.strip()}::{uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

@cuda.jit
def pack_halos(u, top, bottom, left, right, n):
    i = cuda.grid(1)
    if i < n:
        top[i] = u[1, i + 1]
        bottom[i] = u[n, i + 1]
        left[i] = u[i + 1, 1]
        right[i] = u[i + 1, n]

@cuda.jit
def unpack_halos(u, recv_top, recv_bottom, recv_left, recv_right, n,
                 has_top, has_bottom, has_left, has_right):
    i = cuda.grid(1)
    if i < n:
        if has_top:
            u[0, i + 1] = recv_top[i]
        else:
            u[0, i + 1] = 0.0
        if has_bottom:
            u[n + 1, i + 1] = recv_bottom[i]
        else:
            u[n + 1, i + 1] = 0.0
        if has_left:
            u[i + 1, 0] = recv_left[i]
        else:
            u[i + 1, 0] = 0.0
        if has_right:
            u[i + 1, n + 1] = recv_right[i]
        else:
            u[i + 1, n + 1] = 0.0
    # Corners are not used by the 5-point stencil.

@cuda.jit
def tiled_stencil(u, kappa, source, out, dt, dx, dy, lam,
                  r_start, r_end, c_start, c_end,
                  physical_top, physical_bottom, physical_left, physical_right):
    # Each block cooperatively loads an 18x18 tile: 16x16 cells + apron.
    sm_u = cuda.shared.array((18, 18), dtype=float64)
    sm_k = cuda.shared.array((18, 18), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x * 16
    by = cuda.blockIdx.y * 16
    tid = ty * 16 + tx

    # Flattened cooperative load: 324 entries across 256 threads.
    for q in range(tid, 324, 256):
        sy = q // 18
        sx = q - sy * 18
        gy = by + sy
        gx = bx + sx
        sm_u[sy, sx] = u[gy, gx]
        sm_k[sy, sx] = kappa[gy, gx]
    cuda.syncthreads()

    row = by + ty + 1
    col = bx + tx + 1
    if row >= 1 and row <= 256 and col >= 1 and col <= 256:
        if row >= r_start and row <= r_end and col >= c_start and col <= c_end:
            # Fixed homogeneous Dirichlet values on the four global physical edges.
            if ((physical_top and row == 1) or
                (physical_bottom and row == 256) or
                (physical_left and col == 1) or
                (physical_right and col == 256)):
                out[row, col] = 0.0
            else:
                sy = ty + 1
                sx = tx + 1
                uc = sm_u[sy, sx]
                kc = sm_k[sy, sx]
                ku = 0.5 * (kc + sm_k[sy - 1, sx])
                kd = 0.5 * (kc + sm_k[sy + 1, sx])
                kl = 0.5 * (kc + sm_k[sy, sx - 1])
                kr = 0.5 * (kc + sm_k[sy, sx + 1])
                flux_y = (kd * (sm_u[sy + 1, sx] - uc) -
                          ku * (uc - sm_u[sy - 1, sx])) / (dy * dy)
                flux_x = (kr * (sm_u[sy, sx + 1] - uc) -
                          kl * (uc - sm_u[sy, sx - 1])) / (dx * dx)
                out[row, col] = uc + dt * (flux_x + flux_y -
                                            lam * uc * uc * uc +
                                            source[row, col])

def local_fields(coords, seed):
    """Build only this rank's 258x258 arrays; no global field is allocated."""
    rr, cc = coords
    rng = np.random.default_rng(seed)
    # Include ghost coordinates so kappa is defined throughout the shared tile.
    gy = rr * 256 + (np.arange(258, dtype=np.float64) - 1.0)
    gx = cc * 256 + (np.arange(258, dtype=np.float64) - 1.0)
    y = gy / 513.0
    x = gx / 513.0
    X, Y = np.meshgrid(x, y)

    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    # The formula stays in [0.08, 0.24].
    kappa = np.clip(kappa, 0.08, 0.24).astype(np.float64)

    # Seed-dependent Gaussian pulse; deterministic on all ranks.
    centers = rng.uniform(0.25, 0.75, size=(2, 2))
    amps = rng.uniform(0.35, 0.75, size=2)
    u = np.zeros((258, 258), dtype=np.float64)
    for p in range(2):
        cx, cy = centers[p]
        u += amps[p] * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2.0 * 0.075**2))

    # Localized, weak source around the domain center.
    source = 0.002 * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2.0 * 0.045**2))
    source = source.astype(np.float64)

    # Physical Dirichlet edges: zero owned boundary cells on the global perimeter.
    if rr == 0: u[1, 1:257] = 0.0
    if rr == 1: u[256, 1:257] = 0.0
    if cc == 0: u[1:257, 1] = 0.0
    if cc == 1: u[1:257, 256] = 0.0
    return u, kappa, source

def exchange_halos_async(cart, neighbors, u_d, send_d, recv_d,
                         send_h, recv_h, stream):
    """Pack on GPU, stage through pinned host buffers, exchange with nonblocking MPI."""
    n = 256
    pack_halos[1, 256, stream](u_d, send_d["top"], send_d["bottom"],
                                send_d["left"], send_d["right"], n)
    for d in ("top", "bottom", "left", "right"):
        send_d[d].copy_to_host(send_h[d], stream=stream)
    stream.synchronize()

    up, down, left, right = neighbors
    # A receive from a neighbor expects the tag of the neighbor's opposite outgoing face.
    recv_specs = [
        (up,  "top",    101),
        (down,"bottom", 100),
        (left,"left",   103),
        (right,"right", 102),
    ]
    send_specs = [
        (up,  "top",    100),
        (down,"bottom", 101),
        (left,"left",   102),
        (right,"right", 103),
    ]
    requests = []
    for nbr, face, tag in recv_specs:
        if nbr != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_h[face], source=nbr, tag=tag))
        else:
            recv_h[face].fill(0.0)
    for nbr, face, tag in send_specs:
        if nbr != MPI.PROC_NULL:
            requests.append(cart.Isend(send_h[face], dest=nbr, tag=tag))
    if requests:
        MPI.Request.Waitall(requests)

    for d in ("top", "bottom", "left", "right"):
        recv_d[d].copy_to_device(recv_h[d], stream=stream)
    unpack_halos[1, 256, stream](
        u_d, recv_d["top"], recv_d["bottom"], recv_d["left"], recv_d["right"],
        n, up != MPI.PROC_NULL, down != MPI.PROC_NULL,
        left != MPI.PROC_NULL, right != MPI.PROC_NULL)
    stream.synchronize()

def make_buffers():
    send_d = {d: cuda.device_array(256, dtype=np.float64) for d in
              ("top", "bottom", "left", "right")}
    recv_d = {d: cuda.device_array(256, dtype=np.float64) for d in
              ("top", "bottom", "left", "right")}
    send_h = {d: cuda.pinned_array(256, dtype=np.float64) for d in
              ("top", "bottom", "left", "right")}
    recv_h = {d: cuda.pinned_array(256, dtype=np.float64) for d in
              ("top", "bottom", "left", "right")}
    return send_d, recv_d, send_h, recv_h

def run_mode(cart, neighbors, u0, k_h, s_h, dt, mode, steps=STEPS):
    u_a = cuda.to_device(u0)
    u_b = cuda.device_array_like(u_a)
    k_d = cuda.to_device(k_h)
    s_d = cuda.to_device(s_h)
    boundary = cuda.stream()
    interior = cuda.stream()
    send_d, recv_d, send_h, recv_h = make_buffers()
    n = 256
    energy_trace = []
    residual_trace = []
    local_kmax = float(np.max(k_h[1:257, 1:257]))
    kmax = cart.allreduce(local_kmax, op=MPI.MAX)
    initial_host = u0
    initial_interior = initial_host[1:257, 1:257]
    e0 = cart.allreduce(float(np.sum(initial_interior) * DX * DY), op=MPI.SUM)
    l20 = math.sqrt(cart.allreduce(float(np.sum(initial_interior * initial_interior)), op=MPI.SUM))
    energy_trace.append({"step": 0, "energy": e0})
    residual_trace.append({"step": 0, "l2": l20})

    # Compile kernels before timing and PTX audit.
    tiled_stencil[GRID, BLOCK](u_a, k_d, s_d, u_b, dt, DX, DY, LAMBDA_DAMP,
                               2, 255, 2, 255,
                               neighbors[0] == MPI.PROC_NULL,
                               neighbors[1] == MPI.PROC_NULL,
                               neighbors[2] == MPI.PROC_NULL,
                               neighbors[3] == MPI.PROC_NULL)
    cuda.synchronize()

    start = time.perf_counter()
    current, nxt = u_a, u_b
    for step in range(1, steps + 1):

        if mode == "serialized":
            exchange_halos_async(
                cart, neighbors, current,
                send_d, recv_d, send_h, recv_h, boundary
            )

            tiled_stencil[GRID, BLOCK, boundary](
                current, k_d, s_d, nxt,
                dt, DX, DY, LAMBDA_DAMP,
                1, 256, 1, 256,
                neighbors[0] == MPI.PROC_NULL,
                neighbors[1] == MPI.PROC_NULL,
                neighbors[2] == MPI.PROC_NULL,
                neighbors[3] == MPI.PROC_NULL
            )

            boundary.synchronize()

        else:
            # Pack halo data on the GPU
            pack_halos[1, 256, boundary](
                current,
                send_d["top"], send_d["bottom"],
                send_d["left"], send_d["right"],
                n
            )

            # Copy packed halos from GPU to pinned host memory
            for d in ("top", "bottom", "left", "right"):
                send_d[d].copy_to_host(send_h[d], stream=boundary)

            boundary.synchronize()

            up, down, left, right = neighbors
            reqs = []

            # Post nonblocking receives
            for nbr, face, tag in [
                (up, "top", 101),
                (down, "bottom", 100),
                (left, "left", 103),
                (right, "right", 102)
            ]:
                if nbr != MPI.PROC_NULL:
                    reqs.append(
                        cart.Irecv(recv_h[face], source=nbr, tag=tag)
                    )
                else:
                    recv_h[face].fill(0.0)

            # Post nonblocking sends
            for nbr, face, tag in [
                (up, "top", 100),
                (down, "bottom", 101),
                (left, "left", 102),
                (right, "right", 103)
            ]:
                if nbr != MPI.PROC_NULL:
                    reqs.append(
                        cart.Isend(send_h[face], dest=nbr, tag=tag)
                    )

            # Calculate the interior region on a separate CUDA stream
            tiled_stencil[GRID, BLOCK, interior](
                current, k_d, s_d, nxt,
                dt, DX, DY, LAMBDA_DAMP,
                2, 255, 2, 255,
                up == MPI.PROC_NULL,
                down == MPI.PROC_NULL,
                left == MPI.PROC_NULL,
                right == MPI.PROC_NULL
            )

            # Wait for MPI communication
            if reqs:
                MPI.Request.Waitall(reqs)

            # Copy received halos to GPU
            for d in ("top", "bottom", "left", "right"):
                recv_d[d].copy_to_device(recv_h[d], stream=boundary)

            # Unpack received halos
            unpack_halos[1, 256, boundary](
                current,
                recv_d["top"], recv_d["bottom"],
                recv_d["left"], recv_d["right"],
                n,
                up != MPI.PROC_NULL,
                down != MPI.PROC_NULL,
                left != MPI.PROC_NULL,
                right != MPI.PROC_NULL
            )

            # Calculate boundary regions
            tiled_stencil[GRID, BLOCK, boundary](
                current, k_d, s_d, nxt,
                dt, DX, DY, LAMBDA_DAMP,
                1, 1, 1, 256,
                up == MPI.PROC_NULL,
                down == MPI.PROC_NULL,
                left == MPI.PROC_NULL,
                right == MPI.PROC_NULL
            )

            tiled_stencil[GRID, BLOCK, boundary](
                current, k_d, s_d, nxt,
                dt, DX, DY, LAMBDA_DAMP,
                256, 256, 1, 256,
                up == MPI.PROC_NULL,
                down == MPI.PROC_NULL,
                left == MPI.PROC_NULL,
                right == MPI.PROC_NULL
            )

            tiled_stencil[GRID, BLOCK, boundary](
                current, k_d, s_d, nxt,
                dt, DX, DY, LAMBDA_DAMP,
                2, 255, 1, 1,
                up == MPI.PROC_NULL,
                down == MPI.PROC_NULL,
                left == MPI.PROC_NULL,
                right == MPI.PROC_NULL
            )

            tiled_stencil[GRID, BLOCK, boundary](
                current, k_d, s_d, nxt,
                dt, DX, DY, LAMBDA_DAMP,
                2, 255, 256, 256,
                up == MPI.PROC_NULL,
                down == MPI.PROC_NULL,
                left == MPI.PROC_NULL,
                right == MPI.PROC_NULL
            )

            boundary.synchronize()


        # Global reductions every 100 steps.
        if step % 100 == 0:
            host = nxt.copy_to_host()
            interior_u = host[1:257, 1:257]
            local_energy = float(np.sum(interior_u) * DX * DY)
            # L2 norm of the current state (reported as residual proxy).
            local_l2_sq = float(np.sum(interior_u * interior_u))
            total_energy = cart.allreduce(local_energy, op=MPI.SUM)
            total_l2 = math.sqrt(cart.allreduce(local_l2_sq, op=MPI.SUM))
            energy_trace.append({"step": step, "energy": total_energy})
            residual_trace.append({"step": step, "l2": total_l2})
        current, nxt = nxt, current

    cuda.synchronize()
    elapsed_local = (time.perf_counter() - start) * 1000.0
    elapsed = cart.allreduce(elapsed_local, op=MPI.MAX)
    final = current.copy_to_host()
    return final, elapsed, energy_trace, residual_trace, kmax

def ptx_audit():
    # Trigger compilation on a small valid device allocation.
    u = cuda.to_device(np.zeros((258,258), dtype=np.float64))
    k = cuda.to_device(np.ones((258,258), dtype=np.float64) * 0.1)
    s = cuda.to_device(np.zeros((258,258), dtype=np.float64))
    o = cuda.device_array((258,258), dtype=np.float64)
    tiled_stencil[GRID, BLOCK](u,k,s,o,1e-7,DX,DY,LAMBDA_DAMP,1,256,1,256,
                               True, True, True, True)
    cuda.synchronize()
    ptx_result = tiled_stencil.inspect_asm()
    ptx = "\\n".join(ptx_result.values()) if isinstance(ptx_result, dict) else ptx_result
    counts = {
        "st_shared": len(re.findall(r"\bst\.shared(?:\.[a-z0-9]+)*", ptx)),
        "ld_shared": len(re.findall(r"\bld\.shared(?:\.[a-z0-9]+)*", ptx)),
        "bar_sync": len(re.findall(r"\bbar\.sync\b", ptx)),
        "st_shared_f64": len(re.findall(r"\bst\.shared\.f64\b", ptx)),
        "ld_shared_f64": len(re.findall(r"\bld\.shared\.f64\b", ptx)),
    }
    passed = counts["st_shared"] > 0 and counts["ld_shared"] > 0 and counts["bar_sync"] > 0
    return counts, passed

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: distributed_cuda_solver.py YOUR_GITHUB_USERNAME")
    username = sys.argv[1].strip()
    if not username or username.lower().startswith("replace"):
        raise ValueError("Pass your real GitHub username as the command-line argument.")

    world = MPI.COMM_WORLD
    rank, size = world.Get_rank(), world.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks, received {size}")
    cuda.select_device(0)
    uuid = world.bcast(gpu_uuid() if rank == 0 else None, root=0)
    seed = derive_seed(username, uuid)

    cart = world.Create_cart(dims=[2,2], periods=[False,False], reorder=False)
    coords = cart.Get_coords(cart.Get_rank())
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)
    neighbors = (up, down, left, right)
    physical_top = (up == MPI.PROC_NULL)
    physical_bottom = (down == MPI.PROC_NULL)
    physical_left = (left == MPI.PROC_NULL)
    physical_right = (right == MPI.PROC_NULL)

    u0, k_h, s_h = local_fields(coords, seed)
    local_kmax = float(np.max(k_h[1:257,1:257]))
    kmax = cart.allreduce(local_kmax, op=MPI.MAX)
    dt_limit = DX * DX / (4.0 * kmax)
    dt = 0.10 * dt_limit

    if rank == 0:
        print("IDENTITY:", username)
        print("GPU UUID:", uuid)
        print("INSTANCE_SEED:", seed)
        print("MPI ranks:", size, "| Cartesian dims: (2, 2)")
        print("GPU:", "Tesla T4 expected; device 0 selected")
        print("Grid: 512x512 | local padded shape: 258x258")
        print(f"kappa_max={kmax:.16e} dx={DX:.16e} dt_limit={dt_limit:.16e} dt={dt:.16e}")

    # Warm up and run serialized then overlapped from identical initial data.
    serial, serial_ms, e_serial, r_serial, _ = run_mode(
        cart, neighbors, u0, k_h, s_h, dt, "serialized")
    overlap, overlap_ms, e_overlap, r_overlap, _ = run_mode(
        cart, neighbors, u0, k_h, s_h, dt, "overlapped")
    parity_local = float(np.max(np.abs(serial - overlap)))
    parity = cart.allreduce(parity_local, op=MPI.MAX)
    counts, ptx_pass = ptx_audit()

    if rank == 0:
        print(f"serialized_ms={serial_ms:.6f}")
        print(f"overlapped_ms={overlap_ms:.6f}")
        print(f"parity_max_diff={parity:.16e}")
        print("PTX_COUNTS:", json.dumps(counts, sort_keys=True))
        print("ptx_audit_passed:", ptx_pass)
        print("ENERGY_TRACE:", json.dumps(e_overlap))
        print("L2_TRACE:", json.dumps(r_overlap))
        manifest = {
            "github_username": username,
            "gpu_uuid": uuid,
            "instance_seed": seed,
            "mpi_ranks": size,
            "cartesian_dims": [2,2],
            "global_grid": [NY,NX],
            "local_padded_shape": [258,258],
            "steps": STEPS,
            "kappa_max": kmax,
            "dx": DX,
            "dt_limit": dt_limit,
            "dt": dt,
            "serialized_ms": serial_ms,
            "overlapped_ms": overlap_ms,
            "parity_max_diff": parity,
            "ptx_counts": counts,
            "ptx_audit_passed": ptx_pass,
            "energy_trace": e_overlap,
            "l2_trace": r_overlap,
            "verification": {
                "four_ranks": size == 4,
                "parity": parity < 1e-12,
                "ptx": ptx_pass
            }
        }
        with open("submission_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("WROTE submission_manifest.json")
        if parity >= 1e-12:
            raise RuntimeError("Serialized/overlapped parity check failed.")
        if not ptx_pass:
            raise RuntimeError("PTX shared-memory/barrier audit failed.")
    cart.Barrier()

if __name__ == "__main__":
    main()
