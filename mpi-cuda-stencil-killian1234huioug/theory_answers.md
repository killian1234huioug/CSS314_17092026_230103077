# Analytical & Forensic Engineering Responses — MPI + CUDA Stencil

> **Important:** Q3 and Q4 require measurements from your own Colab run. The notebook prints these values and writes them to `submission_manifest.json`. Replace the bracketed fields below with the values from that output. Do not invent telemetry.

## Q1. Memory Coalescing & Strided Column Packing

The local device array is C-contiguous (row-major). A row is stored consecutively, but a column is separated by the row stride: consecutive values in `u[1:257, 1]` are 258 × 8 = 2,064 bytes apart in the padded array. Therefore, a single ordinary contiguous DMA transfer cannot describe only the 256 useful column elements. The GPU pack kernel gathers those strided values into a contiguous 1-D device buffer; the host then transfers exactly 256 × 8 = 2,048 bytes for that column.

If an unstrided bounding-box copy is used for one column, it may transfer a full 256 × 256 rectangle (524,288 bytes) to obtain only 256 × 8 = 2,048 useful bytes. The excess is 522,240 bytes per column. For both left and right columns, the excess is 1,044,480 bytes per rank per exchange. This is why explicit GPU packing reduces PCIe traffic.

At the warp level, row-major adjacent `x` elements are contiguous and can be coalesced. Adjacent `y` elements of a column are separated by the full row pitch, so lanes accessing a column touch distant memory segments rather than one contiguous segment.

## Q2. Shared-Memory Tile Indexing & Bank Conflict Analysis

The shared tile is `18 × 18` doubles. Thread `(tx=0, ty=0)` evaluates its center at `sm_u[1,1]` and reads its four stencil neighbors at `sm_u[0,1]`, `sm_u[2,1]`, `sm_u[1,0]`, and `sm_u[1,2]`. Thread `(tx=15, ty=15)` evaluates `sm_u[16,16]` and reads `sm_u[15,16]`, `sm_u[17,16]`, `sm_u[16,15]`, and `sm_u[16,17]`. The same indexing applies to the diffusivity tile.

On the T4, shared memory has 32 banks, each 4 bytes wide. A float64 value occupies two adjacent 4-byte banks. For a warp accessing consecutive doubles within one row, the 64-bit accesses span bank pairs. The 18-double row pitch is 36 banks, which is 4 banks modulo 32; the warp layout can therefore cause overlapping bank-pair requests between the two 16-thread row segments of a warp. Depending on the exact generated instruction and access pattern, this can appear as a two-way bank conflict. The PTX and hardware transaction behavior should be interpreted for the actual compiled kernel rather than assuming every shared-memory access has identical conflicts.

## Q3. PTX Assembly Audit & Warp Synchronization

The kernel cooperatively loads the 18 × 18 shared-memory tile, then executes `cuda.syncthreads()` before any thread reads neighboring tile entries. The barrier is required because the tile is populated by multiple warps (a 16 × 16 block contains 256 threads, or 8 warps). Without the barrier, one warp could begin stencil arithmetic while another warp has not yet written its center or apron values. The result is a read-after-write hazard: a thread may read stale or uninitialized shared-memory data.

**Insert actual run values:** `st.shared` = [PTX_COUNTS.st_shared], `ld.shared` = [PTX_COUNTS.ld_shared], `bar.sync` = [PTX_COUNTS.bar_sync]. The exact float64 mnemonic counts are `st.shared.f64` = [PTX_COUNTS.st_shared_f64] and `ld.shared.f64` = [PTX_COUNTS.ld_shared_f64]. The manifest stores the measured counts.

## Q4. Multi-Process Context Sharing on a Single Physical GPU

Four MPI ranks are four independent OS processes, each with its own CUDA context, but all select device 0. The NVIDIA driver schedules work from these contexts onto the same physical T4 SMs and arbitrates access to copy engines. The ranks do not receive four independent GPUs: they share compute throughput, memory bandwidth, device memory, and DMA resources. Context switching and contention can reduce the benefit of overlap.

The boundary stream performs packing, host staging, unpacking, and boundary work; the interior stream performs cells that do not depend on newly received halos. MPI communication uses pinned host buffers because standard Colab Open MPI does not provide GPUDirect RDMA for this setup. On a single shared GPU, the two streams can overlap some work, but they still compete for the same SMs and memory system. This differs from four dedicated GPUs connected by InfiniBand, where each rank has independent local compute resources and network transfer can progress concurrently across nodes.

**Insert actual run values:** serialized = [serialized_ms] ms; overlapped = [overlapped_ms] ms; measured overlap change = `100 × (serialized_ms − overlapped_ms) / serialized_ms` = [calculate from manifest] %. A negative improvement means the overlapped implementation was slower for this workload.

## Q5. Point-to-Point Message Deadlock & MPI.PROC_NULL Semantics

With blocking `Send` followed by blocking `Recv` in a fixed directional order, every rank may wait for a matching receive while its peer is also blocked in a send. Small messages may appear to work because MPI can buffer them eagerly. Once a message exceeds the implementation's eager-buffer threshold, MPI may switch to a rendezvous protocol: the sender waits for the receiver to post the matching receive. If all ranks are blocked in sends in a circular dependency, no process posts the receive and the program deadlocks. `Isend` and `Irecv` allow receives and sends to be posted without blocking the calling thread, followed by `Waitall`.

`MPI.PROC_NULL` represents a missing neighbor at a non-periodic physical boundary. Communication with it completes without transferring data. For rank coordinates `(0,0)`, the up and left neighbors are `MPI.PROC_NULL`; the top and left ghost edges are therefore set to the homogeneous Dirichlet value zero, while the down and right halos are received from real neighboring ranks. This avoids special-case blocking communication code for each corner.

## Q6. Explicit CFL Stability & Energy Conservation Proof

For a 2-D explicit diffusion update, the stated stability bound is:

\[
\Delta t_{\max} = \frac{\Delta x^2}{4\kappa_{\max}}, \qquad \Delta x = \frac{1}{513}.
\]

The solver uses `dt = 0.10 × dt_max`, so it is one tenth of the computed diffusion stability limit. The manifest records the measured `kappa_max`, `dt_limit`, and `dt`; calculate the ratio `dt / dt_limit` from those values (expected to be 0.10).

The solver prints global energy `E = Σuᵢⱼ ΔxΔy` at steps 100, 200, 300, and 400, and an L2 state norm at the same checkpoints. A purely diffusive system with zero source and homogeneous Dirichlet boundaries is dissipative. This assignment also includes a localized source term, so strict monotonic decrease of energy is not mathematically guaranteed: the source can add energy. The correct forensic conclusion must therefore follow the actual trace. Report whether the measured values decrease monotonically, and if they do not, explain the source contribution rather than claiming energy conservation. Diffusion with fixed zero boundaries is not generally energy-conserving; it dissipates energy.

**Insert actual run values:** `kappa_max` = [kappa_max], `dt_limit` = [dt_limit], `dt` = [dt]. Energy trace: step 100 = [E100], 200 = [E200], 300 = [E300], 400 = [E400].
theory_answers.md
