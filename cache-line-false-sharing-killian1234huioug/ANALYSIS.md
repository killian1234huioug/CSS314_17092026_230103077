# ANALYSIS.md — Cache Line & False Sharing Mechanics

Student: Kraman Zhaniya | ID: 230103077 | Git: killian1234huioug

Host: Apple M3, 8 logical cores, macOS, `hw.cachelinesize` = **128 bytes** (not 64). Core affinity pinning is not available on macOS (`affinity_supported: false`), so threads were scheduled by the OS. Numbers below come from `cache_lab_manifest.json`.

## Q1 — MESI state transitions (stride = 1)

With stride = 1, `buffer[0]` and `buffer[1]` (and all four thread slots) sit in the same cache line (on this machine a 128-byte line; the same reasoning holds for a 64-byte line).

1. Core 0 holds the line in **M** after its previous store; Core 1's copy is **I**.
2. Core 1 stores to `buffer[1]`. Its L1 does not hold the line in a writable state, so it issues a **Read-For-Ownership (RFO)**.
3. Core 0 receives the snoop, supplies the dirty data and moves **M → I**; Core 1 gets the line and moves **I → M**.
4. Core 0's next store to `buffer[0]` now finds its copy **I**, so it issues its own RFO and Core 1 goes **M → I**.

The line therefore bounces M ↔ I between cores on almost every store, even though no two threads touch the same address. A store can only commit from the store buffer to L1 once the core owns the line (M/E). Core 1's store waits in its store buffer for the RFO round trip, and younger stores queue behind it, so each update pays a cross-core transfer latency instead of an L1 hit. (Apple Silicon uses its own MOESI-style protocol and a weaker memory model than x86, but the ownership requirement for committing a store is the same.)

## Q2 — Quantitative slowdown and coherency overhead

Each thread performs 60,000,000 updates; ns/update = elapsed / 60,000,000.

| Layout | Elapsed (ns) | ns / update (per thread) | ns / update (aggregate, ÷4) |
|---|---|---|---|
| stride = 1 (8 B) | 306,345,458 | 5.106 | 1.276 |
| stride = 4 (32 B) | 294,549,750 | 4.909 | 1.227 |
| stride = 8 (64 B) | 268,536,166 | 4.476 | 1.119 |
| stride = 16 (128 B) | 265,307,209 | 4.422 | 1.105 |
| Register-private baseline | 10,017,042 | 0.167 | 0.042 |

Slowdown of stride = 1 relative to stride = 8: 306,345,458 / 268,536,166 = **1.14×**.

Interpretation:
- Private baseline → stride = 16: 0.167 → 4.422 ns. This ~4.3 ns is the cost of a real load → add → store chain through L1 each iteration (store-to-load forwarding latency), with no sharing.
- Stride = 16 → stride = 1: 4.422 → 5.106 ns, i.e. about 0.68 ns extra per update from cross-core coherence.
- The coherence penalty is much smaller than the textbook "several ×" slowdown, because the loop is already latency-bound by its own dependency chain (the next address depends on the value just loaded), which hides part of the coherence stalls.

## Q3 — Stride 8 vs. stride 16 (spatial prefetcher / line pairing)

Measured: stride = 8 → 268.5 ms (4.476 ns/update); stride = 16 → 265.3 ms (4.422 ns/update); ratio ≈ **1.01×**, essentially no difference.

On many Intel cores the L2 adjacent-line prefetcher pairs an even and an odd 64-byte line into a 128-byte sector, so 64-byte padding can still suffer secondary invalidations: a write to one line may pull or invalidate its buddy in another core's cache. 128-byte spacing puts each thread in its own sector, which removes that effect.

My host is an Apple M3, where the L1 coherency granule is already **128 bytes** (`hw.cachelinesize`). Consequently:
- stride = 1 and stride = 4 (offsets 0–96 B) put all 4 threads into one 128-byte line (4-way false sharing);
- stride = 8 (64 B spacing) puts threads 0,1 in one line and threads 2,3 in another (2-way false sharing);
- stride = 16 (128 B spacing) gives each thread its own line (no false sharing).

The measured order (306.3 → 294.5 → 268.5 → 265.3 ms) follows this model, but the stride 8 → 16 gain is small because the loop's own load-to-store dependency chain dominates the runtime and because the OS scheduled threads without core pinning. 64-byte padding is only fully safe when the hardware line really is 64 bytes and no adjacent-line prefetching pairs lines into 128-byte sectors.

## Q4 — Compiler register promotion vs. real memory stores

An optimizing compiler (LLVM LICM / scalar promotion) can notice that `buffer[slot]` is loaded and stored at a loop-invariant address on every iteration and rewrite the loop to keep the value in a register, storing it once after the loop. This is exactly what `worker_private_accumulate` does by hand, and it is legal because non-atomic, non-volatile accesses may be assumed not to be observed by other threads mid-loop.

In `worker_strided_accumulate` the accessed address is `slot + (buffer[slot] >> 63)`. It depends on the value just loaded from memory, so the compiler cannot prove the address is loop-invariant and must load and store through memory on every iteration. From `inspect_llvm()` (manifest `audit`): 1 × `store i64` and 5 × `load i64` in the function's IR (counted over the whole function), `no_object_fallback: true`, and no `PyObject` calls (nopython mode). The single `store i64` is the per-iteration write to `buffer[idx]`, and the loads include reading `buffer[slot]` / `buffer[idx]` each iteration. The private baseline's 0.167 ns/update versus 4.4 ns/update for stride = 16 confirms the strided kernel really does memory operations every iteration. Note that this relies on a data-dependent address, not on `volatile`.

## Q5 — CPython GIL masking vs. `nogil=True`

Without `nogil=True`, a pure-Python loop holds the Global Interpreter Lock while it runs bytecode, so only one thread executes at a time. The threads are time-sliced, not simultaneous, so two cores never write the same line at the same moment and no sustained RFO ping-pong appears. What does appear is lock contention: threads repeatedly acquire and release the GIL, trigger condition-variable wake-ups and OS context switches, and the extra threads add overhead without any speedup, so total time grows with thread count. That is a software serialization cost independent of memory layout, so changing the stride does not change it.

With `nogil=True` the compiled machine-code loops run truly in parallel on different cores, and the slowdown is caused by hardware cache-line contention, which depends on the stride (306.3 ms for stride = 1 vs. 265.3 ms for stride = 16 in my run). Context switches recorded in the manifest for stride = 1: voluntary = 0, involuntary = 809 (stride = 4: 0 / 631, stride = 8: 0 / 1040, stride = 16: 0 / 242, private baseline: 0 / 41). Zero voluntary switches shows that the nogil threads never block waiting on a lock; the involuntary ones are OS preemptions. A GIL-bound version would instead show frequent lock handoffs and voluntary switches (I did not measure that variant, so this part is theoretical).
