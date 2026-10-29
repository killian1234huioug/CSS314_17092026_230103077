# Karp–Flatt Practice Lab — RESULTS

## Hardware profile

| Item | Value |
|---|---|
| Machine | MacBook Air |
| Chip | Apple M3 |
| `os.cpu_count()` | 8 |
| RAM | 16 GB |
| OS | macOS 26.2|
| Python | 3.9.6 |
| Method | best of 3 runs per configuration |

## Measured results

| Experiment | T1 (s) | T2 (s) | T4 (s) | Sp(2) | Sp(4) | e(2) | e(4) | Diagnosis |
|---|---|---|---|---|---|---|---|---|
| A: Serial Tail | 4.090 | 2.320 | 1.427 | 1.76x | 2.87x | 0.134 | 0.132 | Flat e → fixed Amdahl serial path |
| B: Lock Contention | 1.910 | 6.867 | 7.270 | 0.28x | 0.26x | 6.189 | 4.740 | e ≫ 1, T_p ≫ T1 → lock thrashing |
| C: Privatized State | 0.472 | 0.256 | 0.149 | 1.84x | 3.16x | 0.087 | 0.088 | Small flat e, E4 = 79 % → scaling restored |

Formulas: `Sp = T1 / Tp`, `e(p) = ((p · Tp / T1) − 1) / (p − 1)`.

## Analysis

**A — Serial tail.** e(2) = 0.134 and e(4) = 0.132 (ratio 0.98), so e is flat: a fixed
serial path. The 0.40 s sleep alone predicts e ≈ 0.40 / 4.09 = 0.098; the remaining
≈ 0.15 s is constant process-pool start-up cost (macOS uses `spawn`), which behaves like
extra serial time.

**B — Lock contention.** Adding a second process made the run 3.6x slower (1.91 s → 6.87 s)
and 4 processes were no better (7.27 s). Under pure Amdahl, e equals the serial fraction
and cannot exceed 1, so e = 6.19 and 4.74 prove the slowdown comes from contention
(lock hand-off, kernel sleep/wake, cache-line traffic), not from sequential code.
Note: e(4) < e(2) here. Total time plateaus once the lock is contended (T2 ≈ T4), and when
T_p is flat, e(p) = (p·c − 1)/(p − 1) shrinks toward c as p grows. So "e grows
monotonically" did not hold literally on this machine; the pathology (T4 > T1, e ≫ 1) did.

**C — Privatization.** Zero shared state gives T4 < T1 with E4 = 79 % (> 70 %). e is flat
at ≈ 0.088. That is ≈ 0.04 s of constant pool start-up cost relative to a short 0.47 s
baseline, which is why it sits above the handout's "≈ 0.03–0.05" target. A larger
`total_increments` would dilute it.

## Notes on the handout's reference table

The "Exp B" reference row (T1 = 0.65, T2 = 1.10, T4 = 2.40) gives e(2) ≈ 2.385 and
e(4) ≈ 4.590 by the formula, not the printed 1.380 and 3.920.
