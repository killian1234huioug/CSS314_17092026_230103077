# Task 11 — The Sun and Ni Memory-Bounded Speedup Comparison

## 1. Introduction

Amdahl's Law assumes a fixed problem size, while Gustafson's Law assumes that the problem size increases with the number of processors.

The Sun-Ni model introduces another important factor: available memory. It explains how a larger memory capacity can allow a parallel computer to solve larger problems.

## 2. Gustafson's Law

Gustafson's scaled speedup is:

\[
S_G(P)=P-\alpha(P-1)
\]

where:
- \(P\) is the number of processors.
- \(\alpha\) is the serial fraction measured on the parallel machine.

This model assumes that the workload grows with the processor count while execution time remains approximately constant.

## 3. Sun-Ni Memory-Bounded Model

The Sun-Ni model considers both processor count and memory capacity.

A simplified way to express its main idea is:

\[
W \propto M
\]

where \(W\) is the problem size and \(M\) is the available memory.

If each processor has approximately the same memory capacity \(m\), total memory is:

\[
M(P)=P \cdot m
\]

Therefore, the problem size can grow with the total available memory.

Unlike a simple fixed-size model, the Sun-Ni approach considers memory as a resource that can limit the amount of useful work.

## 4. Comparison

| Model | Main assumption | Main limitation |
|---|---|---|
| Amdahl | Fixed problem size | Serial execution |
| Gustafson | Problem size grows with processors | Communication and serial overhead |
| Sun-Ni | Problem size grows with available memory | Memory capacity and memory access |

## 5. When Gustafson's Law Overestimates or Underestimates Performance

Gustafson's Law can overestimate performance when:
- Communication between processors becomes expensive.
- Memory bandwidth is insufficient.
- Synchronization increases with processor count.
- The workload cannot be divided evenly.

It can underestimate performance when:
- The workload has very little serial work.
- The application benefits from better cache utilization.
- Memory capacity allows a much larger problem than expected.
- The algorithm has additional optimizations not included in the basic model.

## 6. Conclusion

Gustafson's Law focuses on processor scaling, while the Sun-Ni model also considers memory capacity. Both models help explain why larger parallel systems can solve larger problems. However, real performance depends on processors, memory capacity, communication, and algorithm design.
