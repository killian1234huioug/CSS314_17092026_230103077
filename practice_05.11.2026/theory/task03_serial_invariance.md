# Task 03 — The Serial Fraction Invariance Paradox

## 1. Introduction

Amdahl's Law and Gustafson's Law use different ways to measure the serial fraction. This difference is important because both laws study different scaling models.

Amdahl's Law assumes a fixed problem size. Gustafson's Law assumes that the problem size increases when more processors are available.

## 2. Amdahl's Law: Serial Fraction on One Processor

In Amdahl's Law, the serial fraction \(s\) is measured using a single processor.

The total execution time consists of:
- Serial work, which cannot be parallelized.
- Parallel work, which can be divided between processors.

The speedup is:

\[
S_A(P)=\frac{1}{s+\frac{1-s}{P}}
\]

When the number of processors \(P\) becomes very large, the maximum speedup is limited by the serial fraction:

\[
\lim_{P\to\infty}S_A(P)=\frac{1}{s}
\]

Therefore, even a small serial fraction can limit speedup when the problem size remains fixed.

## 3. Gustafson's Law: Serial Fraction on a Parallel Machine

Gustafson's Law measures the serial fraction \(\alpha\) during execution on a parallel machine.

The total parallel execution time is normalized to 1:

\[
\alpha+(1-\alpha)=1
\]

Here:
- \(\alpha\) is the serial part.
- \(1-\alpha\) is the parallel part.

The equivalent serial workload is:

\[
T_1=\alpha+P(1-\alpha)
\]

Therefore, the scaled speedup is:

\[
S_G(P)=\alpha+P(1-\alpha)
\]

or:

\[
S_G(P)=P-\alpha(P-1)
\]

Unlike Amdahl's model, Gustafson's model allows the problem size to grow with the number of processors.

## 4. Why the Apparent Contradiction Is Resolved

The two laws use different reference points.

Amdahl measures the serial fraction for the original fixed-size problem on one processor. Gustafson measures the serial fraction during parallel execution of a larger problem.

For the expanded workload, the serial fraction relative to the equivalent total serial work can be written as:

\[
s_{\text{expanded}}=
\frac{\alpha}{\alpha+P(1-\alpha)}
\]

As \(P\) increases, the parallel workload grows, so this fraction becomes smaller.

This explains why a serial part does not necessarily stop large speedups under weak scaling. The serial time still exists, but the amount of parallel work increases with the processor count.

## 5. Conclusion

The apparent contradiction comes from comparing two different workload assumptions. Amdahl's Law is designed for fixed-size problems, while Gustafson's Law is designed for scaled problems. Measuring the serial fraction on the parallel machine gives a more suitable description of weak scaling. It explains how a program can achieve high speedup even when some serial work remains.
