# Task 04 — Asymptotic Analysis of Parallel Efficiency

## 1. Introduction

Parallel efficiency shows how effectively processors are used. It is calculated as speedup divided by the number of processors:

\[
E(P)=\frac{S(P)}{P}
\]

We compare the efficiency of Gustafson's Law and Amdahl's Law when the number of processors approaches infinity.

## 2. Gustafson's Law

The scaled speedup formula is:

\[
S_G(P)=P-\alpha(P-1)
\]

Parallel efficiency is:

\[
E_G(P)=\frac{S_G(P)}{P}
\]

Substitute the speedup formula:

\[
E_G(P)=\frac{P-\alpha(P-1)}{P}
\]

Simplify:

\[
E_G(P)=1-\alpha+\frac{\alpha}{P}
\]

Now calculate the limit:

\[
\lim_{P\to\infty}E_G(P)
=1-\alpha+\frac{\alpha}{\infty}
\]

Therefore:

\[
\boxed{\lim_{P\to\infty}E_G(P)=1-\alpha}
\]

This means that Gustafson's efficiency approaches a constant value. It does not become zero if the serial fraction is smaller than 1.

## 3. Amdahl's Law

Amdahl's speedup formula is:

\[
S_A(P)=\frac{1}{s+\frac{1-s}{P}}
\]

Its parallel efficiency is:

\[
E_A(P)=\frac{S_A(P)}{P}
\]

Substitute the speedup formula:

\[
E_A(P)=\frac{1}{Ps+(1-s)}
\]

Calculate the limit:

\[
\lim_{P\to\infty}E_A(P)
=\lim_{P\to\infty}\frac{1}{Ps+(1-s)}
\]

For \(s>0\), the denominator approaches infinity. Therefore:

\[
\boxed{\lim_{P\to\infty}E_A(P)=0}
\]

This means that Amdahl's efficiency approaches zero as the processor count increases for a fixed-size problem with a non-zero serial fraction.

## 4. Comparison

| Model | Efficiency formula | Limit |
|---|---|---|
| Gustafson | \(1-\alpha+\frac{\alpha}{P}\) | \(1-\alpha\) |
| Amdahl | \(\frac{1}{Ps+(1-s)}\) | \(0\), if \(s>0\) |

Gustafson's Law maintains a positive limiting efficiency because the workload grows with the number of processors.

Amdahl's Law loses efficiency because the problem size remains fixed while more processors are added.

## 5. Conclusion

The two models have different asymptotic behavior. Under Gustafson's Law, efficiency approaches \(1-\alpha\). Under Amdahl's Law, efficiency approaches zero when the serial fraction is greater than zero. This shows why weak scaling can use large processor systems more effectively than fixed-size strong scaling.
