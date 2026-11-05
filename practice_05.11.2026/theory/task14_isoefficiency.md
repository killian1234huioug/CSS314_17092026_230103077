# Task 14 — Isoefficiency Relation with Gustafson Scaling

## 1. Introduction

Isoefficiency describes how fast the problem size must grow with the number of processors to maintain constant parallel efficiency.

It helps evaluate whether a parallel algorithm can use more processors effectively.

## 2. Parallel Efficiency

Parallel efficiency is:

\[
E(P)=\frac{S(P)}{P}
\]

Under Gustafson's Law:

\[
S(P)=P-\alpha(P-1)
\]

Therefore:

\[
E(P)=1-\alpha+\frac{\alpha}{P}
\]

If the serial fraction \(\alpha\) remains constant, efficiency approaches:

\[
\lim_{P\to\infty}E(P)=1-\alpha
\]

Thus, efficiency remains positive when \(\alpha<1\).

## 3. Isoefficiency Under Ideal Weak Scaling

In an ideal weak-scaling program, each processor receives a constant amount of work \(k\).

If there are \(P\) processors, the total problem size is:

\[
W(P)=kP
\]

Therefore:

\[
\boxed{W(P)=\Theta(P)}
\]

This means that the problem size grows linearly with the number of processors.

If communication and synchronization overhead remain controlled, this growth can maintain approximately constant efficiency.

## 4. Practical Limitations

The linear isoefficiency result applies to an ideal Gustafson-compliant workload.

In real applications, overhead may increase with processor count. For example:

- Communication between processors.
- Synchronization operations.
- Load imbalance.
- Memory access delays.

If these overheads increase significantly, the problem size may need to grow faster than \(P\) to maintain the same efficiency.

The exact isoefficiency function then depends on the algorithm and its overhead.

## 5. Conclusion

Under ideal weak scaling, each processor handles a constant amount of work. Therefore, the total problem size grows linearly with the processor count, giving an isoefficiency function of \(\Theta(P)\).

However, real parallel systems may require additional work to compensate for communication and synchronization overhead. Therefore, the linear result is an ideal case, not a guarantee for every algorithm.
