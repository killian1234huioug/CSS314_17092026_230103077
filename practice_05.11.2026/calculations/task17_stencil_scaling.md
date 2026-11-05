# Task 17 — 3D Stencil Hydrodynamics Simulation

## 1. Introduction

A 3D stencil solver updates values in a cubic grid.

In a 7-point stencil, each interior cell uses its own value and values from six neighboring cells.

The computational work scales with the volume of the grid:

\[
W_{\text{compute}}=O(N^3)
\]

Boundary communication scales with the surface area:

\[
W_{\text{communication}}=O(N^2)
\]

## 2. Weak-Scaling Assumption

Under Gustafson's Law, the total problem volume grows proportionally with the number of processors:

\[
N^3\propto P
\]

Taking the cube root:

\[
\boxed{N\propto P^{1/3}}
\]

## 3. Communication-to-Computation Ratio

The ratio is:

\[
R=\frac{W_{\text{communication}}}{W_{\text{compute}}}
\]

Using the complexity expressions:

\[
R\propto\frac{N^2}{N^3}
\]

Simplifying:

\[
R\propto\frac{1}{N}
\]

Since:

\[
N\propto P^{1/3}
\]

Therefore:

\[
\boxed{R\propto P^{-1/3}}
\]

## 4. Scaling Table

| Processors (P) | Relative grid dimension | Relative communication/computation ratio |
|---:|---:|---:|
| 1 | 1 | 1 |
| 8 | 2 | 0.5 |
| 64 | 4 | 0.25 |
| 512 | 8 | 0.125 |
| 4,096 | 16 | 0.0625 |

These are relative theoretical values.

## 5. Interpretation

As the grid becomes larger, its volume grows faster than its surface area.

This means that the communication-to-computation ratio decreases.

For example, when the processor count increases from 1 to 8, the grid dimension doubles. The relative communication-to-computation ratio becomes 0.5.

This is beneficial because processors perform more useful computation relative to boundary communication.

However, this result assumes ideal domain decomposition and does not include network latency, synchronization, or other communication costs.

## 6. Conclusion

Under ideal weak scaling, the 3D grid dimension grows as \(P^{1/3}\).

The communication-to-computation ratio decreases as \(P^{-1/3}\). Therefore, the surface-to-volume ratio improves as the grid becomes larger.

This helps explain why 3D stencil applications can achieve good weak scaling when communication is well organized.
