# Task 16 — Dense Matrix Multiplication Scaling (GEMM)

## 1. Introduction

Dense matrix multiplication is an important operation in scientific computing.

For two \(N\times N\) matrices:

\[
C=A\times B
\]

The computational complexity is:

\[
W=O(N^3)
\]

The memory requirement is:

\[
M=O(N^2)
\]

We use Gustafson's assumption that the total amount of work increases proportionally with the number of processors while execution time remains approximately constant.

## 2. Deriving the Matrix Dimension

Under weak scaling, total computational work grows proportionally with \(P\):

\[
N^3\propto P
\]

Taking the cube root of both sides:

\[
\boxed{N\propto P^{1/3}}
\]

Therefore, matrix dimension must grow with the cube root of the processor count.

## 3. Memory Growth

Memory grows as:

\[
M\propto N^2
\]

Since:

\[
N\propto P^{1/3}
\]

Substitute:

\[
M\propto(P^{1/3})^2
\]

Therefore:

\[
\boxed{M\propto P^{2/3}}
\]

This means that memory grows more slowly than the total computational work.

## 4. Scaling Table

For illustration, use the scaling relationship:

\[
N(P)=P^{1/3}
\]

| Processors (P) | Relative matrix dimension | Relative memory |
|---:|---:|---:|
| 1 | 1 | 1 |
| 8 | 2 | 4 |
| 64 | 4 | 16 |
| 512 | 8 | 64 |
| 4,096 | 16 | 256 |

These are relative scaling values, not actual matrix dimensions.

## 5. Interpretation

When the number of processors increases by a factor of 8:

- The matrix dimension doubles.
- The computational work increases by a factor of 8.
- The memory requirement increases by a factor of 4.

This allows the system to solve larger matrix multiplication problems while keeping the parallel computation time approximately constant, under ideal weak-scaling assumptions.

## 6. Conclusion

For dense matrix multiplication, the matrix dimension must grow as \(P^{1/3}\) to increase computational work proportionally with the processor count.

The memory requirement grows as \(P^{2/3}\). This shows that computational work and memory capacity have different scaling rates.
