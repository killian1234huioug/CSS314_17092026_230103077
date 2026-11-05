# Task 09 — Multi-Component Workload Partitioning

## 1. Given Information

A scientific simulation has three computational phases:

| Phase | Description | Time fraction |
|---|---|---:|
| A | Initial disk I/O and parameter broadcast | 2% |
| B | Boundary message-passing synchronization | 6% |
| C | Core domain flux computation | 92% |

The first two phases are treated as serial or overhead components.

## 2. Aggregate Serial Fraction

\[
\alpha=0.02+0.06
\]

\[
\boxed{\alpha=0.08=8\%}
\]

The parallel fraction is:

\[
p=0.92=92\%
\]

Check:

\[
0.08+0.92=1
\]

## 3. Scaled Speedup Formula

We use Gustafson's Law:

\[
S(P)=P-\alpha(P-1)
\]

Substitute \(\alpha=0.08\):

\[
S(P)=P-0.08(P-1)
\]

## 4. Calculation for 512 Processors

\[
S(512)=512-0.08(512-1)
\]

\[
S(512)=512-0.08(511)
\]

\[
S(512)=512-40.88
\]

\[
\boxed{S(512)=471.12}
\]

Efficiency:

\[
E(512)=\frac{471.12}{512}
\]

\[
\boxed{E(512)=92.015625\%}
\]

## 5. Calculation for 2,048 Processors

\[
S(2048)=2048-0.08(2048-1)
\]

\[
S(2048)=2048-0.08(2047)
\]

\[
S(2048)=2048-163.76
\]

\[
\boxed{S(2048)=1884.24}
\]

Efficiency:

\[
E(2048)=\frac{1884.24}{2048}
\]

\[
\boxed{E(2048)=92.00390625\%}
\]

## 6. Results Table

| Processors | Scaled Speedup | Efficiency |
|---:|---:|---:|
| 512 | 471.12 | 92.0156% |
| 2,048 | 1,884.24 | 92.0039% |

## 7. Conclusion

The aggregate serial and overhead fraction is 8%, while the parallel fraction is 92%.

When the processor count increases from 512 to 2,048, the scaled speedup increases significantly. Efficiency approaches 92%.

This shows that the simulation can maintain high theoretical efficiency under Gustafson's weak-scaling assumptions.
