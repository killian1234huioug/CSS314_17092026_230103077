# Task 06 — Parametric Speedup Across Processor Densities

## 1. Given Information

The serial fraction is:

\[
\alpha=0.04
\]

The processor counts are:

\[
P=16,64,256,1024,4096,16384
\]

We use Gustafson's scaled speedup formula:

\[
S(P)=P-\alpha(P-1)
\]

Parallel efficiency is:

\[
E(P)=\frac{S(P)}{P}
\]

## 2. Calculations

For \(P=16\):

\[
S(16)=16-0.04(16-1)=15.4
\]

\[
E(16)=\frac{15.4}{16}=0.9625=96.25\%
\]

For \(P=64\):

\[
S(64)=64-0.04(63)=61.48
\]

\[
E(64)=\frac{61.48}{64}=96.0625\%
\]

For \(P=256\):

\[
S(256)=256-0.04(255)=245.8
\]

\[
E(256)=\frac{245.8}{256}=96.015625\%
\]

For \(P=1024\):

\[
S(1024)=1024-0.04(1023)=983.08
\]

\[
E(1024)=\frac{983.08}{1024}=96.00390625\%
\]

For \(P=4096\):

\[
S(4096)=4096-0.04(4095)=3932.2
\]

\[
E(4096)=\frac{3932.2}{4096}=96.00097656\%
\]

For \(P=16384\):

\[
S(16384)=16384-0.04(16383)=15728.68
\]

\[
E(16384)=\frac{15728.68}{16384}=96.00024414\%
\]

## 3. Results Table

| Processors (P) | Speedup S(P) | Efficiency |
|---:|---:|---:|
| 16 | 15.40 | 96.25% |
| 64 | 61.48 | 96.0625% |
| 256 | 245.80 | 96.0156% |
| 1,024 | 983.08 | 96.0039% |
| 4,096 | 3,932.20 | 96.0010% |
| 16,384 | 15,728.68 | 96.0002% |

## 4. Evaluation

As the processor count increases, the scaled speedup also increases.

Efficiency decreases slightly and approaches:

\[
E_{\infty}=1-\alpha=1-0.04=0.96
\]

Therefore, the limiting efficiency is 96%.

This happens because the serial fraction remains constant while the parallel workload grows with the processor count.

## 5. Conclusion

The results show that Gustafson's Law supports high efficiency for large processor counts when the serial fraction is small. In this example, efficiency approaches 96% even when the system uses 16,384 processors.
