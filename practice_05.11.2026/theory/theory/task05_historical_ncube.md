# Task 05 — Deconstructing the "Amdahl's Pessimism" Fallacy

## 1. Historical Background

In 1988, John Gustafson, Gary Montry, and Robert Benner published a study about parallel computing at Sandia National Laboratories.

They used a 1024-processor hypercube computer called the nCUBE. Their research focused on three scientific applications:

- Wave mechanics.
- Fluid dynamics.
- Structural analysis.

The researchers wanted to demonstrate how large parallel systems could solve practical scientific problems efficiently.

## 2. Experimental Results

The original study reported two different types of speedup:

| Scaling model | Measured speedup |
|---|---:|
| Fixed-size problem | 502–637 times |
| Scaled-size problem | 1009–1020 times |

The scaled speedup was higher because the problem size increased with the number of processors.

For the scaled problems, the researchers achieved approximately 98.7% or better parallel efficiency in their reported applications.

## 3. Why Amdahl's Law Predicted a Limitation

Amdahl's Law assumes that the total problem size remains fixed.

Its formula is:

\[
S_A(P)=\frac{1}{s+\frac{1-s}{P}}
\]

For example, if the serial fraction is 0.4%:

\[
s=0.004
\]

With 1024 processors:

\[
S_A(1024)=
\frac{1}{0.004+\frac{0.996}{1024}}
\]

\[
\boxed{S_A(1024)\approx 199.2}
\]

Under these assumptions, Amdahl's Law predicts a maximum speedup of:

\[
S_{\max}=\frac{1}{0.004}=250
\]

Therefore, a speedup above 1000 would not be possible under this fixed-size model.

## 4. How Gustafson's Law Explained the Results

Gustafson's Law uses a different assumption. It allows the problem size to increase when more processors are added.

The scaled speedup formula is:

\[
S_G(P)=P-\alpha(P-1)
\]

For 1024 processors, the measured speedup was between 1009 and 1020.

This result was possible because the researchers increased the problem size while keeping the parallel execution time approximately constant.

More processors were used to solve a larger scientific problem instead of only solving the original problem faster.

## 5. Techniques Used by the Researchers

The original research discussed several techniques for improving parallel performance:

- Reducing communication overhead.
- Reducing serial operations.
- Improving load balance between processors.
- Overlapping communication with computation.
- Using efficient parallel algorithms.

These techniques helped reduce the time that processors spent waiting for other processors.

## 6. Important Difference Between the Models

Amdahl's Law measures performance for a fixed-size problem. Gustafson's Law measures performance when the problem size grows with the processor count.

The nCUBE experiment showed that a large parallel computer could achieve more than 1000 times speedup for scaled scientific problems.

The original paper also reported that the serial fraction for fixed-size applications was approximately 0.0006–0.001. For scaled problems, the equivalent serial fraction was reduced to approximately 3–10 parts per million.

The 0.4% value mentioned in the practice task is not the serial fraction reported for these scaled experiments in the original paper.

## 7. Conclusion

The nCUBE experiment demonstrated the practical value of Gustafson's Law. It did not disprove Amdahl's Law because the two models use different assumptions.

Amdahl's Law remains useful for fixed-size problems. Gustafson's Law is more suitable when scientists want to solve larger problems using more processors.

The main lesson is that adding processors can increase the amount of useful work, not only reduce execution time.
