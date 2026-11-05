# Task 08 — Cross-Over Point Determination (Amdahl vs. Gustafson)

## 1. Given Information

Serial fraction:

\[
s=\alpha=0.08
\]

Number of processors:

\[
P=128
\]

We compare Amdahl's Law and Gustafson's Law.

## 2. Amdahl's Law Calculation

Formula:

\[
S_A(P)=\frac{1}{s+\frac{1-s}{P}}
\]

Substitute the values:

\[
S_A(128)=\frac{1}{0.08+\frac{0.92}{128}}
\]

\[
S_A(128)=\frac{1}{0.08+0.0071875}
\]

\[
S_A(128)=\frac{1}{0.0871875}
\]

Therefore:

\[
\boxed{S_A(128)\approx11.47}
\]

## 3. Gustafson's Law Calculation

Formula:

\[
S_G(P)=P-\alpha(P-1)
\]

Substitute the values:

\[
S_G(128)=128-0.08(128-1)
\]

\[
S_G(128)=128-0.08(127)
\]

\[
S_G(128)=128-10.16
\]

Therefore:

\[
\boxed{S_G(128)=117.84}
\]

## 4. Percentage Discrepancy

The absolute difference is:

\[
D=117.84-11.4695
\]

\[
\boxed{D\approx106.37}
\]

Using Amdahl's result as the reference:

\[
D_A=
\frac{117.84-11.4695}{11.4695}\times100
\]

\[
\boxed{D_A\approx927.42\%}
\]

Using Gustafson's result as the reference:

\[
D_G=
\frac{117.84-11.4695}{117.84}\times100
\]

\[
\boxed{D_G\approx90.27\%}
\]

The percentage depends on the chosen reference value. The first calculation expresses how much larger Gustafson's prediction is than Amdahl's prediction.

## 5. Comparison Table

| Model | Speedup |
|---|---:|
| Amdahl | 11.47 |
| Gustafson | 117.84 |
| Absolute difference | 106.37 |

## 6. Explanation

Amdahl's Law assumes a fixed problem size. The serial part limits the maximum speedup.

Gustafson's Law assumes that the problem size increases with the number of processors. Therefore, more processors can perform more total work in approximately the same time.

This explains why Gustafson's predicted speedup is much higher in this example.

## 7. Conclusion

For 128 processors and an 8% serial fraction, Amdahl's Law predicts a speedup of approximately 11.47, while Gustafson's Law predicts 117.84.

The difference comes from the workload assumptions. Amdahl considers fixed-size scaling, while Gustafson considers weak scaling.
