# Task 07 — Reverse Engineering Maximum Tolerable Serial Overhead

## 1. Given Information

Number of processors:

\[
P=8192
\]

Required scaled speedup:

\[
S(P)\geq7500
\]

Gustafson's formula:

\[
S(P)=P-\alpha(P-1)
\]

We need to find the maximum allowable serial fraction \(\alpha_{\max}\).

## 2. Algebraic Derivation

Substitute the given values:

\[
8192-\alpha(8192-1)\geq7500
\]

\[
8192-8191\alpha\geq7500
\]

Move the terms:

\[
8191\alpha\leq8192-7500
\]

\[
8191\alpha\leq692
\]

Divide both sides by 8191:

\[
\alpha\leq\frac{692}{8191}
\]

Therefore:

\[
\boxed{\alpha_{\max}=0.084483}
\]

As a percentage:

\[
\boxed{\alpha_{\max}\approx8.4483\%}
\]

## 3. Verification

Using the maximum serial fraction:

\[
S(8192)=8192-0.08448297(8191)
\]

\[
S(8192)\approx7500
\]

The required speedup is achieved.

## 4. Conclusion

The maximum allowable serial fraction is approximately 8.4483%.

To achieve a scaled speedup of at least 7,500 on 8,192 processors, the serial fraction must not exceed this value. A lower serial fraction will provide a higher speedup.
