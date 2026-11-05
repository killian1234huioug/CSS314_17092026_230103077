# Task 02 — Derivation of Gustafson's Scaled Speedup

The total parallel execution time is normalized to 1:

\[
s+p=1
\]

Where:
- \(s\) is the serial execution time.
- \(p\) is the parallel execution time.

Therefore:

\[
p=1-s
\]

On P processors, the parallel part represents P times more work than one processor could perform during the same time.

The equivalent serial execution time is:

\[
T_1=s+Pp
\]

Since parallel execution time is normalized to 1, speedup is:

\[
S(P)=\frac{T_1}{T_P}
\]

Substituting:

\[
S(P)=s+Pp
\]

Now substitute \(p=1-s\):

\[
S(P)=s+P(1-s)
\]

Expanding:

\[
S(P)=s+P-Ps
\]

Rearranging:

\[
S(P)=P-s(P-1)
\]

If the serial fraction is represented by \(\alpha\), the final formula is:

\[
\boxed{S(P)=P-\alpha(P-1)}
\]

This is Gustafson's scaled speedup formula.

## Conclusion

Gustafson's Law shows that speedup depends on the number of processors and the serial fraction. A smaller serial fraction allows better parallel performance.
