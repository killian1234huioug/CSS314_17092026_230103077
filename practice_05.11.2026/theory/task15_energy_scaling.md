# Task 15 — Energy and Power Proportionality under Scaled Speedup

## 1. Introduction

Energy consumption is an important factor in parallel computing.

Adding more processors can increase performance, but it can also increase total power consumption.

This task compares energy consumption for one processor and multiple processors under weak scaling.

## 2. Power and Energy Formulas

Assume each processor consumes:

- \(P_{\text{idle}}\): static power.
- \(P_{\text{active}}\): additional dynamic power during computation.

The total power of one active processor is:

\[
P_1=P_{\text{idle}}+P_{\text{active}}
\]

Energy is calculated as power multiplied by execution time:

\[
E=P_{\text{total}}\cdot T
\]

## 3. Energy Consumption on One Processor

For one processor running for time \(T\):

\[
E_1=(P_{\text{idle}}+P_{\text{active}})T
\]

## 4. Energy Consumption on P Processors

If all \(P\) processors are active and each consumes the same power:

\[
P_{\text{total}}=
P(P_{\text{idle}}+P_{\text{active}})
\]

For the same execution time \(T\):

\[
E_P=P(P_{\text{idle}}+P_{\text{active}})T
\]

Therefore:

\[
\boxed{E_P=P\cdot E_1}
\]

Under these assumptions, using \(P\) active processors for the same time consumes \(P\) times more energy than using one processor.

## 5. Energy and Weak Scaling

Under Gustafson's Law, the problem size increases with the number of processors while execution time remains approximately constant.

This means that:
- More processors can complete more work in the same time.
- Total power consumption usually increases.
- Total energy consumption can also increase.
- Energy efficiency depends on how much useful work is completed.

A useful measure is energy per unit of useful work:

\[
E_{\text{work}}=
\frac{\text{Total Energy}}{\text{Useful Work}}
\]

If useful work grows proportionally with processor count, energy per unit of work may remain approximately constant under the simplified assumptions.

## 6. Conclusion

Using more processors under weak scaling increases the amount of useful work, but it also increases total power consumption.

If every processor consumes the same active power for the same execution time, total energy grows proportionally with processor count.

Therefore, high speedup does not automatically mean lower energy consumption. Parallel systems should consider both performance and energy efficiency.
