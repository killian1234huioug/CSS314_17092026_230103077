Task 01 — Strong Scaling vs. Weak Scaling

Amdahl's Law (1967)

Amdahl's Law focuses on a fixed-size problem. The total workload does not change when we increase the number of processors.

The formula is:

[
S_A(P)=\frac{1}{s+\frac{1-s}{P}}
]

Where:

(P) is the number of processors.

(s) is the serial fraction.

(1-s) is the parallel fraction.

When the number of processors approaches infinity, the parallel part becomes very small. However, the serial part remains. Therefore, the maximum speedup is limited:

[
\lim_{P\to\infty}S_A(P)=\frac{1}{s}
]

This is called strong scaling or fixed-size scaling.

Gustafson's Law (1988)

Gustafson's Law focuses on increasing the problem size when more processors are available.

The main idea is to keep execution time approximately constant while processing more data.

The formula is:

[
S_G(P)=P-\alpha(P-1)
]

Where:

(P) is the number of processors.

(\alpha) is the serial fraction measured on the parallel system.

As the number of processors increases, the workload can also increase. Therefore, speedup can grow almost linearly.

This is called weak scaling or fixed-time scaling.

Main Differences

Amdahl's Law

Gustafson's Law

Fixed problem size

Increasing problem size

Strong scaling

Weak scaling

Focuses on reducing execution time

Focuses on processing more work

Speedup has a maximum limit

Speedup can grow almost linearly

Conclusion

Amdahl's Law explains how much faster the same problem can run with more processors. Gustafson's Law explains how much more work can be completed in approximately the same time. Both models are important for understanding parallel computing.
