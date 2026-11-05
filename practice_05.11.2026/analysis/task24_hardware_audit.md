# Task 24: Hardware Audit and Empirical vs. Theoretical Scaling

## 1. Hardware Architecture

The benchmark is executed on an Apple MacBook Air with an Apple M3 processor.

| Component | Specification |
|---|---|
| CPU | Apple M3 |
| CPU architecture | ARM-based Apple Silicon |
| Logical cores | 8 |
| Core configuration | 4 performance cores + 4 efficiency cores |
| Operating system | macOS |
| Memory | Unified memory architecture |

The Apple M3 uses unified memory shared by the CPU and other system components. Its memory architecture differs from a traditional desktop system with separate CPU memory channels.

## 2. Scaling Comparison

The experiment compares four types of results:

1. Ideal linear speedup;
2. Theoretical Gustafson speedup;
3. Measured empirical speedup;
4. Theoretical Amdahl speedup.

The theoretical Gustafson curve uses the empirical serial fraction obtained from Task 23.

Gustafson's formula:

\[
S_G(P)=P-\alpha(P-1)
\]

Amdahl's formula:

\[
S_A(P)=\frac{1}{\alpha+(1-\alpha)/P}
\]

Ideal linear speedup:

\[
S(P)=P
\]

## 3. Standard Error Bars

The benchmark is repeated five times for each thread count.

The mean runtime is:

\[
\bar{T}=\frac{1}{5}\sum_{i=1}^{5}T_i
\]

The standard error is:

\[
SE=\frac{s}{\sqrt{5}}
\]

where \(s\) is the sample standard deviation.

Error bars show the uncertainty in the measured mean runtime.

## 4. Reasons for Performance Differences

Measured results may differ from theoretical predictions for several reasons.

### Thermal Throttling

During long or intensive workloads, the processor may reduce its operating frequency to control temperature. This can increase execution time.

### Performance and Efficiency Cores

The Apple M3 has performance and efficiency cores. These cores have different processing capabilities. Therefore, adding more threads does not always produce equal performance improvements.

### Memory Bandwidth

The workload may become limited by memory access instead of computation. In this situation, additional threads may not improve performance.

### Scheduling and Synchronization

Operating-system scheduling, thread management, and synchronization can introduce additional overhead.

### NUMA Boundaries

NUMA effects are relevant mainly to systems with non-uniform memory access. The Apple M3 uses unified memory, so traditional multi-socket NUMA boundaries are not a defining feature of this system.

## 5. Conclusion

Theoretical scaling models provide an ideal reference for parallel performance. Real measurements can differ because of processor architecture, memory bandwidth, scheduling, synchronization, and thermal behavior.

The comparison between ideal, Gustafson, Amdahl, and empirical results helps explain the limits of weak scaling on real hardware.
