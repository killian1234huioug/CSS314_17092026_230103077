# Task 19: Monte Carlo Path Simulation — The Ideal Weak Scaling Model

## 1. Introduction

Monte Carlo simulation uses random samples to estimate numerical results. Each simulation path can usually be calculated independently. Therefore, Monte Carlo is an example of an embarrassingly parallel workload.

Under ideal weak scaling, every processor performs the same amount of work. When the processor count increases, the total number of simulation paths also increases.

## 2. Ideal Weak Scaling

Let:

- \(P\) be the number of processors;
- \(k\) be the number of paths per processor;
- \(W(k)\) be the computation time for one processor.

The total number of paths is:

\[
N=kP
\]

If every processor performs its work independently, the computation time is approximately:

\[
T(P)=W(k)
\]

Therefore, the ideal weak-scaling runtime remains constant as \(P\) increases.

In the ideal case, the serial fraction is:

\[
\alpha=0
\]

Gustafson's speedup becomes:

\[
S(P)=P
\]

## 3. Sources of Non-Zero Serial Fraction

Real Monte Carlo engines have some overhead.

### 3.1 Random Number Generator Management

Each processor needs a random number stream. The program must initialize seeds and make sure that streams do not produce unwanted overlap.

Seed initialization and stream management introduce overhead. If all processors depend on one central random number generator, contention may also occur.

### 3.2 Reduction Operations

After calculating local results, processors must combine their partial sums or averages.

For example, each processor calculates a local sum. The program then combines these sums to calculate the final result.

This operation requires communication and synchronization.

### 3.3 Output Serialization

The program may need to collect and write results to a file. If all processors send data to one output process, this process can become a bottleneck.

These operations create a non-zero effective serial fraction.

## 4. Impact of Tree Reductions

A tree reduction combines partial results in multiple stages.

For \(P\) processors, a balanced binary reduction tree has approximately:

\[
L=\lceil\log_2 P\rceil
\]

communication stages.

For example:

| Processors \(P\) | Reduction stages |
|---:|---:|
| 2 | 1 |
| 4 | 2 |
| 8 | 3 |
| 16 | 4 |
| 32 | 5 |
| 64 | 6 |

If each stage has communication cost \(t_c\), the approximate reduction time is:

\[
T_{\text{reduce}}\approx t_c\log_2 P
\]

The total runtime can be modeled as:

\[
T(P)=T_{\text{local}}+t_c\log_2 P+T_{\text{other}}
\]

Here, \(T_{\text{local}}\) is the fixed local simulation time, and \(T_{\text{other}}\) includes seed management and output overhead.

## 5. Effect on Gustafson's Speedup

An effective serial fraction can be estimated as:

\[
\alpha(P)=\frac{T_{\text{overhead}}(P)}
{T_{\text{local}}+T_{\text{overhead}}(P)}
\]

Gustafson's formula is:

\[
S(P)=P-\alpha(P)(P-1)
\]

As reduction overhead increases with \(\log P\), the effective serial fraction may also increase. Consequently, speedup becomes lower than the ideal value \(P\).

A numerical value for \(\alpha\) cannot be determined without measurements of the actual simulation and communication times.

## 6. Conclusion

Monte Carlo simulation is close to an ideal weak-scaling workload because individual paths are independent. However, random number management, tree reductions, and output serialization introduce overhead. A balanced reduction tree reduces the number of communication stages to approximately \(\log_2 P\), but it does not remove communication costs completely.
