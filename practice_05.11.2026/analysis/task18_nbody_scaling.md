# Task 18: N-Body Gravitational Problem Scaling

## 1. Introduction

The N-body problem simulates the gravitational interaction between particles. Two common algorithms are the direct all-pairs method and the Barnes-Hut tree-code method. They have different computational complexities, so their scaling behavior is also different.

## 2. Direct All-Pairs Method

The direct method calculates the gravitational force between every pair of particles.

Its computational complexity is:

\[
T(N)=O(N^2)
\]

If the work is distributed ideally across \(P\) processors, the parallel computation time is approximately:

\[
T_p(N,P)=O\left(\frac{N^2}{P}\right)
\]

To maintain constant computation time as the number of processors increases, we require:

\[
\frac{N^2}{P}=\text{constant}
\]

Therefore:

\[
N^2\propto P
\]

\[
\boxed{N\propto P^{1/2}}
\]

This means that the particle count can grow approximately with the square root of the processor count while keeping the ideal computation time constant.

## 3. Barnes-Hut Tree-Code Method

Barnes-Hut groups distant particles into tree nodes and approximates their gravitational effects. This reduces the typical computational complexity to:

\[
T(N)=O(N\log N)
\]

With ideal parallel distribution:

\[
T_p(N,P)=O\left(\frac{N\log N}{P}\right)
\]

For constant computation time:

\[
\frac{N\log N}{P}=\text{constant}
\]

Thus:

\[
N\log N\propto P
\]

A useful asymptotic approximation is:

\[
\boxed{N=O\left(\frac{P}{\log P}\right)}
\]

This is an approximate growth relationship, not an exact solution.

## 4. Comparison

| Property | Direct Method | Barnes-Hut |
|---|---|---|
| Complexity | \(O(N^2)\) | \(O(N\log N)\) |
| Ideal constant-time particle growth | \(N\propto\sqrt P\) | Approximately \(N\log N\propto P\) |
| Main advantage | Simple and accurate pair calculations | Lower computational complexity |
| Main limitation | Many pair interactions | Tree construction and irregular communication |

## 5. Serial-to-Parallel Ratio

The direct method has a large amount of pairwise computation that can be distributed among processors. Under ideal parallelization, increasing \(P\) and \(N\) according to \(N\propto\sqrt P\) keeps the parallel computation time approximately constant.

Barnes-Hut performs less total work for large \(N\), but tree construction, tree traversal, and communication introduce additional overhead. These operations may become important as \(P\) increases.

The serial-to-parallel ratio cannot be calculated as one exact value from the complexity classes alone. It depends on the implementation, communication cost, tree construction, and load balancing.

## 6. Conclusion

The direct method supports particle growth proportional to \(\sqrt P\) under ideal constant-time scaling. Barnes-Hut supports a larger particle count, approximately satisfying \(N\log N\propto P\). However, its tree operations and communication can reduce the expected scaling efficiency.
