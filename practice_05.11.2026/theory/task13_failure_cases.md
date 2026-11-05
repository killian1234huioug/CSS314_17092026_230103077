# Task 13 — Boundary Case: When Gustafson Fails

## 1. Introduction

Gustafson's Law assumes that the problem size can increase with the number of processors while execution time remains approximately constant.

However, some algorithms cannot maintain this condition. Communication, synchronization, and data dependencies can reduce parallel performance.

This section examines two examples.

## 2. Example 1: Breadth-First Search (BFS)

Breadth-First Search is used to explore graph structures.

Its performance depends on:
- The number of vertices and edges.
- The current search frontier.
- Memory access patterns.
- Communication between processors.

When the graph becomes larger, processors may need to exchange more information about visited vertices and frontier updates.

Some processors may receive much more work than others. This creates load imbalance.

As a result, communication and waiting time increase. The total execution time may no longer remain constant.

Therefore, BFS can fail to achieve ideal weak scaling.

## 3. Example 2: Parallel Sorting

Parallel sorting divides data between processors and sorts different parts simultaneously.

However, some sorting algorithms require processors to exchange data and synchronize during the sorting process.

When the dataset grows:
- More data must be exchanged.
- Global synchronization can become expensive.
- Processors may wait for slower processors.
- Data redistribution can increase overhead.

These costs can grow with the problem size and processor count.

Therefore, parallel sorting may not maintain constant execution time under weak scaling.

## 4. Why Gustafson's Assumption Fails

The main reason is that increasing the problem size can also increase communication and synchronization costs.

The parallel execution time can be represented as:

\[
T(P)=T_{\text{compute}}+
T_{\text{communication}}+
T_{\text{synchronization}}
\]

Even if computation is divided between more processors, communication and synchronization may increase.

This breaks the assumption of constant parallel execution time.

## 5. Conclusion

Gustafson's Law is useful for understanding weak scaling, but it does not guarantee ideal performance for every algorithm.

BFS and parallel sorting can experience communication overhead, synchronization delays, and load imbalance. These factors prevent execution time from remaining constant as the problem size grows.
