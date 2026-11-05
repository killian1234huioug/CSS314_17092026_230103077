# Task 20: Parallel Graph Traversal and Irregular Workloads

## 1. Introduction

Graph analytics algorithms such as Breadth-First Search (BFS) and PageRank are difficult to scale efficiently. Unlike dense matrix operations, graph algorithms often have irregular memory access and an uneven amount of work between processors.

Power-law graphs make this problem more difficult because a small number of vertices may have a very large number of connections.

## 2. Breadth-First Search (BFS)

BFS explores a graph level by level, starting from a source vertex.

The algorithm processes the current frontier and discovers neighboring vertices. In parallel BFS, different processors explore different parts of the frontier.

However, several problems can occur:

- **Frontier imbalance:** Some processors receive more vertices or edges than others.
- **Irregular memory access:** Neighboring vertices may be stored in different memory locations.
- **Synchronization:** Processors must coordinate when they discover and mark visited vertices.
- **Communication:** Information about newly discovered vertices may need to be shared.

As a result, some processors may finish early while others continue working.

## 3. PageRank

PageRank calculates the importance of vertices using their incoming links.

The algorithm repeatedly updates vertex scores based on the scores of neighboring vertices.

In parallel PageRank, processors calculate updates for different vertices. However, they still need to exchange information about neighboring vertices.

The main challenges include:

- Irregular access to graph data;
- Communication between processors;
- Synchronization during score updates;
- Uneven workloads caused by different vertex degrees.

These overheads can increase as the graph and processor count grow.

## 4. Why Power-Law Graphs Are Difficult

In a power-law graph, most vertices have few connections, while a small number of vertices have many connections.

These highly connected vertices can create bottlenecks.

For example, one processor may receive a vertex with thousands of edges, while another processor receives many vertices with only a few edges.

This creates load imbalance.

The system may also need additional synchronization when many processors access or update related graph information.

## 5. Comparison with Dense Linear Algebra

| Property | Dense Matrix Operations | Graph Analytics |
|---|---|---|
| Memory access | More regular | Irregular |
| Work distribution | Usually predictable | Often uneven |
| Communication | Can be structured | Data-dependent |
| Synchronization | Often easier to organize | Can be frequent |
| Weak scaling | More predictable | More difficult |

Dense matrix multiplication has regular data structures and predictable operations. This makes it easier to divide the workload equally between processors.

Graph algorithms depend on the graph structure. Therefore, increasing the number of processors does not always divide the work evenly.

## 6. Connection to Gustafson's Law

Gustafson's Law assumes that the parallel part of a workload can grow while the execution time remains approximately constant.

For graph processing, this assumption is difficult to achieve because:

1. More vertices and edges can increase communication.
2. High-degree vertices can create workload imbalance.
3. Irregular memory access can reduce processor utilization.
4. Synchronization can become a bottleneck.
5. Data-dependent operations make execution time less predictable.

Therefore, the effective serial fraction and communication overhead may increase with the processor count.

Gustafson's ideal speedup is:

\[
S(P)=P-\alpha(P-1)
\]

If the effective serial fraction \(\alpha\) increases, the achieved speedup becomes lower than the ideal linear speedup.

## 7. Conclusion

BFS and PageRank are harder to scale than regular dense matrix operations. Their performance depends on graph structure, memory access, communication, and load balancing. Power-law graphs create additional problems because a few highly connected vertices can become bottlenecks. These factors make ideal Gustafson weak scaling difficult to achieve in graph analytics.
