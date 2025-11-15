This is the outline of the main ideas from "The Faiss Library," including defining formulas, as requested.

## Outline: The Faiss Library

### I. Introduction and Core Concepts
A. **Faiss's Purpose:** A dedicated library for **vector similarity search**, forming a core functionality of vector databases.

B. **Context:** Similarity search relies on *embeddings* (vector representations) where locality encodes semantics. The distance between embeddings reflects the similarity between media items.

C. **Faiss's Role:** Faiss is a toolkit of indexing methods and primitives (search, cluster, compress, transform) dedicated to **Approximate Nearest Neighbor Search (ANNS)**.

D. **Basic Index Operations:**
1. **Nearest Neighbor Search (NN):** Computes the closest database vector $x_i$ to a query vector $q \in \mathbb{R}^d$.
$$\mathrm{n} = \underset{i=1..N}{\mathrm{argmin}}\|q-x_{i}\|. \quad (1)$$
2. **$k$-Nearest Neighbor Search ($k$-NN):** Returns the indices of the $k$ smallest distances.
$$(n_{1},...,n_{k},*,...,*)=\underset{i=1..N}{\mathrm{argsort}}\|q-x_{i}\|, \quad (2)$$
3. **Range Search:** Finds all elements within a radius $\varepsilon$.
$$R=\{n=1..N\mathrm{\ s.t.\ }\|q-x_{n}\|\leq\varepsilon\}, \quad (3)$$

### II. Performance Axes and Trade-offs (Section 3)
A. **Core Constraints:** Faiss manages the fundamental trade-off between **speed**, **memory usage**, and **accuracy**. These are referred to as the *active constraints*.

B. **Accuracy Metrics:**
1. **$k$-NN Accuracy:** Measured using **"$n$-recall@$k$"** (fraction of $n$ ground-truth nearest neighbors found in the $k$ first results).
2. **Range Search Accuracy:** Evaluated using standard retrieval metrics: **precision** ($P$) and **recall** ($R$).
3. **Vector Compression Quality:** Measured by the **Mean Squared Error (MSE)** between the original vector ($x$) and the reconstructed vector ($D(C(x))$).
$$\mathrm{MSE}=\mathbb{E}_{x}\big[\|D(C(x))-x\|_{2}^{2}\big]. \quad (4)$$

C. **Resource Metrics:**
1. **Speed:** Measured primarily by **search time**.
2. **Memory Usage:** Includes **constant memory overhead** (for training data) and **per-vector memory overhead** (e.g., graph edges).
3. **Index Building Time:** Decomposed into **training time** and **addition time per vector**.

D. **Exploring Trade-off Settings:**
1. Faiss identifies **Pareto-optimal settings** ($\mathcal{P}^{*}$), which are the fastest for a given accuracy or the most accurate for a given time budget.
$$\mathcal{P}^{*}=\big\{\pi\in\mathcal{P}|\nexists\pi^{\prime}\in\mathcal{P}\mathrm{\ s.t.\ }(S(\pi^{\prime}),A(\pi^{\prime}))>(S(\pi),A(\pi))\big\}. \quad (5)$$
2. Search-time hyperparameters ($\pi$) exhibit a monotonic effect on speed ($S$) and accuracy ($A$):
$$\pi^{\prime}\geq\pi\Rightarrow\left\{\begin{array}{c}S(\pi^{\prime})\leq S(\pi),\\ A(\pi^{\prime})\geq A(\pi).\\ \end{array}\right. \quad (6)$$

E. **Refining (IndexRefine):** Combining a fast, inaccurate index (for a shortlist) with a slower, more accurate search (re-ranking) to improve **accuracy** while maintaining overall **speed**.

### III. Vector Compression Techniques (Section 4)
A. **Purpose:** Reduce **memory usage** by converting a vector $x$ to an integer code $C(x)$ and using a decoder $D$ to reconstruct an approximation $x'$.

B. **Asymmetric Distance Computation (ADC):** Search performed in the compressed domain without compressing the query vector $q$, which is generally more **accurate**.
$$n=\underset{i=1..N}{\mathrm{argmin}}\|q-D(C(x_{i}))\|=\underset{i=1..N}{\mathrm{argmin}}\|q-D(C_{i})\|, \quad (10)$$

C. **Key Quantizers (Codecs):**
1. **k-means (Kmeans):** High **accuracy** but impractical for codes larger than $\approx 3$ bytes due to exponential **memory usage** and **encoding complexity**.
2. **Scalar Quantizers (SQ):** Encode each dimension independently (e.g., LSH, SQ8, SQfp16).
3. **Multi-codebook Quantizers (MCQ):** Product Quantizer (PQ), Residual Quantizer (RQ), and Local Search Quantizer (LSQ).

D. **Additive Quantizers (RQ, LSQ):**
1. **Decoding:** The reconstruction $x'$ is the sum of entries from $M$ codebooks ($T_m$).
$$x^{\prime}=D(C(x))=T_{1}[c_{1}]+...+T_{M}[c_{M}]. \quad (12)$$
2. **Compressed-domain Search (Speed):** Inner product search (IP) is accelerated using Look-Up Tables (LUTs).
$$\langle q,x^{\prime}\rangle=\sum_{m=1}^{M}\langle T_{m}[c_{m}],q\rangle=\sum_{m=1}^{M}\mathrm{LUT}_{m}[c_{m}]. \quad (14)$$
3. **L2 Distance Decomposition:** L2 distances are computed using the inner product:
$$\|q-x^{\prime}\|^{2}=\|q\|^{2}+\|x^{\prime}\|^{2}-2\langle q,x^{\prime}\rangle. \quad (15)$$

### IV. Non-Exhaustive Search (Pruning) (Section 5)
A. **Goal:** Crucial for datasets $N > 10k$ to achieve fast search by quickly focusing on a subset of vectors.

B. **Inverted Files (IVF):** Clusters vectors into $K_{\mathrm{IVF}}$ lists using a coarse quantizer. Only $P_{\mathrm{IVF}}$ (nprobe) nearest clusters are visited during search.
1. **Distance Computation Estimate (Simplest Case):** The search time is minimized when $K_{\mathrm{IVF}}$ is set proportional to $\sqrt{N}$.
$$N_{\mathrm{distances}}={K_{\mathrm{IVF}}}+{P_{\mathrm{IVF}}}\times N/{K_{\mathrm{IVF}}} \quad (18)$$
2. **Encoding Residuals:** Compression is generally more **accurate** when applied to the residual vector (the difference between the original vector and its closest centroid).

C. **Graph-based Algorithms (HNSW, NSG):**
1. **Mechanism:** Build a directed graph; search involves following edges toward nodes closest to the query.
2. **Trade-off:** The number of outgoing edges per node adjusts the trade-off between **search speed** and **memory usage**. HNSW supports on-the-fly vector addition, while NSG generally achieves better trade-offs at the cost of longer **build time**.

### V. Design Principles and Optimization
A. **Index Composition:** Faiss indexes are often built as a chain of components (e.g., `PCA160,IVF20000_HNSW,PQ20x10,RFlat`) using the `index_factory` function.

B. **Optimization Strategy:** Code is optimized only when **runtime** is critical, ensuring correctness by comparing optimized versions against non-optimized ones.

C. **Hardware Optimization:**
1. **CPU Vectorization (SIMD):** Faiss exploits SIMD (AVX/AVX2/AVX512) to maximize arithmetic throughput, often adapting data layout (e.g., `FastScan` variants).
2. **GPU Faiss:** Leverages the throughput-optimized architecture of GPUs, excelling in distance computation. It addresses the sequential nature of ranking by using an efficient GPU $k$-selection algorithm.

D. **Interfacing:** Faiss is designed to be easily embedded, providing C++ core, Python wrappers (via SWIG), and a pure C API. The library is modular and provides low-level APIs for storage control (e.g., `InvertedLists` objects) for vector database integration.