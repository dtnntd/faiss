# FAISS Quickstart Guide

## Cài đặt nhanh

```bash
# Sử dụng uv (nhanh)
uv sync

# Hoặc pip
pip install -r requirements.txt
```

## Chạy ví dụ

```bash
# Example 01: Basic vector search
uv run python examples/01_basic_search.py

# Example 02: IVF index (fast search)
uv run python examples/02_ivf_index.py

# Example 03: Product Quantization (compression)
uv run python examples/03_pq_compression.py

# Example 04: HNSW graph-based search
uv run python examples/04_hnsw_graph.py

# Example 06: Real-world applications
uv run python examples/06_real_world.py
```

## Chọn Index nhanh

| Dataset Size | Recommended Index | Command |
|--------------|-------------------|---------|
| < 10K | `IndexFlatL2` | `faiss.IndexFlatL2(d)` |
| 10K - 100K | `IndexHNSWFlat` | `faiss.IndexHNSWFlat(d, 32)` |
| 100K - 1M | `IndexIVFFlat` | `faiss.IndexIVFFlat(q, d, 100)` |
| > 1M | `IndexIVFPQ` | `faiss.IndexIVFPQ(q, d, 100, 8, 8)` |

## Code template cơ bản

```python
import numpy as np
import faiss

# 1. Tạo data
dimension = 128
n_vectors = 10000
vectors = np.random.random((n_vectors, dimension)).astype('float32')

# 2. Tạo index
index = faiss.IndexFlatL2(dimension)

# 3. Thêm vectors
index.add(vectors)

# 4. Search
k = 10  # số neighbors
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k)

print(f"Top {k} nearest neighbors: {indices[0]}")
```

## IVF Index (cho dataset lớn)

```python
# 1. Tạo quantizer
quantizer = faiss.IndexFlatL2(dimension)

# 2. Tạo IVF index
nlist = 100  # số clusters
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# 3. Train
index.train(vectors)

# 4. Add
index.add(vectors)

# 5. Search với nprobe
index.nprobe = 10  # search trong 10 clusters
distances, indices = index.search(query, k)
```

## HNSW (cho accuracy cao)

```python
# 1. Tạo HNSW
M = 32  # connections
index = faiss.IndexHNSWFlat(dimension, M)

# 2. Set build parameters
index.hnsw.efConstruction = 40

# 3. Add (không cần train!)
index.add(vectors)

# 4. Search với efSearch
index.hnsw.efSearch = 64
distances, indices = index.search(query, k)
```

## Save/Load Index

```python
# Save
faiss.write_index(index, "my_index.faiss")

# Load
index = faiss.read_index("my_index.faiss")
```

## Tips

- **Normalize vectors** cho cosine similarity: `faiss.normalize_L2(vectors)`
- **Batch search** nhanh hơn single query
- **Tune parameters** trên validation set
- **Monitor recall** để đảm bảo quality

## Tài liệu đầy đủ

Xem [README.md](README.md) để hiểu chi tiết về từng index type và parameters.