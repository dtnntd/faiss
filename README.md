# FAISS Demo - Hướng dẫn và Ví dụ Minh họa

Dự án này cung cấp các ví dụ minh họa chi tiết về cách sử dụng FAISS (Facebook AI Similarity Search) cho tìm kiếm vector và xây dựng vector database.

## Giới thiệu về FAISS

FAISS là một thư viện mạnh mẽ cho việc tìm kiếm similarity và clustering của dense vectors. Được phát triển bởi Meta AI Research, FAISS có thể xử lý hàng tỷ vectors với hiệu suất cao.

**Ứng dụng:**
- Tìm kiếm hình ảnh tương tự
- Semantic search cho văn bản
- Recommendation systems
- Retrieval-Augmented Generation (RAG)
- Vector databases

## Cài đặt

### Sử dụng uv (khuyến nghị - nhanh hơn pip)

**Cài đặt uv** (nếu chưa có):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Cài đặt dependencies:**
```bash
# uv sẽ tự động tạo virtual environment và cài đặt
uv sync
```

**Chạy ví dụ:**
```bash
uv run python examples/01_basic_search.py
```

### Hoặc sử dụng pip (cách truyền thống)

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Cài đặt
pip install -r requirements.txt
```

### Kiểm tra cài đặt
```bash
uv run python -c "import faiss; print(f'FAISS version: {faiss.__version__}')"
```

## Cấu trúc dự án

```
faiss/
├── README.md                 # File này
├── requirements.txt          # Dependencies
├── examples/                 # Các ví dụ minh họa
│   ├── 01_basic_search.py   # Tìm kiếm vector cơ bản
│   ├── 02_ivf_index.py      # Inverted File Index
│   ├── 03_pq_compression.py # Product Quantization
│   ├── 04_hnsw_graph.py     # HNSW Graph Index
│   └── 06_real_world.py     # Ứng dụng thực tế
├── data/                     # Dữ liệu mẫu (tự động tạo)
└── utils/                    # Công cụ hỗ trợ
    ├── data_generator.py    # Tạo dữ liệu test
    ├── benchmark.py         # Đo hiệu suất
    └── visualization.py     # Trực quan hóa
```

## Các ví dụ

### 01. Basic Search - Tìm kiếm cơ bản
**File:** [examples/01_basic_search.py](examples/01_basic_search.py)

Minh họa:
- Tạo vectors và xây dựng Flat index (brute-force)
- Tìm kiếm k-nearest neighbors
- So sánh kết quả với exhaustive search
- Đo thời gian và độ chính xác

```bash
python examples/01_basic_search.py
```

### 02. IVF Index - Inverted File Index
**File:** [examples/02_ivf_index.py](examples/02_ivf_index.py)

Minh họa:
- Clustering vectors với k-means
- Xây dựng IVF index với số lượng centroids khác nhau
- Tìm kiếm với tham số nprobe
- Trade-off giữa tốc độ và độ chính xác

```bash
python examples/02_ivf_index.py
```

### 03. Product Quantization - Nén vector
**File:** [examples/03_pq_compression.py](examples/03_pq_compression.py)

Minh họa:
- Scalar Quantization (SQ)
- Product Quantization (PQ)
- So sánh memory usage
- Đánh giá độ chính xác sau nén

```bash
python examples/03_pq_compression.py
```

### 04. HNSW Graph - Graph-based Search
**File:** [examples/04_hnsw_graph.py](examples/04_hnsw_graph.py)

Minh họa:
- Xây dựng Hierarchical Navigable Small World graph
- Tham số efConstruction và efSearch
- Benchmark với IVF
- Phù hợp cho datasets nhỏ-trung bình (<10M vectors)

```bash
python examples/04_hnsw_graph.py
```

### 06. Real-world Example - Ứng dụng thực tế
**File:** [examples/06_real_world.py](examples/06_real_world.py)

Minh họa:
- Tìm kiếm sản phẩm tương tự
- Semantic search
- Recommendation system đơn giản

```bash
python examples/06_real_world.py
```

## Các khái niệm chính

### 1. Index Types

| Index Type | Tốc độ | Bộ nhớ | Độ chính xác | Khi nào dùng |
|------------|--------|---------|--------------|--------------|
| **Flat** | Chậm | Cao | 100% | Dataset nhỏ, cần kết quả chính xác |
| **IVF** | Nhanh | Trung bình | 90-95% | Dataset lớn (>1M vectors) |
| **HNSW** | Rất nhanh | Cao | 95-99% | Dataset <10M, cần tốc độ cao |
| **PQ** | Trung bình | Rất thấp | 80-90% | Giới hạn bộ nhớ |

### 2. Metrics

- **L2 Distance**: Khoảng cách Euclidean, dùng cho most cases
- **Inner Product**: Cho cosine similarity (sau khi normalize)
- **Cosine Similarity**: Góc giữa các vectors

### 3. Trade-offs

FAISS cho phép cân bằng giữa:
- **Accuracy vs Speed**: IVF nprobe, HNSW efSearch
- **Accuracy vs Memory**: Compression methods (PQ, SQ)
- **Index time vs Search time**: HNSW efConstruction

## Best Practices

### Chọn Index phù hợp

```python
# Dataset nhỏ (<100K vectors)
index = faiss.IndexFlatL2(dimension)

# Dataset trung bình (100K - 1M)
index = faiss.IndexHNSWFlat(dimension, 32)

# Dataset lớn (>1M vectors)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Giới hạn bộ nhớ
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
```

### Tối ưu hiệu suất

1. **Normalize vectors** nếu dùng Inner Product
2. **Train index** với representative sample (>30 * nlist vectors)
3. **Tune parameters** với validation set
4. **Batch queries** để tận dụng SIMD/GPU

## Tài liệu tham khảo

- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Paper gốc](https://arxiv.org/abs/2401.08281)

## Troubleshooting

### Lỗi thường gặp

**1. ImportError: No module named 'faiss'**
```bash
pip install faiss-cpu
```

**2. IndexError khi search**
- Kiểm tra dimension của query vector
- Đảm bảo đã train index (với IVF, PQ)

**3. Kết quả kém chính xác**
- Tăng nprobe (IVF) hoặc efSearch (HNSW)
- Kiểm tra cách normalize vectors
- Tăng số lượng training vectors

**4. Search quá chậm**
- Giảm nprobe/efSearch
- Sử dụng compression (PQ)
- Cân nhắc GPU acceleration

## Liên hệ & Đóng góp

Nếu có câu hỏi hoặc muốn đóng góp, vui lòng tạo issue hoặc pull request.

## License

Các ví dụ này được cung cấp cho mục đích học tập và nghiên cứu.
