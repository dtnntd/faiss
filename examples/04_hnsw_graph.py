"""
Example 04: HNSW (Hierarchical Navigable Small World) Graph
Ví dụ 4: HNSW - Graph-based search

Minh họa:
- Xây dựng HNSW graph
- Tham số efConstruction và efSearch
- So sánh với IVF
- Best for medium-sized datasets
"""

import numpy as np
import faiss
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import generate_random_vectors, generate_query_vectors
from utils.benchmark import (
    benchmark_index, print_index_info, compare_indexes,
    progressive_recall_benchmark, get_index_size
)


def main():
    print("="*70)
    print("FAISS Example 04: HNSW (Hierarchical Navigable Small World)")
    print("="*70)

    # ===== Cấu hình =====
    dimension = 128
    n_vectors = 50000        # HNSW tốt với dataset <10M
    n_queries = 100
    k = 10

    # HNSW parameters
    M = 32                   # Số connections mỗi node (higher = better accuracy, more memory)
    efConstruction = 40      # Size của dynamic list khi build (higher = better quality, slower build)
    efSearch = 16            # Size của dynamic list khi search (higher = better recall, slower search)

    print(f"\nCấu hình:")
    print(f"  Dimension: {dimension}")
    print(f"  Database size: {n_vectors:,} vectors")
    print(f"  Number of queries: {n_queries}")
    print(f"\nHNSW parameters:")
    print(f"  M (connections): {M}")
    print(f"  efConstruction: {efConstruction}")
    print(f"  efSearch: {efSearch}")

    # ===== Giải thích HNSW =====
    print(f"\n{'='*70}")
    print("HNSW - Cách hoạt động")
    print(f"{'='*70}")
    print("""
HNSW xây dựng multi-layer graph:

1. CẤU TRÚC:
   - Multiple layers của graphs
   - Layer cao: Sparse, long-range connections (highway)
   - Layer thấp: Dense, short-range connections (local streets)
   - Mỗi node có M connections với neighbors

2. BUILD PHASE (efConstruction):
   - Insert từng vector vào graph
   - Tìm efConstruction candidates ở mỗi layer
   - Connect với M nearest neighbors
   - efConstruction cao -> build chậm, quality tốt

3. SEARCH PHASE (efSearch):
   - Start từ entry point ở layer cao nhất
   - Greedy search xuống các layers thấp hơn
   - Maintain dynamic list of efSearch candidates
   - efSearch cao -> search chậm, recall tốt

4. PARAMETERS:
   M (connections per node):
     • Thường: 16-64
     • Cao -> accuracy tốt, memory nhiều
     • Khuyến nghị: 32

   efConstruction (build quality):
     • Thường: 40-500
     • Cao -> build chậm, quality tốt
     • Khuyến nghị: 40-200

   efSearch (search quality):
     • Thường: 16-512
     • Trade-off chính: accuracy vs speed
     • Tune runtime, không cần rebuild

Ưu điểm:
  ✓ Accuracy rất cao (>95%)
  ✓ Search rất nhanh
  ✓ Không cần training
  ✓ Có thể tune efSearch runtime

Nhược điểm:
  ✗ Memory usage cao (graph structure)
  ✗ Build time lâu với dataset lớn
  ✗ Không scale tốt > 10M vectors
    """)

    # ===== Tạo dữ liệu =====
    print(f"{'='*70}")
    print("Bước 1: Tạo dữ liệu")
    print(f"{'='*70}")

    print(f"Đang tạo {n_vectors:,} vectors...")
    database_vectors = generate_random_vectors(n_vectors, dimension)
    query_vectors = generate_query_vectors(n_queries, dimension)
    print(f"  ✓ Database vectors: {database_vectors.shape}")

    # ===== Ground Truth =====
    print(f"\n{'='*70}")
    print("Bước 2: Tạo Ground Truth")
    print(f"{'='*70}")

    print("Tạo Flat index...")
    index_flat = faiss.IndexFlatL2(dimension)
    index_flat.add(database_vectors)
    _, gt_indices = index_flat.search(query_vectors, k)
    print("  ✓ Ground truth ready")

    # ===== Xây dựng HNSW =====
    print(f"\n{'='*70}")
    print("Bước 3: Xây dựng HNSW Index")
    print(f"{'='*70}")

    print(f"\nTạo IndexHNSWFlat với M={M}, efConstruction={efConstruction}...")
    index_hnsw = faiss.IndexHNSWFlat(dimension, M)

    # Set efConstruction
    index_hnsw.hnsw.efConstruction = efConstruction

    print(f"\nĐang xây dựng HNSW graph...")
    print(f"  (Có thể mất vài giây cho {n_vectors:,} vectors...)")

    start_time = time.time()
    index_hnsw.add(database_vectors)
    build_time = time.time() - start_time

    print(f"  ✓ Build hoàn tất trong {build_time:.3f} giây")
    print(f"  ✓ Average time per vector: {build_time/n_vectors*1000:.3f} ms")

    # In thông tin
    print_index_info(index_hnsw, "\nHNSW Index")

    # HNSW specific info
    print(f"\n  HNSW Graph Info:")
    print(f"    Max level: {index_hnsw.hnsw.max_level}")
    print(f"    Entry point: {index_hnsw.hnsw.entry_point}")
    print(f"    efConstruction: {index_hnsw.hnsw.efConstruction}")
    print(f"    M (connections): {M}")

    # ===== So sánh với IVF =====
    print(f"\n{'='*70}")
    print("Bước 4: So sánh HNSW vs IVF")
    print(f"{'='*70}")

    print("\nTạo IVF index để so sánh...")
    nlist = int(np.sqrt(n_vectors))
    quantizer = faiss.IndexFlatL2(dimension)
    index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    print(f"Training IVF với nlist={nlist}...")
    index_ivf.train(database_vectors)
    index_ivf.add(database_vectors)
    index_ivf.nprobe = 10

    # Set efSearch cho HNSW
    index_hnsw.hnsw.efSearch = efSearch

    # Compare
    indexes = {
        'Flat': index_flat,
        f'HNSW (M={M}, ef={efSearch})': index_hnsw,
        f'IVF (nlist={nlist}, nprobe=10)': index_ivf,
    }

    compare_indexes(indexes, query_vectors, gt_indices, k)

    # ===== Trade-off efSearch =====
    print(f"\n{'='*70}")
    print("Bước 5: Trade-off efSearch (Speed vs Accuracy)")
    print(f"{'='*70}")

    print("\nefSearch càng lớn -> search chính xác hơn nhưng chậm hơn")

    efSearch_values = [4, 8, 16, 32, 64, 128, 256]

    results = progressive_recall_benchmark(
        index_hnsw,
        query_vectors,
        gt_indices,
        'efSearch',
        efSearch_values,
        k
    )

    print(f"\nNhận xét:")
    print(f"  - efSearch=16: Fast, recall ~90%")
    print(f"  - efSearch=64: Balanced, recall ~97%")
    print(f"  - efSearch=256: Slow, recall ~99%")
    print(f"\n  Có thể tune efSearch runtime mà không cần rebuild index!")

    # ===== Thử nghiệm M khác nhau =====
    print(f"\n{'='*70}")
    print("Bước 6: So sánh M values (connections)")
    print(f"{'='*70}")

    print("\nSo sánh different M values:")
    print(f"  (Cần rebuild index cho mỗi M)")

    print(f"\n{'M':<10} {'Build Time (s)':<20} {'Recall@10':<15} {'Size (MB)':<15}")
    print(f"{'-'*60}")

    M_values = [8, 16, 32, 64]

    for M_test in M_values:
        # Create index
        idx_test = faiss.IndexHNSWFlat(dimension, M_test)
        idx_test.hnsw.efConstruction = 40

        # Build
        start = time.time()
        idx_test.add(database_vectors)
        build_t = time.time() - start

        # Search với efSearch=64
        idx_test.hnsw.efSearch = 64
        _, pred_indices = idx_test.search(query_vectors, k)

        # Metrics
        from utils.benchmark import calculate_recall
        recall = calculate_recall(pred_indices, gt_indices, k)
        size_mb = get_index_size(idx_test)['size_mb']

        print(f"{M_test:<10} {build_t:<20.3f} {recall:<15.4f} {size_mb:<15.2f}")

    print(f"\nNhận xét:")
    print(f"  - M cao: Build chậm, memory nhiều, accuracy tốt")
    print(f"  - M thấp: Build nhanh, memory ít, accuracy thấp")
    print(f"  - Sweet spot: M=32 cho hầu hết use cases")

    # ===== Memory Analysis =====
    print(f"\n{'='*70}")
    print("Bước 7: Phân tích Memory Usage")
    print(f"{'='*70}")

    flat_size = get_index_size(index_flat)['size_mb']
    hnsw_size = get_index_size(index_hnsw)['size_mb']
    ivf_size = get_index_size(index_ivf)['size_mb']

    print(f"\nMemory Comparison:")
    print(f"  Flat:  {flat_size:.2f} MB (baseline)")
    print(f"  IVF:   {ivf_size:.2f} MB ({ivf_size/flat_size:.2f}x)")
    print(f"  HNSW:  {hnsw_size:.2f} MB ({hnsw_size/flat_size:.2f}x)")

    print(f"\nHNSW Memory breakdown:")
    print(f"  Vectors: {database_vectors.nbytes / (1024**2):.2f} MB")
    print(f"  Graph:   {(hnsw_size - flat_size):.2f} MB")
    print(f"  Estimated connections: ~{M * n_vectors * 2} (M * n * 2 directions)")

    # ===== Build Time Scaling =====
    print(f"\n{'='*70}")
    print("Bước 8: Build Time Scaling Analysis")
    print(f"{'='*70}")

    print(f"\nTest build time với different dataset sizes:")
    print(f"{'Vectors':<15} {'Build Time (s)':<20} {'Vectors/sec':<15}")
    print(f"{'-'*50}")

    sizes = [1000, 5000, 10000, 25000]

    for size in sizes:
        idx_test = faiss.IndexHNSWFlat(dimension, 32)
        idx_test.hnsw.efConstruction = 40

        start = time.time()
        idx_test.add(database_vectors[:size])
        build_t = time.time() - start

        throughput = size / build_t

        print(f"{size:<15,} {build_t:<20.3f} {throughput:<15.1f}")

    print(f"\nNhận xét:")
    print(f"  - Build time tăng ~linear với số vectors")
    print(f"  - Với dataset lớn (>10M), IVF thường nhanh hơn")

    # ===== Best Use Cases =====
    print(f"\n{'='*70}")
    print("Bước 9: HNSW vs IVF - Khi nào dùng gì?")
    print(f"{'='*70}")

    print(f"\n{'Metric':<25} {'HNSW':<25} {'IVF':<25}")
    print(f"{'-'*75}")
    print(f"{'Dataset size':<25} {'<10M vectors':<25} {'>1M vectors':<25}")
    print(f"{'Accuracy':<25} {'Rất cao (95-99%)':<25} {'Cao (90-95%)':<25}")
    print(f"{'Speed':<25} {'Rất nhanh':<25} {'Nhanh':<25}")
    print(f"{'Memory':<25} {'Cao (1.5-2x)':<25} {'Trung bình (1.1x)':<25}")
    print(f"{'Build time':<25} {'Lâu':<25} {'Trung bình':<25}")
    print(f"{'Training':<25} {'Không cần':<25} {'Cần':<25}")
    print(f"{'Runtime tuning':<25} {'efSearch':<25} {'nprobe':<25}")

    # ===== Tóm tắt =====
    print(f"\n{'='*70}")
    print("TÓM TẮT - HNSW")
    print(f"{'='*70}")

    print(f"\n✓ Parameters:")
    print(f"  - M (connections): 16-64, thường dùng 32")
    print(f"  - efConstruction: 40-200 (build quality)")
    print(f"  - efSearch: 16-512 (search quality, tune runtime)")

    print(f"\n✓ Khi nào dùng HNSW:")
    print(f"  - Dataset <10M vectors")
    print(f"  - Cần accuracy rất cao (>95%)")
    print(f"  - Memory không phải vấn đề")
    print(f"  - Search latency quan trọng")

    print(f"\n✓ Best practices:")
    print(f"  - M=32 cho most cases")
    print(f"  - efConstruction=40 cho build nhanh, 100-200 cho quality tốt")
    print(f"  - Tune efSearch để balance speed/accuracy")
    print(f"  - Không dùng cho dataset >10M (chuyển sang IVF)")

    print(f"\n✓ Production tips:")
    print(f"  - Build offline với efConstruction cao")
    print(f"  - Tune efSearch online theo latency requirements")
    print(f"  - Monitor memory usage (grows với M)")
    print(f"  - Consider sharding nếu dataset quá lớn")

    print(f"\n{'='*70}")
    print("Chạy example 06 để xem ứng dụng thực tế!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()