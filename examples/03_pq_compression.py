"""
Example 03: Product Quantization (PQ) - Vector Compression
Ví dụ 3: Nén vector với Product Quantization

Minh họa:
- Scalar Quantization (SQ)
- Product Quantization (PQ)
- So sánh memory usage
- Trade-off accuracy vs memory
"""

import numpy as np
import faiss
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import generate_random_vectors, generate_query_vectors
from utils.benchmark import (
    benchmark_index, print_index_info, compare_indexes, get_index_size
)


def main():
    print("="*70)
    print("FAISS Example 03: Product Quantization - Vector Compression")
    print("="*70)

    # ===== Cấu hình =====
    dimension = 128
    n_vectors = 100000
    n_queries = 100
    k = 10

    # PQ parameters
    m = 8                    # Số sub-vectors (phải là ước của dimension)
    nbits = 8                # Bits cho mỗi sub-vector (-> 2^8 = 256 centroids/sub-vector)

    # IVF parameters
    nlist = 100
    nprobe = 10

    print(f"\nCấu hình:")
    print(f"  Dimension: {dimension}")
    print(f"  Database size: {n_vectors:,} vectors")
    print(f"  Number of queries: {n_queries}")
    print(f"\nProduct Quantization parameters:")
    print(f"  m (sub-vectors): {m}")
    print(f"  nbits (bits per code): {nbits}")
    print(f"  Compression: {dimension * 4} bytes -> {m * nbits // 8} bytes per vector")
    print(f"  Compression ratio: {dimension * 4 / (m * nbits / 8):.1f}x")

    # ===== Giải thích PQ =====
    print(f"\n{'='*70}")
    print("Product Quantization - Cách hoạt động")
    print(f"{'='*70}")
    print(f"""
Product Quantization (PQ) nén vectors bằng cách:

1. CHIA VECTOR:
   - Chia mỗi {dimension}D vector thành {m} sub-vectors
   - Mỗi sub-vector có {dimension//m} dimensions

2. QUANTIZE MỖI SUB-VECTOR:
   - Train {2**nbits} centroids cho mỗi sub-vector (bằng k-means)
   - Mỗi sub-vector được thay bằng index của centroid gần nhất
   - Lưu index ({nbits} bits) thay vì full vector ({dimension//m} * 32 bits)

3. KẾT QUẢ:
   - Vector gốc: {dimension} * 4 bytes = {dimension * 4} bytes
   - Vector nén: {m} * {nbits//8} byte = {m * nbits // 8} bytes
   - Compression ratio: {dimension * 4 / (m * nbits / 8):.1f}x

4. TÌM KIẾM:
   - Pre-compute distances từ query đến tất cả centroids
   - Tính approximate distance bằng lookup table
   - Rất nhanh!

Ưu điểm:
  ✓ Giảm memory consumption rất nhiều
  ✓ Tăng tốc search (ít data để xử lý)
  ✓ Có thể kết hợp với IVF

Nhược điểm:
  ✗ Mất độ chính xác (lossy compression)
  ✗ Cần training
  ✗ m phải chia hết dimension
    """)

    # ===== Tạo dữ liệu =====
    print(f"{'='*70}")
    print("Bước 1: Tạo dữ liệu")
    print(f"{'='*70}")

    print(f"Đang tạo {n_vectors:,} vectors...")
    database_vectors = generate_random_vectors(n_vectors, dimension)
    query_vectors = generate_query_vectors(n_queries, dimension)
    print(f"  ✓ Database memory (uncompressed): {database_vectors.nbytes / (1024**2):.2f} MB")

    # ===== Ground Truth =====
    print(f"\n{'='*70}")
    print("Bước 2: Tạo Ground Truth")
    print(f"{'='*70}")

    print("Tạo Flat index...")
    index_flat = faiss.IndexFlatL2(dimension)
    index_flat.add(database_vectors)
    _, gt_indices = index_flat.search(query_vectors, k)
    print("  ✓ Ground truth ready")

    # ===== Scalar Quantization (SQ) =====
    print(f"\n{'='*70}")
    print("Bước 3: Scalar Quantization (SQ)")
    print(f"{'='*70}")

    print("""
Scalar Quantization (đơn giản hơn PQ):
  - Quantize mỗi dimension độc lập
  - float32 (4 bytes) -> int8 (1 byte)
  - Compression: 4x
  - Nhanh, đơn giản, ít mất accuracy hơn PQ
    """)

    print("Tạo IndexScalarQuantizer...")
    index_sq = faiss.IndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_8bit)

    print("Training SQ...")
    start_time = time.time()
    index_sq.train(database_vectors)
    index_sq.add(database_vectors)
    sq_build_time = time.time() - start_time

    print(f"  ✓ Build time: {sq_build_time:.3f}s")
    print_index_info(index_sq, "  Scalar Quantizer")

    # ===== Product Quantization (PQ) =====
    print(f"\n{'='*70}")
    print("Bước 4: Product Quantization (PQ)")
    print(f"{'='*70}")

    print(f"Tạo IndexPQ với m={m}, nbits={nbits}...")
    index_pq = faiss.IndexPQ(dimension, m, nbits)

    print(f"\nTraining PQ (learning {m} * {2**nbits} centroids)...")
    start_time = time.time()
    index_pq.train(database_vectors)
    train_time = time.time() - start_time
    print(f"  ✓ Training time: {train_time:.3f}s")

    print(f"\nAdding vectors...")
    start_time = time.time()
    index_pq.add(database_vectors)
    add_time = time.time() - start_time
    print(f"  ✓ Adding time: {add_time:.3f}s")
    print(f"  ✓ Total build time: {train_time + add_time:.3f}s")

    print_index_info(index_pq, "\n  Product Quantizer")

    # ===== IVF + PQ (Best of both worlds) =====
    print(f"\n{'='*70}")
    print("Bước 5: IVF + PQ (Kết hợp)")
    print(f"{'='*70}")

    print("""
IndexIVFPQ = IVF + PQ:
  - IVF: Giảm search space (speed)
  - PQ: Giảm memory (compression)
  - Best for large-scale systems!
    """)

    quantizer = faiss.IndexFlatL2(dimension)
    index_ivfpq = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)

    print(f"Training IVFPQ...")
    start_time = time.time()
    index_ivfpq.train(database_vectors)
    index_ivfpq.add(database_vectors)
    ivfpq_build_time = time.time() - start_time

    print(f"  ✓ Build time: {ivfpq_build_time:.3f}s")
    index_ivfpq.nprobe = nprobe
    print_index_info(index_ivfpq, "  IVF+PQ")

    # ===== Memory Comparison =====
    print(f"\n{'='*70}")
    print("Bước 6: So sánh Memory Usage")
    print(f"{'='*70}")

    indexes_memory = {
        'Flat (no compression)': index_flat,
        'Scalar Quantizer (SQ8)': index_sq,
        f'Product Quantizer (PQ{m}x{nbits})': index_pq,
        f'IVF+PQ (nlist={nlist})': index_ivfpq,
    }

    print(f"\n{'Index':<30} {'Size (MB)':<15} {'Compression':<15}")
    print(f"{'-'*60}")

    flat_size = get_index_size(index_flat)['size_mb']

    for name, idx in indexes_memory.items():
        size_mb = get_index_size(idx)['size_mb']
        compression = flat_size / size_mb if size_mb > 0 else 0
        print(f"{name:<30} {size_mb:<15.2f} {compression:<15.1f}x")

    # ===== Accuracy Comparison =====
    print(f"\n{'='*70}")
    print("Bước 7: So sánh Accuracy")
    print(f"{'='*70}")

    indexes_accuracy = {
        'Flat': index_flat,
        'SQ8': index_sq,
        f'PQ{m}x{nbits}': index_pq,
        f'IVF+PQ (nprobe={nprobe})': index_ivfpq,
    }

    compare_indexes(indexes_accuracy, query_vectors, gt_indices, k)

    # ===== Thử nghiệm PQ khác nhau =====
    print(f"\n{'='*70}")
    print("Bước 8: Thử nghiệm PQ configurations khác nhau")
    print(f"{'='*70}")

    print("\nSo sánh different m values (với nbits=8 cố định):")
    print(f"{'m':<10} {'Compression':<15} {'Recall@10':<15} {'Size (MB)':<15}")
    print(f"{'-'*55}")

    m_values = [4, 8, 16, 32]  # Các ước của 128

    for m_test in m_values:
        if dimension % m_test != 0:
            continue

        # Create and train
        idx_test = faiss.IndexPQ(dimension, m_test, nbits)
        idx_test.train(database_vectors[:50000])  # Train với subset để nhanh
        idx_test.add(database_vectors)

        # Benchmark
        from utils.benchmark import benchmark_index, calculate_recall
        _, pred_indices = idx_test.search(query_vectors, k)
        recall = calculate_recall(pred_indices, gt_indices, k)

        size_mb = get_index_size(idx_test)['size_mb']
        compression = flat_size / size_mb

        print(f"{m_test:<10} {compression:<15.1f}x {recall:<15.4f} {size_mb:<15.2f}")

    print(f"\nNhận xét:")
    print(f"  - m nhỏ: Compression cao hơn, accuracy thấp hơn")
    print(f"  - m lớn: Compression thấp hơn, accuracy cao hơn")
    print(f"  - Trade-off: Chọn m sao cho cân bằng memory và accuracy")

    # ===== Distance computation analysis =====
    print(f"\n{'='*70}")
    print("Bước 9: Phân tích Distance Approximation")
    print(f"{'='*70}")

    # Tìm kiếm với các index
    print("\nSo sánh distances từ các index khác nhau:")

    query = query_vectors[0:1]

    dist_flat, _ = index_flat.search(query, k)
    dist_pq, _ = index_pq.search(query, k)
    dist_sq, _ = index_sq.search(query, k)

    print(f"\n{'Rank':<8} {'Flat (exact)':<20} {'PQ (approx)':<20} {'SQ (approx)':<20}")
    print(f"{'-'*68}")

    for i in range(min(5, k)):
        print(f"{i+1:<8} {dist_flat[0,i]:<20.4f} {dist_pq[0,i]:<20.4f} {dist_sq[0,i]:<20.4f}")

    # ===== Tóm tắt =====
    print(f"\n{'='*70}")
    print("TÓM TẮT - Product Quantization")
    print(f"{'='*70}")

    print(f"\n✓ Compression methods:")
    print(f"  - Scalar Quantization: Đơn giản, 4x compression, accuracy tốt")
    print(f"  - Product Quantization: Flexible, >10x compression, accuracy trung bình")
    print(f"  - IVF+PQ: Best for production (speed + memory)")

    print(f"\n✓ Chọn parameters:")
    print(f"  - m: Số sub-vectors (phải chia hết dimension)")
    print(f"    • Lớn hơn -> accuracy cao hơn, compression thấp hơn")
    print(f"    • Thường dùng: 8, 16, 32")
    print(f"  - nbits: Bits per code")
    print(f"    • Thường dùng: 8 (256 centroids)")
    print(f"    • Có thể dùng 4, 6 để compress nhiều hơn")

    print(f"\n✓ Khi nào dùng PQ:")
    print(f"  - Dataset rất lớn (>10M vectors)")
    print(f"  - Memory là bottleneck")
    print(f"  - Chấp nhận accuracy ~80-90%")
    print(f"  - Kết hợp với IVF cho best results")

    print(f"\n✓ Best practices:")
    print(f"  - Train với representative sample (>256 * m vectors)")
    print(f"  - Normalize vectors trước khi PQ")
    print(f"  - Dùng IVF+PQ thay vì PQ alone")
    print(f"  - Tune m và nbits trên validation set")

    print(f"\n{'='*70}")
    print("Chạy example 04 để học về HNSW graph-based search!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()