"""
Example 01: Basic Vector Search with FAISS
Ví dụ 1: Tìm kiếm vector cơ bản với FAISS

Minh họa:
- Tạo vectors ngẫu nhiên
- Xây dựng Flat index (brute-force, 100% accuracy)
- Tìm kiếm k-nearest neighbors
- Đo thời gian và hiệu suất
"""

import numpy as np
import faiss
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import generate_random_vectors, generate_query_vectors
from utils.benchmark import benchmark_index, print_index_info


def main():
    print("="*70)
    print("FAISS Example 01: Basic Vector Search")
    print("="*70)

    # ===== Cấu hình =====
    dimension = 128          # Số chiều của vector
    n_vectors = 10000        # Số lượng vectors trong database
    n_queries = 100          # Số lượng query vectors
    k = 10                   # Số neighbors cần tìm

    print(f"\nCấu hình:")
    print(f"  Dimension: {dimension}")
    print(f"  Database size: {n_vectors:,} vectors")
    print(f"  Number of queries: {n_queries}")
    print(f"  K neighbors: {k}")

    # ===== Tạo dữ liệu =====
    print(f"\n{'='*70}")
    print("Bước 1: Tạo dữ liệu")
    print(f"{'='*70}")

    print("Đang tạo database vectors...")
    database_vectors = generate_random_vectors(n_vectors, dimension)
    print(f"  ✓ Tạo xong {n_vectors:,} vectors, shape: {database_vectors.shape}")
    print(f"  ✓ Data type: {database_vectors.dtype}")
    print(f"  ✓ Memory size: {database_vectors.nbytes / (1024**2):.2f} MB")

    print("\nĐang tạo query vectors...")
    query_vectors = generate_query_vectors(n_queries, dimension)
    print(f"  ✓ Tạo xong {n_queries} queries")

    # ===== Xây dựng Index =====
    print(f"\n{'='*70}")
    print("Bước 2: Xây dựng FAISS Index")
    print(f"{'='*70}")

    print("\nTạo IndexFlatL2 (brute-force, exact search)...")
    print("  IndexFlatL2 sử dụng L2 distance (Euclidean distance)")
    print("  Độ chính xác: 100% (exhaustive search)")
    print("  Tốc độ: Chậm nhất, nhưng chính xác nhất")

    # Tạo index
    start_time = time.time()
    index = faiss.IndexFlatL2(dimension)

    # Thêm vectors vào index
    print(f"\nĐang thêm {n_vectors:,} vectors vào index...")
    index.add(database_vectors)
    build_time = time.time() - start_time

    print(f"  ✓ Xây dựng index hoàn tất trong {build_time:.3f} giây")

    # In thông tin index
    print_index_info(index, "\nFlat Index")

    # ===== Tìm kiếm =====
    print(f"\n{'='*70}")
    print("Bước 3: Tìm kiếm K-Nearest Neighbors")
    print(f"{'='*70}")

    print(f"\nĐang tìm kiếm {k} nearest neighbors cho {n_queries} queries...")

    # Warm-up
    _ = index.search(query_vectors[:5], k)

    # Benchmark search
    start_time = time.time()
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - start_time

    print(f"  ✓ Tìm kiếm hoàn tất trong {search_time:.3f} giây")
    print(f"  ✓ Queries per second (QPS): {n_queries / search_time:.1f}")
    print(f"  ✓ Average time per query: {search_time / n_queries * 1000:.3f} ms")

    # ===== Phân tích kết quả =====
    print(f"\n{'='*70}")
    print("Bước 4: Phân tích kết quả")
    print(f"{'='*70}")

    print(f"\nKích thước kết quả:")
    print(f"  Distances shape: {distances.shape}")
    print(f"  Indices shape: {indices.shape}")

    # Hiển thị kết quả cho query đầu tiên
    print(f"\nKết quả cho query đầu tiên (query #0):")
    print(f"  Query vector shape: {query_vectors[0].shape}")
    print(f"\n  Top {k} nearest neighbors:")
    print(f"  {'Rank':<6} {'Index':<10} {'Distance':<15}")
    print(f"  {'-'*35}")

    for rank in range(k):
        idx = indices[0, rank]
        dist = distances[0, rank]
        print(f"  {rank+1:<6} {idx:<10} {dist:<15.4f}")

    # Thống kê về distances
    print(f"\nThống kê distances:")
    print(f"  Min distance: {distances.min():.4f}")
    print(f"  Max distance: {distances.max():.4f}")
    print(f"  Mean distance: {distances.mean():.4f}")
    print(f"  Median distance: {np.median(distances):.4f}")

    # ===== So sánh với numpy =====
    print(f"\n{'='*70}")
    print("Bước 5: Verification - So sánh với NumPy")
    print(f"{'='*70}")

    print("\nTính toán brute-force bằng NumPy cho query đầu tiên...")
    query = query_vectors[0:1]

    start_time = time.time()
    # Tính L2 distance bằng NumPy
    numpy_distances = np.sum((database_vectors - query)**2, axis=1)
    # Tìm k smallest
    k_smallest_idx = np.argsort(numpy_distances)[:k]
    k_smallest_dist = numpy_distances[k_smallest_idx]
    numpy_time = time.time() - start_time

    print(f"  ✓ NumPy computation time: {numpy_time:.4f} seconds")

    # So sánh kết quả
    print(f"\nSo sánh kết quả FAISS vs NumPy:")
    faiss_result = set(indices[0, :k])
    numpy_result = set(k_smallest_idx)

    if faiss_result == numpy_result:
        print("  ✓ Kết quả GIỐNG NHAU - FAISS chính xác!")
    else:
        print("  ✗ Kết quả KHÁC NHAU")
        print(f"    FAISS: {sorted(list(faiss_result))[:5]}...")
        print(f"    NumPy: {sorted(list(numpy_result))[:5]}...")

    # So sánh distances
    dist_diff = np.abs(distances[0, :k] - k_smallest_dist).max()
    print(f"  Max difference in distances: {dist_diff:.6f}")

    # ===== Tóm tắt =====
    print(f"\n{'='*70}")
    print("TÓM TẮT")
    print(f"{'='*70}")

    print(f"\n✓ IndexFlatL2 characteristics:")
    print(f"  - Accuracy: 100% (exact search)")
    print(f"  - Build time: {build_time:.3f}s")
    print(f"  - Search time: {search_time:.3f}s ({n_queries/search_time:.1f} QPS)")
    print(f"  - Memory: {database_vectors.nbytes / (1024**2):.2f} MB")

    print(f"\n✓ Khi nào dùng IndexFlatL2:")
    print(f"  - Dataset nhỏ (<100K vectors)")
    print(f"  - Cần độ chính xác 100%")
    print(f"  - Dùng làm baseline để so sánh với các index khác")
    print(f"  - Tạo ground truth cho evaluation")

    print(f"\n✓ Hạn chế:")
    print(f"  - Chậm với dataset lớn (O(n) complexity)")
    print(f"  - Không scale tốt với hàng triệu vectors")
    print(f"  - Nên dùng IVF, HNSW cho datasets lớn")

    print(f"\n{'='*70}")
    print("Chạy example 02 để tìm hiểu về IVF index (nhanh hơn)!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()