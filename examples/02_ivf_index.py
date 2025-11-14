"""
Example 02: IVF (Inverted File) Index
Ví dụ 2: IVF Index - Tìm kiếm nhanh với clustering

Minh họa:
- Clustering vectors với k-means
- Xây dựng IVF index
- Trade-off giữa tốc độ và độ chính xác với nprobe
- So sánh với Flat index
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
    print("FAISS Example 02: IVF (Inverted File) Index")
    print("="*70)

    # ===== Cấu hình =====
    dimension = 128
    n_vectors = 100000       # Tăng lên 100K để thấy rõ hiệu quả
    n_queries = 100
    k = 10

    # IVF parameters
    nlist = 100              # Số lượng clusters (centroids)
    nprobe = 10              # Số clusters để search (trade-off speed vs accuracy)

    print(f"\nCấu hình:")
    print(f"  Dimension: {dimension}")
    print(f"  Database size: {n_vectors:,} vectors")
    print(f"  Number of queries: {n_queries}")
    print(f"  K neighbors: {k}")
    print(f"\nIVF parameters:")
    print(f"  nlist (clusters): {nlist}")
    print(f"  nprobe (search clusters): {nprobe}")

    # ===== Giải thích IVF =====
    print(f"\n{'='*70}")
    print("IVF Index - Cách hoạt động")
    print(f"{'='*70}")
    print("""
IVF (Inverted File) Index sử dụng k-means clustering:

1. TRAINING PHASE:
   - Chia database thành 'nlist' clusters bằng k-means
   - Mỗi cluster có một centroid (trung tâm)
   - Mỗi vector được assign vào cluster gần nhất

2. INDEXING PHASE:
   - Vectors được lưu trong inverted lists của cluster tương ứng
   - Tạo cấu trúc: cluster_id -> list of vectors

3. SEARCH PHASE:
   - Tìm 'nprobe' clusters gần query nhất
   - Chỉ search trong các clusters đó (thay vì toàn bộ database)
   - Trade-off: nprobe nhỏ = nhanh, nprobe lớn = chính xác

Ưu điểm:
  ✓ Nhanh hơn nhiều so với Flat index
  ✓ Hiệu quả cho dataset lớn (>100K vectors)
  ✓ Có thể điều chỉnh speed/accuracy với nprobe

Nhược điểm:
  ✗ Cần training phase
  ✗ Độ chính xác < 100%
  ✗ Cần tune nlist và nprobe
    """)

    # ===== Tạo dữ liệu =====
    print(f"{'='*70}")
    print("Bước 1: Tạo dữ liệu")
    print(f"{'='*70}")

    print(f"Đang tạo {n_vectors:,} vectors...")
    database_vectors = generate_random_vectors(n_vectors, dimension)
    query_vectors = generate_query_vectors(n_queries, dimension)
    print(f"  ✓ Database vectors: {database_vectors.shape}")
    print(f"  ✓ Query vectors: {query_vectors.shape}")

    # ===== Tạo Ground Truth (Flat Index) =====
    print(f"\n{'='*70}")
    print("Bước 2: Tạo Ground Truth với Flat Index")
    print(f"{'='*70}")

    print("\nTạo Flat index để có ground truth (100% accurate)...")
    index_flat = faiss.IndexFlatL2(dimension)
    index_flat.add(database_vectors)

    print("Đang tìm kiếm với Flat index...")
    gt_distances, gt_indices = index_flat.search(query_vectors, k)
    print("  ✓ Ground truth ready")

    # ===== Xây dựng IVF Index =====
    print(f"\n{'='*70}")
    print("Bước 3: Xây dựng IVF Index")
    print(f"{'='*70}")

    print(f"\nTạo IVF index với {nlist} clusters...")

    # Quantizer: index để tìm cluster gần nhất
    quantizer = faiss.IndexFlatL2(dimension)

    # Tạo IVF index
    index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    print(f"  Index type: {type(index_ivf).__name__}")
    print(f"  Is trained: {index_ivf.is_trained}")

    # Training
    print(f"\nĐang train IVF index (k-means clustering)...")
    start_time = time.time()
    index_ivf.train(database_vectors)
    train_time = time.time() - start_time

    print(f"  ✓ Training hoàn tất trong {train_time:.3f} giây")
    print(f"  ✓ Is trained: {index_ivf.is_trained}")

    # Adding vectors
    print(f"\nĐang thêm {n_vectors:,} vectors vào index...")
    start_time = time.time()
    index_ivf.add(database_vectors)
    add_time = time.time() - start_time

    print(f"  ✓ Adding hoàn tất trong {add_time:.3f} giây")
    print(f"  ✓ Total build time: {train_time + add_time:.3f} giây")

    # In thông tin
    print_index_info(index_ivf, "\nIVF Index")

    # ===== So sánh với Flat Index =====
    print(f"\n{'='*70}")
    print("Bước 4: So sánh IVF vs Flat Index")
    print(f"{'='*70}")

    # Set nprobe
    index_ivf.nprobe = nprobe

    # Compare
    indexes = {
        'Flat (100% accurate)': index_flat,
        f'IVF (nprobe={nprobe})': index_ivf,
    }

    compare_indexes(indexes, query_vectors, gt_indices, k)

    # ===== Thử nghiệm với các nprobe khác nhau =====
    print(f"\n{'='*70}")
    print("Bước 5: Trade-off Speed vs Accuracy (thay đổi nprobe)")
    print(f"{'='*70}")

    print("\nnprobe càng lớn = search càng nhiều clusters = chính xác hơn nhưng chậm hơn")

    nprobe_values = [1, 2, 5, 10, 20, 50, 100]

    results = progressive_recall_benchmark(
        index_ivf,
        query_vectors,
        gt_indices,
        'nprobe',
        nprobe_values,
        k
    )

    # Visualize trade-off
    print(f"\nTrade-off Analysis:")
    print(f"  nprobe=1:   Nhanh nhất, recall thấp nhất")
    print(f"  nprobe=100: Chậm nhất, recall cao nhất (gần Flat)")
    print(f"  nprobe=10:  Sweet spot cho hầu hết use cases")

    # ===== Memory Comparison =====
    print(f"\n{'='*70}")
    print("Bước 6: So sánh Memory Usage")
    print(f"{'='*70}")

    flat_size = get_index_size(index_flat)
    ivf_size = get_index_size(index_ivf)

    print(f"\nMemory Usage:")
    print(f"  Flat Index:  {flat_size['size_mb']:.2f} MB")
    print(f"  IVF Index:   {ivf_size['size_mb']:.2f} MB")
    print(f"  Overhead:    {ivf_size['size_mb'] - flat_size['size_mb']:.2f} MB "
          f"({(ivf_size['size_mb']/flat_size['size_mb'] - 1)*100:.1f}%)")

    print(f"\nGiải thích:")
    print(f"  IVF cần thêm bộ nhớ cho:")
    print(f"    - {nlist} centroids")
    print(f"    - Inverted lists structure")
    print(f"    - Quantizer index")

    # ===== Phân tích cluster distribution =====
    print(f"\n{'='*70}")
    print("Bước 7: Phân tích Cluster Distribution")
    print(f"{'='*70}")

    # Đếm số vectors trong mỗi cluster
    print("\nĐang phân tích phân phối vectors trong clusters...")

    cluster_sizes = []
    for i in range(nlist):
        list_size = index_ivf.invlists.list_size(i)
        cluster_sizes.append(list_size)

    cluster_sizes = np.array(cluster_sizes)

    print(f"\nThống kê clusters:")
    print(f"  Tổng clusters: {nlist}")
    print(f"  Tổng vectors: {cluster_sizes.sum()}")
    print(f"  Mean vectors/cluster: {cluster_sizes.mean():.1f}")
    print(f"  Min vectors/cluster: {cluster_sizes.min()}")
    print(f"  Max vectors/cluster: {cluster_sizes.max()}")
    print(f"  Std deviation: {cluster_sizes.std():.1f}")

    # Clusters lớn nhất
    top_5_idx = np.argsort(cluster_sizes)[-5:][::-1]
    print(f"\n5 clusters lớn nhất:")
    for idx in top_5_idx:
        print(f"  Cluster {idx}: {cluster_sizes[idx]} vectors")

    # ===== Tóm tắt =====
    print(f"\n{'='*70}")
    print("TÓM TẮT - IVF Index")
    print(f"{'='*70}")

    print(f"\n✓ Hiệu suất:")
    print(f"  - Train time: {train_time:.3f}s")
    print(f"  - Add time: {add_time:.3f}s")
    print(f"  - Nhanh hơn Flat khi search (với nprobe < nlist)")
    print(f"  - Trade-off: nprobe={nprobe} đạt ~{results['recalls'][3]:.1%} recall")

    print(f"\n✓ Chọn nlist:")
    print(f"  - Rule of thumb: nlist = sqrt(n_vectors)")
    print(f"  - Với {n_vectors:,} vectors → nlist ≈ {int(np.sqrt(n_vectors))}")
    print(f"  - Dataset lớn hơn → nlist lớn hơn")

    print(f"\n✓ Chọn nprobe:")
    print(f"  - nprobe = 1: Fastest, lowest accuracy")
    print(f"  - nprobe = nlist/10: Good balance")
    print(f"  - nprobe = nlist: Same as Flat (no benefit)")

    print(f"\n✓ Khi nào dùng IVF:")
    print(f"  - Dataset lớn (>100K vectors)")
    print(f"  - Chấp nhận accuracy ~90-95%")
    print(f"  - Cần tốc độ search nhanh")
    print(f"  - Có thể afford training time")

    print(f"\n{'='*70}")
    print("Chạy example 03 để học về Product Quantization (nén vector)!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()