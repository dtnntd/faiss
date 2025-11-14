"""
Benchmark utilities for FAISS indexes
Công cụ đo lường hiệu suất cho các FAISS indexes
"""

import time
import sys
import numpy as np
import faiss
from typing import Dict, Tuple, Optional


def benchmark_index(
    index: faiss.Index,
    queries: np.ndarray,
    k: int = 10,
    ground_truth: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Benchmark một FAISS index

    Args:
        index: FAISS index đã được train và add vectors
        queries: Query vectors
        k: Số lượng nearest neighbors cần tìm
        ground_truth: Ground truth IDs (từ brute-force search) để tính recall

    Returns:
        Dictionary chứa các metrics:
        - search_time: Thời gian search (seconds)
        - qps: Queries per second
        - recall@k: Recall nếu có ground_truth
    """
    results = {}

    # Warm-up
    index.search(queries[:10], k)

    # Benchmark search time
    start_time = time.time()
    distances, indices = index.search(queries, k)
    search_time = time.time() - start_time

    results['search_time'] = search_time
    results['qps'] = len(queries) / search_time

    # Tính recall nếu có ground truth
    if ground_truth is not None:
        recall = calculate_recall(indices, ground_truth, k)
        results[f'recall@{k}'] = recall

    return results


def calculate_recall(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: int
) -> float:
    """
    Tính recall@k

    Args:
        predicted: Predicted neighbor IDs, shape (n_queries, k)
        ground_truth: Ground truth neighbor IDs, shape (n_queries, k_gt)
        k: Số lượng neighbors để tính recall

    Returns:
        Recall@k (0.0 đến 1.0)
    """
    n_queries = predicted.shape[0]
    recall = 0.0

    for i in range(n_queries):
        # Lấy k neighbors từ ground truth
        gt_set = set(ground_truth[i, :k])
        # Lấy k neighbors từ prediction
        pred_set = set(predicted[i, :k])

        # Số lượng matches
        matches = len(gt_set.intersection(pred_set))
        recall += matches / k

    return recall / n_queries


def get_index_size(index: faiss.Index) -> Dict[str, float]:
    """
    Ước tính kích thước bộ nhớ của index

    Args:
        index: FAISS index

    Returns:
        Dictionary với memory usage (MB)
    """
    # Serialize index ra bytes để đo kích thước chính xác
    index_bytes = faiss.serialize_index(index)
    size_mb = len(index_bytes) / (1024 * 1024)

    return {
        'size_mb': size_mb,
        'size_bytes': len(index_bytes)
    }


def compare_indexes(
    indexes: Dict[str, faiss.Index],
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10
) -> None:
    """
    So sánh nhiều indexes và in kết quả

    Args:
        indexes: Dictionary {name: index}
        queries: Query vectors
        ground_truth: Ground truth IDs
        k: Number of neighbors
    """
    print(f"\n{'='*70}")
    print(f"{'Index Comparison':^70}")
    print(f"{'='*70}")
    print(f"{'Index':<20} {'Recall@'+str(k):<15} {'QPS':<15} {'Size (MB)':<15}")
    print(f"{'-'*70}")

    for name, index in indexes.items():
        # Benchmark
        results = benchmark_index(index, queries, k, ground_truth)

        # Get size
        size_info = get_index_size(index)

        # Print results
        print(f"{name:<20} "
              f"{results[f'recall@{k}']:<15.4f} "
              f"{results['qps']:<15.1f} "
              f"{size_info['size_mb']:<15.2f}")

    print(f"{'='*70}\n")


def benchmark_build_time(
    index_factory_fn,
    vectors: np.ndarray,
    train_vectors: Optional[np.ndarray] = None
) -> Tuple[faiss.Index, float]:
    """
    Đo thời gian xây dựng index

    Args:
        index_factory_fn: Function tạo index (không có arguments)
        vectors: Vectors để add vào index
        train_vectors: Vectors để train (nếu cần)

    Returns:
        Tuple of (index, build_time)
    """
    start_time = time.time()

    # Tạo index
    index = index_factory_fn()

    # Train nếu cần
    if train_vectors is not None and not index.is_trained:
        index.train(train_vectors)

    # Add vectors
    index.add(vectors)

    build_time = time.time() - start_time

    return index, build_time


def print_index_info(index: faiss.Index, name: str = "Index") -> None:
    """
    In thông tin chi tiết về index

    Args:
        index: FAISS index
        name: Tên của index
    """
    print(f"\n{name} Information:")
    print(f"  Type: {type(index).__name__}")
    print(f"  Dimension: {index.d}")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Is trained: {index.is_trained}")
    print(f"  Metric type: {index.metric_type}")

    # Size info
    size_info = get_index_size(index)
    print(f"  Memory size: {size_info['size_mb']:.2f} MB")

    # Index-specific info
    if hasattr(index, 'nlist'):
        print(f"  Number of clusters (nlist): {index.nlist}")
    if hasattr(index, 'nprobe'):
        print(f"  Number of probes (nprobe): {index.nprobe}")
    if hasattr(index, 'hnsw'):
        print(f"  HNSW M: {index.hnsw.max_level}")


def progressive_recall_benchmark(
    index,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    param_name: str,
    param_values: list,
    k: int = 10
) -> Dict[str, list]:
    """
    Benchmark index với các giá trị tham số khác nhau

    Args:
        index: FAISS index
        queries: Query vectors
        ground_truth: Ground truth IDs
        param_name: Tên parameter cần thay đổi (vd: 'nprobe', 'efSearch')
        param_values: List các giá trị parameter
        k: Number of neighbors

    Returns:
        Dictionary with results for each parameter value
    """
    results = {
        'param_values': param_values,
        'recalls': [],
        'qps': [],
        'search_times': []
    }

    print(f"\nProgressive benchmark for {param_name}:")
    print(f"{'Value':<10} {'Recall@'+str(k):<15} {'QPS':<15} {'Time (s)':<15}")
    print(f"{'-'*55}")

    for value in param_values:
        # Set parameter (handle nested attributes for HNSW)
        if hasattr(index, param_name):
            setattr(index, param_name, value)
        elif hasattr(index, 'hnsw') and hasattr(index.hnsw, param_name):
            # For HNSW indexes, efSearch is on index.hnsw
            setattr(index.hnsw, param_name, value)
        else:
            raise AttributeError(f"Parameter '{param_name}' not found on index or index.hnsw")

        # Benchmark
        bench_results = benchmark_index(index, queries, k, ground_truth)

        # Store results
        results['recalls'].append(bench_results[f'recall@{k}'])
        results['qps'].append(bench_results['qps'])
        results['search_times'].append(bench_results['search_time'])

        # Print
        print(f"{value:<10} "
              f"{bench_results[f'recall@{k}']:<15.4f} "
              f"{bench_results['qps']:<15.1f} "
              f"{bench_results['search_time']:<15.4f}")

    return results


if __name__ == "__main__":
    print("Testing benchmark utilities...")

    # Create simple test data
    d = 64
    n_base = 10000
    n_queries = 100

    np.random.seed(42)
    base_vectors = np.random.random((n_base, d)).astype('float32')
    queries = np.random.random((n_queries, d)).astype('float32')

    # Create indexes
    index_flat = faiss.IndexFlatL2(d)
    index_flat.add(base_vectors)

    # Get ground truth
    _, gt_indices = index_flat.search(queries, 10)

    # Benchmark
    results = benchmark_index(index_flat, queries, 10, gt_indices)
    print(f"\nFlat index results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    # Print index info
    print_index_info(index_flat, "Flat Index")

    print("\nAll tests passed!")