"""
Data generator utilities for FAISS examples
Tạo dữ liệu test cho các ví dụ FAISS
"""

import numpy as np
from typing import Tuple


def generate_random_vectors(
    n_vectors: int,
    dimension: int,
    seed: int = 42
) -> np.ndarray:
    """
    Tạo vectors ngẫu nhiên

    Args:
        n_vectors: Số lượng vectors cần tạo
        dimension: Số chiều của mỗi vector
        seed: Random seed để tái tạo kết quả

    Returns:
        np.ndarray: Mảng vectors có shape (n_vectors, dimension)
    """
    np.random.seed(seed)
    vectors = np.random.random((n_vectors, dimension)).astype('float32')
    return vectors


def generate_query_vectors(
    n_queries: int,
    dimension: int,
    seed: int = 123
) -> np.ndarray:
    """
    Tạo query vectors để test

    Args:
        n_queries: Số lượng queries
        dimension: Số chiều của mỗi vector
        seed: Random seed

    Returns:
        np.ndarray: Mảng query vectors
    """
    np.random.seed(seed)
    queries = np.random.random((n_queries, dimension)).astype('float32')
    return queries


def generate_clustered_vectors(
    n_clusters: int,
    points_per_cluster: int,
    dimension: int,
    cluster_std: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo vectors được phân cụm (để test clustering-based indexes)

    Args:
        n_clusters: Số lượng clusters
        points_per_cluster: Số điểm trong mỗi cluster
        dimension: Số chiều
        cluster_std: Độ lệch chuẩn của mỗi cluster
        seed: Random seed

    Returns:
        Tuple of (vectors, labels)
    """
    np.random.seed(seed)

    vectors = []
    labels = []

    for i in range(n_clusters):
        # Tạo center ngẫu nhiên cho mỗi cluster
        center = np.random.random(dimension).astype('float32')

        # Tạo points xung quanh center
        cluster_points = center + np.random.randn(
            points_per_cluster, dimension
        ).astype('float32') * cluster_std

        vectors.append(cluster_points)
        labels.extend([i] * points_per_cluster)

    vectors = np.vstack(vectors)
    labels = np.array(labels)

    return vectors, labels


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors về unit length (cho cosine similarity)

    Args:
        vectors: Input vectors

    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Tránh chia cho 0
    norms = np.maximum(norms, 1e-10)
    return (vectors / norms).astype('float32')


def generate_text_like_embeddings(
    n_documents: int,
    dimension: int = 384,
    n_topics: int = 10,
    seed: int = 42
) -> np.ndarray:
    """
    Tạo embeddings giống như text embeddings thực tế
    (có cấu trúc clusters tương ứng với topics)

    Args:
        n_documents: Số lượng documents
        dimension: Dimension của embeddings (default 384 giống SBERT)
        n_topics: Số lượng topics
        seed: Random seed

    Returns:
        Document embeddings đã được normalize
    """
    np.random.seed(seed)

    # Tạo topic vectors
    topic_vectors = np.random.randn(n_topics, dimension).astype('float32')
    topic_vectors = normalize_vectors(topic_vectors)

    # Mỗi document là mixture của các topics
    vectors = []
    for _ in range(n_documents):
        # Random topic weights
        weights = np.random.dirichlet(np.ones(n_topics) * 0.5)

        # Weighted sum của topic vectors
        doc_vector = np.sum(
            weights[:, np.newaxis] * topic_vectors,
            axis=0
        )

        # Thêm một chút noise
        doc_vector += np.random.randn(dimension).astype('float32') * 0.1

        vectors.append(doc_vector)

    vectors = np.vstack(vectors)
    return normalize_vectors(vectors)


if __name__ == "__main__":
    # Test các hàm
    print("Testing data generators...")

    # Test random vectors
    vectors = generate_random_vectors(1000, 128)
    print(f"Random vectors shape: {vectors.shape}")
    print(f"Random vectors dtype: {vectors.dtype}")

    # Test clustered vectors
    vectors, labels = generate_clustered_vectors(5, 200, 64)
    print(f"\nClustered vectors shape: {vectors.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    # Test normalized vectors
    norm_vectors = normalize_vectors(vectors)
    norms = np.linalg.norm(norm_vectors, axis=1)
    print(f"\nNormalized vectors - mean norm: {norms.mean():.4f}")
    print(f"Normalized vectors - std norm: {norms.std():.4f}")

    # Test text embeddings
    text_emb = generate_text_like_embeddings(500, 384, 10)
    print(f"\nText embeddings shape: {text_emb.shape}")
    print(f"Text embeddings mean norm: {np.linalg.norm(text_emb, axis=1).mean():.4f}")

    print("\nAll tests passed!")