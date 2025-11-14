"""
Example 06: Real-World Application
Ví dụ 6: Ứng dụng thực tế - Product Recommendation & Semantic Search

Minh họa:
- Product similarity search (e-commerce)
- Semantic search for documents
- User-based recommendation
- Practical tips and patterns
"""

import numpy as np
import faiss
import time
import sys
import os
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import generate_text_like_embeddings, normalize_vectors
from utils.benchmark import benchmark_index, print_index_info


# ===== Simulated Product Database =====
class ProductDatabase:
    """Mô phỏng database sản phẩm e-commerce"""

    def __init__(self, n_products: int = 10000):
        self.n_products = n_products

        # Tạo metadata cho products
        self.products = self._generate_products()

        # Tạo embeddings (giả sử từ model như CLIP, ResNet, etc.)
        print("Tạo product embeddings (giả sử từ image/text encoder)...")
        self.embeddings = generate_text_like_embeddings(
            n_products,
            dimension=256,  # Dimension của embedding model
            n_topics=20     # 20 product categories
        )

    def _generate_products(self) -> List[Dict]:
        """Tạo fake product data"""
        categories = [
            "Electronics", "Clothing", "Home & Kitchen", "Books",
            "Sports", "Toys", "Beauty", "Automotive", "Food", "Pet Supplies"
        ]

        products = []
        for i in range(self.n_products):
            products.append({
                'id': i,
                'name': f'Product {i}',
                'category': categories[i % len(categories)],
                'price': np.random.uniform(10, 1000),
                'rating': np.random.uniform(1, 5),
            })
        return products

    def get_product(self, product_id: int) -> Dict:
        """Lấy thông tin product"""
        return self.products[product_id]


def demo_product_search():
    """Demo 1: Product Similarity Search"""

    print("="*70)
    print("DEMO 1: Product Similarity Search (E-Commerce)")
    print("="*70)

    print("""
Use case: Recommend similar products
- User đang xem sản phẩm A
- Tìm các sản phẩm tương tự để recommend
- Dựa trên visual/text embeddings
    """)

    # Tạo product database
    print("\n" + "="*70)
    print("Khởi tạo Product Database")
    print("="*70)

    db = ProductDatabase(n_products=10000)
    print(f"  ✓ Loaded {len(db.products):,} products")
    print(f"  ✓ Embedding dimension: {db.embeddings.shape[1]}")

    # Build index
    print("\n" + "="*70)
    print("Build FAISS Index")
    print("="*70)

    dimension = db.embeddings.shape[1]
    n_products = len(db.products)

    # Chọn index type dựa vào dataset size
    if n_products < 10000:
        print("Dataset nhỏ -> sử dụng HNSW")
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 64
    else:
        print("Dataset lớn -> sử dụng IVF")
        nlist = int(np.sqrt(n_products))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(db.embeddings)
        index.nprobe = 10

    # Add products
    print("Đang thêm products vào index...")
    start_time = time.time()
    index.add(db.embeddings)
    build_time = time.time() - start_time

    print(f"  ✓ Build time: {build_time:.3f}s")
    print_index_info(index, "  Product Index")

    # Search for similar products
    print("\n" + "="*70)
    print("Tìm sản phẩm tương tự")
    print("="*70)

    # Chọn một sản phẩm ngẫu nhiên
    query_product_id = 42
    query_product = db.get_product(query_product_id)

    print(f"\nSản phẩm gốc:")
    print(f"  ID: {query_product['id']}")
    print(f"  Name: {query_product['name']}")
    print(f"  Category: {query_product['category']}")
    print(f"  Price: ${query_product['price']:.2f}")

    # Search
    k = 10  # Tìm 10 sản phẩm tương tự
    query_vector = db.embeddings[query_product_id:query_product_id+1]

    distances, indices = index.search(query_vector, k + 1)  # +1 vì kết quả đầu là chính nó

    print(f"\nTop {k} sản phẩm tương tự:")
    print(f"{'Rank':<6} {'ID':<8} {'Name':<20} {'Category':<20} {'Price':<12} {'Distance':<12}")
    print("-" * 78)

    for rank, (idx, dist) in enumerate(zip(indices[0][1:], distances[0][1:]), 1):  # Skip first (itself)
        product = db.get_product(int(idx))
        print(f"{rank:<6} {product['id']:<8} {product['name']:<20} "
              f"{product['category']:<20} ${product['price']:<11.2f} {dist:<12.4f}")

    print(f"\n✓ Use case: 'Customers who viewed this also viewed...'")


def demo_semantic_search():
    """Demo 2: Semantic Document Search"""

    print("\n\n" + "="*70)
    print("DEMO 2: Semantic Document Search")
    print("="*70)

    print("""
Use case: Search documents by meaning
- User search query: "machine learning algorithms"
- Tìm documents có nội dung liên quan
- Dựa trên semantic embeddings (SBERT, OpenAI, etc.)
    """)

    # Simulate document database
    print("\n" + "="*70)
    print("Khởi tạo Document Database")
    print("="*70)

    n_documents = 5000
    dimension = 384  # SBERT all-MiniLM-L6-v2 dimension

    # Fake documents
    topics = [
        "Machine Learning", "Web Development", "Data Science",
        "Mobile Apps", "Cloud Computing", "Cybersecurity",
        "AI Ethics", "Database Systems", "DevOps", "Blockchain"
    ]

    documents = []
    for i in range(n_documents):
        documents.append({
            'id': i,
            'title': f'Document {i}: {topics[i % len(topics)]}',
            'topic': topics[i % len(topics)],
            'views': np.random.randint(0, 10000),
        })

    # Generate embeddings (giả sử từ SBERT)
    print(f"Tạo embeddings cho {n_documents:,} documents...")
    doc_embeddings = generate_text_like_embeddings(
        n_documents,
        dimension=dimension,
        n_topics=len(topics)
    )

    print(f"  ✓ Documents: {n_documents:,}")
    print(f"  ✓ Embedding dimension: {dimension}")

    # Build index
    print("\n" + "="*70)
    print("Build Index")
    print("="*70)

    # Dùng Inner Product cho normalized embeddings (cosine similarity)
    print("Sử dụng IndexFlatIP (Inner Product) cho cosine similarity...")

    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
    index.add(doc_embeddings)

    print(f"  ✓ Index type: {type(index).__name__}")
    print(f"  ✓ Metric: Inner Product (cosine similarity)")

    # Simulate search
    print("\n" + "="*70)
    print("Search Query")
    print("="*70)

    # Giả sử query từ user được embed bằng cùng model
    query_text = "deep learning and neural networks"
    print(f"\nUser query: '{query_text}'")

    # Fake query embedding (trong thực tế: embed query bằng SBERT)
    # Giả sử query gần với topic "Machine Learning" và "AI Ethics"
    query_embedding = normalize_vectors(
        np.random.randn(1, dimension).astype('float32')
    )

    # Search
    k = 5
    scores, indices = index.search(query_embedding, k)

    print(f"\nTop {k} relevant documents:")
    print(f"{'Rank':<6} {'ID':<8} {'Title':<45} {'Score':<12}")
    print("-" * 71)

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        doc = documents[int(idx)]
        print(f"{rank:<6} {doc['id']:<8} {doc['title']:<45} {score:<12.4f}")

    print(f"\n✓ Higher score = more similar (Inner Product)")
    print(f"✓ Use case: Documentation search, Q&A systems, etc.")


def demo_user_recommendations():
    """Demo 3: User-based Recommendations"""

    print("\n\n" + "="*70)
    print("DEMO 3: User-based Recommendations")
    print("="*70)

    print("""
Use case: Find similar users for collaborative filtering
- Tìm users có behavior tương tự
- Recommend items mà similar users thích
- Dựa trên user embeddings (từ interaction history)
    """)

    print("\n" + "="*70)
    print("Khởi tạo User Database")
    print("="*70)

    n_users = 10000
    dimension = 128  # User embedding dimension

    # Simulate user embeddings (từ interaction matrix, neural CF, etc.)
    print(f"Tạo embeddings cho {n_users:,} users...")
    user_embeddings = generate_text_like_embeddings(
        n_users,
        dimension=dimension,
        n_topics=15  # 15 user segments
    )

    # User metadata
    users = [
        {
            'id': i,
            'username': f'user_{i}',
            'segment': f'segment_{i % 15}',
            'activity_score': np.random.uniform(0, 100)
        }
        for i in range(n_users)
    ]

    # Build index
    print("\n" + "="*70)
    print("Build User Index")
    print("="*70)

    # Sử dụng IVF cho scalability
    nlist = 100
    quantizer = faiss.IndexFlatIP(dimension)  # IP cho cosine similarity
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    print("Training index...")
    index.train(user_embeddings)
    index.add(user_embeddings)
    index.nprobe = 10

    print(f"  ✓ Users indexed: {n_users:,}")

    # Find similar users
    print("\n" + "="*70)
    print("Tìm Similar Users")
    print("="*70)

    target_user_id = 123
    target_user = users[target_user_id]

    print(f"\nTarget user:")
    print(f"  ID: {target_user['id']}")
    print(f"  Username: {target_user['username']}")
    print(f"  Segment: {target_user['segment']}")

    # Search
    k = 10
    query_vector = user_embeddings[target_user_id:target_user_id+1]
    scores, indices = index.search(query_vector, k + 1)

    print(f"\nTop {k} similar users:")
    print(f"{'Rank':<6} {'ID':<8} {'Username':<20} {'Segment':<15} {'Similarity':<12}")
    print("-" * 61)

    for rank, (idx, score) in enumerate(zip(indices[0][1:], scores[0][1:]), 1):
        user = users[int(idx)]
        print(f"{rank:<6} {user['id']:<8} {user['username']:<20} "
              f"{user['segment']:<15} {score:<12.4f}")

    print(f"\n✓ Use case: Collaborative filtering, social recommendations")
    print(f"✓ Next step: Recommend items that similar users liked")


def demo_practical_tips():
    """Demo 4: Production Best Practices"""

    print("\n\n" + "="*70)
    print("DEMO 4: Production Best Practices")
    print("="*70)

    print("""
Practical tips cho production systems:
    """)

    # Tip 1: Save/Load Index
    print("\n" + "="*70)
    print("Tip 1: Save và Load Index")
    print("="*70)

    dimension = 128
    n_vectors = 1000

    vectors = np.random.random((n_vectors, dimension)).astype('float32')

    # Create index
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Save to disk
    index_file = "data/product_index.faiss"
    os.makedirs("data", exist_ok=True)

    print(f"Saving index to {index_file}...")
    faiss.write_index(index, index_file)
    print(f"  ✓ Index saved ({os.path.getsize(index_file) / 1024:.2f} KB)")

    # Load from disk
    print(f"\nLoading index from {index_file}...")
    loaded_index = faiss.read_index(index_file)
    print(f"  ✓ Index loaded")
    print(f"  ✓ Vectors: {loaded_index.ntotal}")

    # Tip 2: Batch Search
    print("\n" + "="*70)
    print("Tip 2: Batch Search (hiệu quả hơn single query)")
    print("="*70)

    n_queries = 100
    queries = np.random.random((n_queries, dimension)).astype('float32')

    # Single query (slow)
    print("Single query search (1 by 1):")
    start = time.time()
    for i in range(n_queries):
        _ = index.search(queries[i:i+1], 10)
    single_time = time.time() - start
    print(f"  Time: {single_time:.4f}s")

    # Batch search (fast)
    print("\nBatch search (all at once):")
    start = time.time()
    _ = index.search(queries, 10)
    batch_time = time.time() - start
    print(f"  Time: {batch_time:.4f}s")

    print(f"\n  ✓ Speedup: {single_time/batch_time:.1f}x faster!")

    # Tip 3: ID Mapping
    print("\n" + "="*70)
    print("Tip 3: ID Mapping (FAISS IDs vs Database IDs)")
    print("="*70)

    print("""
FAISS sử dụng sequential IDs (0, 1, 2, ...)
Database của bạn có thể dùng custom IDs (UUID, etc.)

Solution: Dùng IndexIDMap
    """)

    # Original IDs from database
    db_ids = np.array([1001, 2005, 3010, 4020, 5030], dtype='int64')
    vectors_with_ids = np.random.random((5, dimension)).astype('float32')

    # Create index with ID mapping
    base_index = faiss.IndexFlatL2(dimension)
    index_with_ids = faiss.IndexIDMap(base_index)

    # Add with custom IDs
    index_with_ids.add_with_ids(vectors_with_ids, db_ids)

    print(f"Added {len(db_ids)} vectors with custom IDs")

    # Search
    query = np.random.random((1, dimension)).astype('float32')
    _, indices = index_with_ids.search(query, 3)

    print(f"\nSearch results return custom IDs:")
    print(f"  IDs: {indices[0]}")
    print(f"  ✓ These match your database IDs!")

    # Tip 4: Index Factory
    print("\n" + "="*70)
    print("Tip 4: Index Factory (dễ dàng tạo complex indexes)")
    print("="*70)

    print("""
Index Factory string: Tạo indexes từ description string
Ví dụ:
  - "Flat": IndexFlatL2
  - "IVF100,Flat": IVF with 100 clusters
  - "IVF100,PQ8": IVF + Product Quantization
  - "HNSW32": HNSW with M=32
    """)

    # Examples
    index_flat = faiss.index_factory(dimension, "Flat")
    index_ivf = faiss.index_factory(dimension, "IVF100,Flat")
    index_ivfpq = faiss.index_factory(dimension, "IVF100,PQ8")
    index_hnsw = faiss.index_factory(dimension, "HNSW32")

    print(f"\nCreated indexes:")
    print(f"  Flat: {type(index_flat).__name__}")
    print(f"  IVF100,Flat: {type(index_ivf).__name__}")
    print(f"  IVF100,PQ8: {type(index_ivfpq).__name__}")
    print(f"  HNSW32: {type(index_hnsw).__name__}")

    # Cleanup
    if os.path.exists(index_file):
        os.remove(index_file)


def main():
    print("="*70)
    print("FAISS Example 06: Real-World Applications")
    print("="*70)

    # Run demos
    demo_product_search()
    demo_semantic_search()
    demo_user_recommendations()
    demo_practical_tips()

    # Final summary
    print("\n\n" + "="*70)
    print("TÓM TẮT - Production Checklist")
    print("="*70)

    print("""
✓ Index Selection:
  - <10K vectors: HNSW hoặc Flat
  - 10K-1M: HNSW hoặc IVF
  - >1M: IVF với PQ compression
  - Millions to billions: Sharded IVF+PQ

✓ Distance Metric:
  - L2: Euclidean distance (default)
  - Inner Product: Cosine similarity (với normalized vectors)
  - Normalize vectors nếu dùng Inner Product

✓ Performance:
  - Batch queries khi có thể
  - Tune nprobe (IVF) hoặc efSearch (HNSW)
  - Monitor memory và latency
  - Consider GPU nếu QPS cao

✓ Persistence:
  - Save index với faiss.write_index()
  - Load với faiss.read_index()
  - Version control cho index configs

✓ Scalability:
  - Shard index nếu quá lớn
  - Distributed search với multiple replicas
  - Update strategies: rebuild vs incremental

✓ Monitoring:
  - Track recall@k
  - Monitor search latency (p50, p95, p99)
  - Memory usage
  - Index build time

✓ Common Use Cases:
  1. Product recommendations (e-commerce)
  2. Semantic search (documents, Q&A)
  3. Image similarity (visual search)
  4. User matching (social, dating apps)
  5. Anomaly detection
  6. Deduplication
  7. RAG (Retrieval-Augmented Generation)
    """)

    print("="*70)
    print("Hoàn thành tất cả examples! 🎉")
    print("="*70)
    print("\nTiếp theo:")
    print("  - Thử với dữ liệu thật của bạn")
    print("  - Tune parameters cho use case cụ thể")
    print("  - Deploy lên production với monitoring")
    print("  - Đọc FAISS docs: https://github.com/facebookresearch/faiss/wiki")
    print()


if __name__ == "__main__":
    main()