# FAISS Jupyter Notebooks - Hướng dẫn Trực quan

Bộ notebooks này cung cấp các ví dụ trực quan và tương tác để học FAISS.

## 📚 Danh sách Notebooks

### 1. `01_basic_search_visualization.ipynb`
**Mức độ**: Beginner
**Thời gian**: ~10 phút

**Nội dung**:
- Tìm kiếm vector cơ bản với IndexFlat
- Visualization vectors trong không gian 2D (PCA)
- Phân tích phân bố khoảng cách
- So sánh FAISS vs NumPy brute-force
- Performance benchmarking

**Bạn sẽ học được**:
- Cách hoạt động của vector search
- L2 distance và nearest neighbors
- Cách visualize high-dimensional vectors
- Đánh giá accuracy và speed

**Visualizations**:
- Distance distribution histograms
- Distance heatmaps
- 2D PCA projections
- Performance comparison charts

---

### 2. `02_ivf_clustering_visualization.ipynb`
**Mức độ**: Intermediate
**Thời gian**: ~15 phút

**Nội dung**:
- IVF (Inverted File) indexing
- Clustering và centroids
- Trade-off giữa nprobe và recall/speed
- Search coverage analysis
- So sánh IVF vs Flat

**Bạn sẽ học được**:
- Cách IVF chia database thành clusters
- Ảnh hưởng của nlist và nprobe
- Cách tune parameters cho best performance
- Trade-offs giữa accuracy và speed

**Visualizations**:
- Cluster size distributions
- 2D cluster visualization
- nprobe vs recall/QPS curves
- Search coverage analysis
- Pareto frontier charts

---

### 3. `03_index_comparison_visualization.ipynb`
**Mức độ**: Advanced
**Thời gian**: ~20 phút

**Nội dung**:
- So sánh toàn diện: Flat, IVF, PQ, IVF+PQ, HNSW
- Multi-metric analysis
- Pareto frontier analysis
- Decision guide cho việc chọn index
- Best practices và recommendations

**Bạn sẽ học được**:
- Strengths và weaknesses của mỗi index
- Khi nào dùng index nào
- Trade-offs giữa accuracy, speed, memory
- Production considerations

**Visualizations**:
- Comprehensive comparison charts
- Radar/spider charts
- Pareto frontier analysis
- Decision tree guide
- Multi-dimensional comparisons

---

### 4. `04_hnsw_visualization.ipynb`
**Mức độ**: Advanced
**Thời gian**: ~20 phút

**Nội dung**:
- HNSW (Hierarchical Navigable Small World) graph structure
- Visualization của hierarchical layers
- Trade-off giữa M, efConstruction, efSearch
- So sánh HNSW vs IVF
- Performance tuning guide

**Bạn sẽ học được**:
- Cách HNSW graph hoạt động
- Layer hierarchy và search process
- Parameter tuning cho optimal performance
- Khi nào nên dùng HNSW
- Memory vs accuracy trade-offs

**Visualizations**:
- HNSW layer distribution
- Hierarchical structure pyramid
- efSearch vs recall/speed trade-offs
- M parameter comparison
- HNSW vs IVF performance

---

### 5. `06_real_world_visualization.ipynb`
**Mức độ**: Production-Ready
**Thời gian**: ~25 phút

**Nội dung**:
- Product recommendation system (E-commerce)
- User-based recommendations
- Batch processing và optimization
- Production performance analysis
- Best practices và deployment patterns

**Bạn sẽ học được**:
- Build production-ready recommendation system
- Handle real-world data patterns
- Performance optimization techniques
- Monitoring và reliability patterns
- Scaling strategies

**Visualizations**:
- Product embeddings space
- Recommendation visualization
- User profile analysis
- Production performance metrics
- Latency và throughput charts

---

## 🚀 Cách sử dụng

### Yêu cầu

```bash
# Install dependencies
pip install numpy faiss-cpu matplotlib seaborn scikit-learn jupyter

# Hoặc dùng uv
uv pip install numpy faiss-cpu matplotlib seaborn scikit-learn jupyter
```

### Chạy Notebooks

```bash
# Start Jupyter
jupyter notebook

# Hoặc dùng JupyterLab
jupyter lab
```

Sau đó mở từng notebook trong thư mục `examples/`.

### Thứ tự học tập đề xuất

**Track 1 - Learning Path** (Học từ cơ bản đến nâng cao):
1. **Beginner**: `01_basic_search_visualization.ipynb` - Hiểu vector search cơ bản
2. **Intermediate**: `02_ivf_clustering_visualization.ipynb` - IVF clustering
3. **Advanced**: `03_index_comparison_visualization.ipynb` - So sánh tất cả indexes
4. **Expert**: `04_hnsw_visualization.ipynb` - HNSW deep dive

**Track 2 - Production Path** (Xây dựng ứng dụng thực tế):
1. Start: `01_basic_search_visualization.ipynb` - Foundation
2. Choose: `02_ivf_clustering_visualization.ipynb` OR `04_hnsw_visualization.ipynb`
3. Compare: `03_index_comparison_visualization.ipynb` - Pick best index
4. Deploy: `06_real_world_visualization.ipynb` - Production patterns

---

## 📊 Outputs

Mỗi notebook sẽ tạo ra các file visualizations:

### Notebook 01:
- `01_distance_analysis.png` - Phân tích phân bố khoảng cách
- `01_vector_space_2d.png` - Visualization vectors trong 2D
- `01_performance_comparison.png` - So sánh performance

### Notebook 02:
- `02_cluster_distribution.png` - Phân bố cluster sizes
- `02_ivf_clusters_2d.png` - Visualization clusters trong 2D
- `02_nprobe_tradeoff.png` - Trade-off curves
- `02_ivf_vs_flat_comparison.png` - So sánh detailed
- `02_search_coverage.png` - Coverage analysis

### Notebook 03:
- `03_comprehensive_comparison.png` - So sánh tất cả indexes
- `03_radar_comparison.png` - Radar chart
- `03_pareto_analysis.png` - Pareto frontier
- `03_decision_guide.png` - Decision tree

### Notebook 04:
- `04_hnsw_layers.png` - HNSW layer structure
- `04_efSearch_tradeoff.png` - efSearch parameter curves
- `04_efSearch_analysis.png` - Detailed efSearch analysis
- `04_M_comparison.png` - M parameter comparison
- `04_hnsw_vs_ivf.png` - HNSW vs IVF comparison

### Notebook 06:
- `06_product_embeddings.png` - Product embedding space
- `06_recommendation_visualization.png` - Recommendation results
- `06_production_performance.png` - Performance metrics

---

## 💡 Tips và Best Practices

### Performance
- **Dataset lớn**: Giảm `n_vectors` trong config để chạy nhanh hơn
- **Memory**: Close các figures sau khi show để giải phóng memory
- **Reproducibility**: Set `np.random.seed()` ở đầu mỗi notebook

### Visualization
- **High DPI**: Plots được save ở 300 DPI cho quality cao
- **Interactive**: Dùng `%matplotlib widget` cho interactive plots
- **Customization**: Modify color schemes và styles theo ý thích

### Learning
- **Run từng cell**: Đọc kỹ output và plots
- **Experiment**: Thay đổi parameters và xem kết quả
- **Compare**: So sánh với kết quả expected trong comments

---

## 🔧 Troubleshooting

### ImportError
```python
# Nếu không tìm thấy utils module
import sys
sys.path.append('../')  # Adjust path nếu cần
```

### Memory Issues
```python
# Giảm dataset size
n_vectors = 10000  # Thay vì 100000
n_queries = 50     # Thay vì 200
```

### Slow Performance
```python
# Skip một số experiments
# Comment out các cells nặng nếu chỉ muốn xem visualizations
```

---

## 📖 Tài liệu tham khảo

- **FAISS Documentation**: https://faiss.ai/
- **FAISS Wiki**: https://github.com/facebookresearch/faiss/wiki
- **Paper**: "Billion-scale similarity search with GPUs" (Johnson et al., 2017)

---

## 🤝 Đóng góp

Nếu bạn có ý tưởng cho notebooks mới hoặc improvements:
1. Fork repository
2. Tạo notebook mới
3. Submit pull request

### Ý tưởng cho notebooks tương lai:
- GPU acceleration with FAISS-GPU
- Image search với CLIP embeddings
- Text search với BERT/Sentence Transformers
- Advanced filtering và metadata
- Distributed FAISS với multiple machines
- Custom distance metrics
- Online learning và index updates

---

## ⚠️ Notes

- Notebooks sử dụng random data - kết quả sẽ khác với real-world data
- Always benchmark trên YOUR actual data
- Parameters trong notebooks là starting points, cần tune cho use case của bạn
- Visualizations work best với matplotlib backend 'inline' hoặc 'widget'

---

## 📞 Support

Nếu có vấn đề hoặc câu hỏi:
- Check FAISS documentation
- Review code comments trong notebooks
- Open GitHub issue

---

**Happy Learning! 🎓**

Chúc bạn học tốt FAISS và vector search!
