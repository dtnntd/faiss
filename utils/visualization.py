"""
Visualization utilities for FAISS examples
Công cụ trực quan hóa cho các ví dụ FAISS
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def plot_recall_vs_time(
    results: Dict[str, list],
    title: str = "Recall vs Search Time Trade-off",
    save_path: Optional[str] = None
) -> None:
    """
    Vẽ biểu đồ trade-off giữa recall và thời gian search

    Args:
        results: Dictionary chứa:
            - 'param_values': List các giá trị parameter
            - 'recalls': List các recall values
            - 'search_times' hoặc 'qps': Thời gian hoặc queries per second
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn để lưu biểu đồ (nếu không None)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    param_values = results.get('param_values', [])
    recalls = results.get('recalls', [])

    # Plot 1: Recall vs Parameter Value
    ax1.plot(param_values, recalls, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Parameter Value', fontsize=12)
    ax1.set_ylabel('Recall@k', fontsize=12)
    ax1.set_title('Recall vs Parameter', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Recall vs QPS or Search Time
    if 'qps' in results:
        qps = results['qps']
        ax2.plot(qps, recalls, 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Queries Per Second (QPS)', fontsize=12)
        ax2.set_ylabel('Recall@k', fontsize=12)
        ax2.set_title('Recall vs QPS Trade-off', fontsize=14)
    elif 'search_times' in results:
        search_times = results['search_times']
        ax2.plot(search_times, recalls, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Search Time (seconds)', fontsize=12)
        ax2.set_ylabel('Recall@k', fontsize=12)
        ax2.set_title('Recall vs Search Time', fontsize=14)

    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Đã lưu biểu đồ vào: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_memory_usage(
    index_names: List[str],
    memory_sizes: List[float],
    title: str = "Memory Usage Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Vẽ biểu đồ so sánh memory usage của các indexes

    Args:
        index_names: List tên các indexes
        memory_sizes: List kích thước bộ nhớ (MB)
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn để lưu biểu đồ (nếu không None)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    x_pos = np.arange(len(index_names))
    bars = ax.bar(x_pos, memory_sizes, alpha=0.8, edgecolor='black')

    # Color bars based on size
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(memory_sizes)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels on bars
    for i, (name, size) in enumerate(zip(index_names, memory_sizes)):
        ax.text(i, size + max(memory_sizes) * 0.02,
                f'{size:.2f} MB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Index Type', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(index_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add a horizontal line for reference (e.g., original size)
    if len(memory_sizes) > 0:
        ax.axhline(y=memory_sizes[0], color='r', linestyle='--',
                   alpha=0.5, label=f'Baseline: {memory_sizes[0]:.2f} MB')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Đã lưu biểu đồ vào: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_index_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = ['recall', 'qps'],
    title: str = "Index Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Vẽ biểu đồ so sánh nhiều metrics của các indexes

    Args:
        results: Dictionary {index_name: {metric: value}}
        metrics: List các metrics cần plot
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn để lưu biểu đồ (nếu không None)
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    index_names = list(results.keys())

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[name].get(metric, 0) for name in index_names]

        x_pos = np.arange(len(index_names))
        bars = ax.bar(x_pos, values, alpha=0.8, edgecolor='black')

        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.02,
                    f'{v:.3f}' if v < 100 else f'{v:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Index Type', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(index_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Đã lưu biểu đồ vào: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("Testing visualization utilities...")

    # Test plot_recall_vs_time
    test_results = {
        'param_values': [1, 2, 4, 8, 16, 32],
        'recalls': [0.5, 0.65, 0.75, 0.85, 0.92, 0.96],
        'qps': [1500, 1200, 900, 600, 400, 250],
        'search_times': [0.067, 0.083, 0.111, 0.167, 0.250, 0.400]
    }

    print("\nTesting plot_recall_vs_time...")
    plot_recall_vs_time(test_results, title="Test: Recall vs Time")

    # Test plot_memory_usage
    test_names = ['Flat', 'IVF', 'PQ', 'IVF+PQ', 'HNSW']
    test_sizes = [512.0, 520.5, 64.0, 72.5, 380.2]

    print("\nTesting plot_memory_usage...")
    plot_memory_usage(test_names, test_sizes, title="Test: Memory Usage")

    # Test plot_index_comparison
    test_comparison = {
        'Flat': {'recall': 1.0, 'qps': 150},
        'IVF': {'recall': 0.95, 'qps': 800},
        'HNSW': {'recall': 0.98, 'qps': 1200}
    }

    print("\nTesting plot_index_comparison...")
    plot_index_comparison(test_comparison, metrics=['recall', 'qps'])

    print("\n✓ All visualization tests completed!")
