# FAISS Examples - Installation Guide

## Quick Install

### Using uv (Recommended - Fast!)
```bash
# Install uv if not already installed
pip install uv

# Install all dependencies
uv pip install -e .

# Or install from requirements.txt
uv pip install -r requirements.txt
```

### Using pip
```bash
# Install from pyproject.toml
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

## Dependencies Overview

### Core Libraries
- **faiss-cpu** (or faiss-gpu): Vector similarity search
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities

### Visualization
- **matplotlib**: Plotting library
- **seaborn**: Statistical visualization

### Notebooks
- **jupyter**: Interactive notebooks
- **notebook**: Jupyter notebook interface
- **ipykernel**: Jupyter kernel

### Vietnamese Search Demo
- **google-generativeai**: Google embedding API
- **python-dotenv**: Environment variables

## Installation by Use Case

### 1. Basic FAISS Examples (01-04)
```bash
pip install faiss-cpu numpy matplotlib seaborn scikit-learn jupyter
```

### 2. Vietnamese Text Search (Example 07)
```bash
pip install faiss-cpu numpy pandas jupyter google-generativeai python-dotenv
```

### 3. Everything
```bash
pip install -r requirements.txt
```

## GPU Support

If you have CUDA-capable GPU:

```bash
# Uninstall CPU version
pip uninstall faiss-cpu

# Install GPU version
pip install faiss-gpu

# Check CUDA version compatibility
# FAISS requires CUDA 11.x or 12.x
```

## Verification

```python
# Test FAISS installation
import faiss
print(f"FAISS version: {faiss.__version__}")

# Test if GPU is available (faiss-gpu only)
print(f"GPU available: {faiss.get_num_gpus()}")

# Test Google API (for Vietnamese search)
import google.generativeai as genai
print("Google Generative AI installed successfully")
```

## Troubleshooting

### Issue: FAISS installation fails
**Solution**:
```bash
# Try specific version
pip install faiss-cpu==1.7.4

# Or use conda
conda install -c pytorch faiss-cpu
```

### Issue: Google API import error
**Solution**:
```bash
pip install --upgrade google-generativeai
```

### Issue: Jupyter kernel not found
**Solution**:
```bash
python -m ipykernel install --user --name faiss --display-name "Python (FAISS)"
```

## Development Setup

For contributors:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or manually install dev tools
pip install black isort flake8 mypy pytest
```

## Minimum Requirements

- Python >= 3.8 (3.11+ recommended)
- pip >= 21.0
- Operating System: Linux, macOS, or Windows

## Recommended Setup

- Python 3.11+
- 8GB RAM (16GB+ for large datasets)
- SSD for faster I/O
- GPU with CUDA 11+ (optional, for faiss-gpu)
