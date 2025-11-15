"""
Vietnamese Text Embedding với Google API
Sử dụng Google Generative AI để tạo embeddings cho văn bản tiếng Việt
"""

import os
import numpy as np
from typing import List
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()


def embed_with_google_api(texts: List[str], model_name: str = "models/text-embedding-004") -> np.ndarray:
    """
    Embed văn bản bằng Google Generative AI API

    Args:
        texts: List các văn bản cần embed
        model_name: Tên model embedding
            - "models/text-embedding-004": Latest, 768 dimensions
            - "models/embedding-001": Legacy, 768 dimensions

    Returns:
        numpy array of embeddings, shape (n_texts, embedding_dim)
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "Please install google-generativeai: pip install google-generativeai"
        )

    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment. "
            "Please add it to .env file or set as environment variable."
        )

    # Configure API
    genai.configure(api_key=api_key)

    print(f"Using model: {model_name}")
    print(f"Embedding {len(texts)} texts...")

    embeddings = []
    batch_size = 100  # Google API recommends batching

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            # Embed batch
            result = genai.embed_content(
                model=model_name,
                content=batch,
                task_type="retrieval_document",  # For indexing documents
                title="Vietnamese Text Corpus"
            )

            batch_embeddings = result['embedding']
            embeddings.extend(batch_embeddings)

            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

            # Rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.5)  # Avoid hitting rate limits

        except Exception as e:
            print(f"Error embedding batch {i//batch_size + 1}: {e}")
            raise

    embeddings_array = np.array(embeddings, dtype='float32')
    print(f"✓ Created embeddings: {embeddings_array.shape}")

    return embeddings_array


def embed_query(query: str, model_name: str = "models/text-embedding-004") -> np.ndarray:
    """
    Embed một query để search

    Args:
        query: Query text
        model_name: Tên model

    Returns:
        numpy array embedding, shape (1, embedding_dim)
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "Please install google-generativeai: pip install google-generativeai"
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found")

    genai.configure(api_key=api_key)

    result = genai.embed_content(
        model=model_name,
        content=query,
        task_type="retrieval_query"  # For search queries
    )

    embedding = np.array([result['embedding']], dtype='float32')
    return embedding


def create_and_save_embeddings(
    input_file: str = "vietnamese_dataset.txt",
    output_file: str = "vietnamese_embeddings.npy",
    model_name: str = "models/text-embedding-004"
):
    """
    Đọc dataset, tạo embeddings và lưu file

    Args:
        input_file: File chứa văn bản (mỗi dòng 1 chunk)
        output_file: File để lưu embeddings
        model_name: Tên model Google
    """
    # Read texts
    data_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(data_dir, input_file)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Dataset not found: {input_path}\n"
            "Run vietnamese_dataset_generator.py first!"
        )

    print(f"Reading texts from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"✓ Loaded {len(texts)} texts")

    # Create embeddings
    print(f"\nCreating embeddings with Google API...")
    embeddings = embed_with_google_api(texts, model_name)

    # Save embeddings
    output_path = os.path.join(data_dir, output_file)
    np.save(output_path, embeddings)
    print(f"✓ Saved embeddings to {output_path}")

    # Save texts separately for reference
    texts_file = output_file.replace('.npy', '_texts.txt')
    texts_path = os.path.join(data_dir, texts_file)
    with open(texts_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    print(f"✓ Saved texts to {texts_path}")

    # Print stats
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total texts: {len(texts)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Total size: {embeddings.nbytes / (1024**2):.2f} MB")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")

    return embeddings, texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed Vietnamese text với Google API")
    parser.add_argument(
        "--input",
        default="vietnamese_dataset.txt",
        help="Input text file"
    )
    parser.add_argument(
        "--output",
        default="vietnamese_embeddings.npy",
        help="Output embeddings file"
    )
    parser.add_argument(
        "--model",
        default="models/text-embedding-004",
        choices=["models/text-embedding-004", "models/embedding-001"],
        help="Google embedding model"
    )

    args = parser.parse_args()

    try:
        create_and_save_embeddings(args.input, args.output, args.model)
        print("\n✅ Embedding complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Created .env file with GOOGLE_API_KEY")
        print("  2. Installed: pip install google-generativeai python-dotenv")
        print("  3. Generated dataset: python vietnamese_dataset_generator.py")
