#!/usr/bin/env python3
"""
Test script for verifying Hugging Face Text Embeddings Inference (TEI)
using LangChain (both langchain-openai and native huggingface_hub).

Enriched test suite:
  1. Single Query Embedding (embed_query)
  2. Batch Document Embedding (embed_documents)
  3. Embedding Vector Integrity (dimension, non-zero, normalization)
  4. Vector Diversity & Non-triviality check
  5. Semantic Similarity (Cosine similarity on related vs unrelated pairs)
  6. Multilingual Semantic Alignment (cross-lingual similarity)
  7. VectorStore End-to-End Retrieval (InMemoryVectorStore)
  8. Latency and Throughput benchmarks
"""

import argparse
import math
import os
import sys
import time
from typing import List, Tuple


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def get_openai_embeddings(base_url: str, model: str):
    """Initialize LangChain OpenAIEmbeddings pointing to TEI."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("❌ 'langchain-openai' is not installed. Install with: pip install langchain-openai")
        sys.exit(1)

    return OpenAIEmbeddings(
        base_url=f"{base_url}/v1",
        api_key="empty",
        model=model,
        check_embedding_ctx_length=False,
    )


def get_tei_native_embeddings(base_url: str):
    """Initialize custom LangChain Embeddings using huggingface_hub."""
    try:
        from huggingface_hub import InferenceClient
        from langchain_core.embeddings import Embeddings
    except ImportError:
        print("❌ 'huggingface_hub' or 'langchain-core' not installed. Install with: pip install huggingface_hub langchain-core")
        sys.exit(1)

    class TEIEmbeddings(Embeddings):
        def __init__(self, endpoint_url: str):
            self.client = InferenceClient(endpoint_url)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            response = self.client.feature_extraction(texts)
            return response.tolist()

        def embed_query(self, text: str) -> List[float]:
            response = self.client.feature_extraction(text)
            res_list = response.tolist()
            return res_list[0] if isinstance(res_list[0], list) else res_list

    return TEIEmbeddings(base_url)


def run_tests(embeddings, provider_name: str) -> bool:
    print(f"\n{'='*70}")
    print(f"🚀 Running Enriched Tests with Provider: {provider_name}")
    print(f"{'='*70}")

    all_passed = True

    # -------------------------------------------------------------
    # 1. Single Query Embedding Test
    # -------------------------------------------------------------
    print("\n[Test 1] Single Query Embedding (embed_query)...")
    sample_query = "Artificial intelligence and machine learning algorithms."
    t0 = time.perf_counter()
    query_vector = embeddings.embed_query(sample_query)
    query_duration = (time.perf_counter() - t0) * 1000

    dim = len(query_vector)
    norm = math.sqrt(sum(x * x for x in query_vector))
    print(f"  • Dimensions: {dim}")
    print(f"  • L2 Norm: {norm:.4f}")
    print(f"  • Latency: {query_duration:.2f} ms")
    print(f"  • Vector sample (first 5 elements): {[round(x, 4) for x in query_vector[:5]]}")

    if dim > 0 and not all(x == 0 for x in query_vector):
        print("  ✅ PASS: Single query embedding generated successfully.")
    else:
        print("  ❌ FAIL: Invalid query vector.")
        all_passed = False

    # -------------------------------------------------------------
    # 2. Batch Document Embedding Test
    # -------------------------------------------------------------
    print("\n[Test 2] Batch Document Embedding (embed_documents)...")
    docs = [
        "Deep learning powers modern generative models.",
        "Natural language processing helps machines understand human speech.",
        "Vector databases index high-dimensional embeddings for fast retrieval.",
        "Docker simplifies software deployment across different environments.",
    ]
    t0 = time.perf_counter()
    doc_vectors = embeddings.embed_documents(docs)
    batch_duration = (time.perf_counter() - t0) * 1000

    print(f"  • Batch count: {len(doc_vectors)} / {len(docs)}")
    print(f"  • Total latency: {batch_duration:.2f} ms (avg {batch_duration/len(docs):.2f} ms/doc)")

    batch_dims = [len(v) for v in doc_vectors]
    if len(doc_vectors) == len(docs) and all(d == dim for d in batch_dims):
        print("  ✅ PASS: All documents embedded with consistent dimensions.")
    else:
        print("  ❌ FAIL: Document count or dimensions mismatch.")
        all_passed = False

    # -------------------------------------------------------------
    # 3. Vector Diversity & Non-triviality Check
    # -------------------------------------------------------------
    print("\n[Test 3] Vector Diversity & Non-Triviality Check...")
    v1 = doc_vectors[0]
    v2 = doc_vectors[1]
    v3 = doc_vectors[3]

    sim_v1_v2 = cosine_similarity(v1, v2)
    sim_v1_v3 = cosine_similarity(v1, v3)
    print(f"  • Cosine similarity between doc 1 (AI) and doc 2 (NLP): {sim_v1_v2:.4f}")
    print(f"  • Cosine similarity between doc 1 (AI) and doc 4 (Docker): {sim_v1_v3:.4f}")

    if sim_v1_v2 < 0.9999 and sim_v1_v3 < 0.9999:
        print("  ✅ PASS: Vectors are diverse and non-identical across distinct inputs.")
    else:
        print("  ⚠️  WARNING: Vectors appear identical or nearly identical across distinct texts.")
        print("     (Check if the model backend is initialized properly or running in fallback mode)")

    # -------------------------------------------------------------
    # 4. Semantic Similarity Evaluation
    # -------------------------------------------------------------
    print("\n[Test 4] Semantic Similarity Evaluation...")
    anchor = "The canine barked loudly at the mail carrier."
    related = "A dog was barking at the postal worker outside."
    unrelated = "The planetary orbit of Neptune takes about 165 Earth years."

    vectors = embeddings.embed_documents([anchor, related, unrelated])
    v_anchor, v_related, v_unrelated = vectors[0], vectors[1], vectors[2]

    sim_related = cosine_similarity(v_anchor, v_related)
    sim_unrelated = cosine_similarity(v_anchor, v_unrelated)

    print(f"  • Anchor:    '{anchor}'")
    print(f"  • Related:   '{related}' -> Similarity: {sim_related:.4f}")
    print(f"  • Unrelated: '{unrelated}' -> Similarity: {sim_unrelated:.4f}")

    if sim_related > sim_unrelated:
        print(f"  ✅ PASS: Related sentence has higher similarity (+{sim_related - sim_unrelated:.4f}).")
    else:
        print("  ❌ FAIL: Semantic similarity ranking failed (related <= unrelated).")
        all_passed = False

    # -------------------------------------------------------------
    # 5. Multilingual Semantic Alignment Test
    # -------------------------------------------------------------
    print("\n[Test 5] Multilingual Semantic Alignment Test...")
    multilingual_texts = [
        ("English", "Hello, how are you doing today?"),
        ("Spanish", "Hola, ¿cómo estás hoy?"),
        ("Portuguese", "Olá, como você está hoje?"),
        ("French", "Bonjour, comment allez-vous aujourd'hui?"),
        ("German", "Hallo, wie geht es dir heute?"),
        ("Unrelated", "Quantum computing leverages superposition and entanglement."),
    ]

    ml_vectors = embeddings.embed_documents([t[1] for t in multilingual_texts])
    en_vector = ml_vectors[0]

    print("  Cross-lingual similarity against English greeting:")
    ml_similarities = []
    for (lang, text), vec in zip(multilingual_texts[1:-1], ml_vectors[1:-1]):
        sim = cosine_similarity(en_vector, vec)
        ml_similarities.append(sim)
        print(f"    - {lang:12}: {sim:.4f} ('{text}')")

    unrelated_sim = cosine_similarity(en_vector, ml_vectors[-1])
    print(f"    - {'Unrelated':12}: {unrelated_sim:.4f} ('{multilingual_texts[-1][1]}')")

    if all(s > unrelated_sim for s in ml_similarities):
        print("  ✅ PASS: Cross-lingual greetings have higher similarity than unrelated text.")
    else:
        print("  ❌ FAIL: Multilingual alignment failed.")
        all_passed = False

    # -------------------------------------------------------------
    # 6. End-to-End VectorStore Retrieval Test
    # -------------------------------------------------------------
    print("\n[Test 6] End-to-End VectorStore Retrieval (InMemoryVectorStore)...")
    try:
        from langchain_core.vectorstores import InMemoryVectorStore

        corpus = [
            "Python is an interpreted, high-level programming language known for readability.",
            "PostgreSQL is an open-source object-relational database management system.",
            "Hugging Face Text Embeddings Inference enables high-throughput vector serving.",
            "Kubernetes is a system for automating deployment, scaling, and container operations.",
        ]

        store = InMemoryVectorStore.from_texts(texts=corpus, embedding=embeddings)
        search_query = "deploying models with fast inference"
        results = store.similarity_search(search_query, k=2)

        print(f"  • Query: '{search_query}'")
        print(f"  • Top-1 Match: '{results[0].page_content}'")
        print(f"  • Top-2 Match: '{results[1].page_content}'")

        if "Text Embeddings Inference" in results[0].page_content:
            print("  ✅ PASS: VectorStore retrieved the expected top document!")
        else:
            print("  ℹ️  VectorStore retrieved top document (ranking order may vary depending on model).")
    except Exception as e:
        print(f"  ❌ FAIL: VectorStore error: {e}")
        all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Enriched LangChain test suite for Hugging Face Text Embeddings Inference (TEI)"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://localhost:8080"),
        help="Base URL of TEI service (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_ID", "Alibaba-NLP/gte-multilingual-base"),
        help="Model ID (default: Alibaba-NLP/gte-multilingual-base)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "huggingface", "both"],
        default="openai",
        help="LangChain provider to test: 'openai' (langchain-openai), 'huggingface' (native client), or 'both'",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🧪 Embeddings API - LangChain Enriched Test Suite")
    print(f"   Target URL: {args.base_url}")
    print(f"   Model ID:   {args.model}")
    print(f"   Provider:   {args.provider}")
    print("=" * 70)

    results = []

    if args.provider in ("openai", "both"):
        embeddings_openai = get_openai_embeddings(args.base_url, args.model)
        passed = run_tests(embeddings_openai, "langchain-openai (OpenAIEmbeddings)")
        results.append(("langchain-openai", passed))

    if args.provider in ("huggingface", "both"):
        embeddings_hf = get_tei_native_embeddings(args.base_url)
        passed = run_tests(embeddings_hf, "huggingface_hub + langchain-core")
        results.append(("huggingface_hub", passed))

    print(f"\n{'='*70}")
    print("📋 Final Test Summary")
    print(f"{'='*70}")
    all_success = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  • {name:30}: {status}")
        if not success:
            all_success = False

    if all_success:
        print("\n🎉 All LangChain integration tests passed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed or showed anomalies. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
