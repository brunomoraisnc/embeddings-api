# Embeddings API

High-performance text embeddings service powered by [Hugging Face Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference).

## Overview

This project provides a production-ready setup for serving text embeddings using Hugging Face's TEI engine.
By default, it is configured with:
- **Model**: `onnx-community/gte-multilingual-base` (ONNX-optimized for high-speed, reliable CPU inference)
- **Image**: `ghcr.io/huggingface/text-embeddings-inference:cpu-1.7`
- **Platform**: `linux/amd64`
- **Data Type**: `float32`
- **Pooling**: `cls`
- **Port**: `8080`
- **Volume Mount**: `./data:/data` (caches model weights locally to avoid re-downloading on container restarts)

---

## Quick Start

### 1. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

You can customize `.env` as needed:
```env
PORT=8080
MODEL_ID=onnx-community/gte-multilingual-base
DTYPE=float32
POOLING=cls
TEI_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-1.7
# HF_TOKEN=hf_xxx (only required for gated/private models)
```

### 2. Start with Docker Compose

To start the service in detached mode:
```bash
docker compose up -d
```

To view the logs:
```bash
docker compose logs -f
```

To stop the service:
```bash
docker compose down
```

### 3. Expected Deployment Output

Once started, the container initializes the ONNX Runtime engine, warms up, and reports healthy.

Checking the container status:
```bash
docker compose ps
```
**Expected Output:**
```text
NAME             IMAGE                                                   COMMAND                  SERVICE          STATUS                    PORTS
embeddings-api   ghcr.io/huggingface/text-embeddings-inference:cpu-1.7   "text-embeddings-rou…"   embeddings-api   Up (healthy)              0.0.0.0:8080->80/tcp
```

Viewing the container logs:
```bash
docker compose logs --tail 20
```
**Expected Output:**
```text
INFO text_embeddings_backend: backends/src/lib.rs:363: Model ONNX weights downloaded
INFO text_embeddings_router: router/src/lib.rs:252: Warming up model
INFO text_embeddings_router::http::server: router/src/http/server.rs:1847: Starting HTTP server: 0.0.0.0:80
INFO text_embeddings_router::http::server: router/src/http/server.rs:1848: Ready
```

---

### Alternative: Run with `docker run`

You can also run the container directly without compose:
```bash
docker run --platform linux/amd64 \
  -p 8080:80 \
  -v $PWD/data:/data \
  --pull always \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 \
  --model-id onnx-community/gte-multilingual-base \
  --dtype float32 \
  --pooling cls \
  --max-batch-tokens 2048 \
  --max-client-batch-size 8
```

### Optional: GPU Acceleration

If deploying to a machine with NVIDIA GPUs and the NVIDIA Container Toolkit:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Building the Custom Dockerfile

To build and run the provided [Dockerfile](file:///Users/brunomorais22/Documents/projects/embeddings-api/Dockerfile):
```bash
docker build -t embeddings-api .
docker run -p 8080:80 -v $PWD/data:/data embeddings-api
```

---

## Testing the API

### 1. Basic Endpoint Test (`test_api.sh`)

A shell script to quickly verify that the service is running and all endpoints respond:
```bash
./test_api.sh
# or specify a custom port:
./test_api.sh 8080
```

**Expected Output:**
```text
============================================================
Testing Embeddings API (TEI) at http://localhost:8080
============================================================

1. Checking /health...
✅ /health returned 200 OK

2. Checking /info...
Model Info:
"model_id":"onnx-community/gte-multilingual-base"

3. Testing POST /embed...
Embeddings generated successfully. Response snippet:
[[-0.027042571,-0.044719003,-0.034720976,0.005166299,0.037887368,...

4. Testing POST /v1/embeddings (OpenAI compatible)...
OpenAI endpoint response snippet:
{"object":"list","data":[{"object":"embedding","embedding":[-0.027042571,-0.044719003,...

============================================================
✅ All tests passed successfully!
============================================================
```

### 2. Enriched LangChain Test Suite (`test_api_with_langchain.py`)

A comprehensive Python test suite using LangChain to validate:
- Single query embeddings (`embed_query`)
- Batch document embeddings (`embed_documents`)
- Vector dimensions, L2 normalization, and latency benchmarks
- Vector diversity (flags if model returns identical/trivial vectors)
- Semantic similarity ranking (related vs. unrelated texts)
- Cross-lingual semantic alignment (multilingual texts)
- End-to-end vector store retrieval using `InMemoryVectorStore`

To run:
```bash
# Install dependencies
pip install langchain-openai langchain-core huggingface_hub

# Run tests with langchain-openai (default)
python3 test_api_with_langchain.py

# Run tests with both langchain-openai and native huggingface_hub
python3 test_api_with_langchain.py --provider both

# Custom host or model
python3 test_api_with_langchain.py --base-url http://localhost:8080 --model onnx-community/gte-multilingual-base
```

**Expected Output:**
```text
======================================================================
🧪 Embeddings API - LangChain Enriched Test Suite
   Target URL: http://localhost:8080
   Model ID:   onnx-community/gte-multilingual-base
   Provider:   both
======================================================================

======================================================================
🚀 Running Enriched Tests with Provider: langchain-openai (OpenAIEmbeddings)
======================================================================

[Test 1] Single Query Embedding (embed_query)...
  • Dimensions: 768
  • L2 Norm: 1.0000
  • Latency: ~50-150 ms
  ✅ PASS: Single query embedding generated successfully.

[Test 2] Batch Document Embedding (embed_documents)...
  • Batch count: 4 / 4
  • Total latency: ~130-160 ms (avg ~35-40 ms/doc)
  ✅ PASS: All documents embedded with consistent dimensions.

[Test 3] Vector Diversity & Non-Triviality Check...
  • Cosine similarity between doc 1 (AI) and doc 2 (NLP): 0.7098
  • Cosine similarity between doc 1 (AI) and doc 4 (Docker): 0.5077
  ✅ PASS: Vectors are diverse and non-identical across distinct inputs.

[Test 4] Semantic Similarity Evaluation...
  • Anchor:    'The canine barked loudly at the mail carrier.'
  • Related:   'A dog was barking at the postal worker outside.' -> Similarity: 0.8918
  • Unrelated: 'The planetary orbit of Neptune takes about 165 Earth years.' -> Similarity: 0.4125
  ✅ PASS: Related sentence has higher similarity (+0.4793).

[Test 5] Multilingual Semantic Alignment Test...
  Cross-lingual similarity against English greeting:
    - Spanish     : 0.8412 ('Hola, ¿cómo estás hoy?')
    - Portuguese  : 0.9247 ('Olá, como você está hoje?')
    - French      : 0.8501 ('Bonjour, comment allez-vous aujourd'hui?')
    - German      : 0.9175 ('Hallo, wie geht es dir heute?')
    - Unrelated   : 0.3129 ('Quantum computing leverages superposition and entanglement.')
  ✅ PASS: Cross-lingual greetings have higher similarity than unrelated text.

[Test 6] End-to-End VectorStore Retrieval (InMemoryVectorStore)...
  • Query: 'deploying models with fast inference'
  • Top-1 Match: 'Hugging Face Text Embeddings Inference enables high-throughput vector serving.'
  • Top-2 Match: 'Kubernetes is a system for automating deployment, scaling, and container operations.'
  ✅ PASS: VectorStore retrieved the expected top document!

======================================================================
📋 Final Test Summary
======================================================================
  • langchain-openai              : ✅ PASSED
  • huggingface_hub               : ✅ PASSED

🎉 All LangChain integration tests passed successfully!
```

---

## API Endpoints & Usage

### 1. Health Check
```bash
curl http://localhost:8080/health
```

### 2. Model Info
```bash
curl http://localhost:8080/info
```

### 3. Generate Embeddings (`/embed`)

#### Using `curl`
```bash
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["What is machine learning?", "Deep learning powers AI."]}'
```

#### Using Python (`requests`)
```python
import requests

url = "http://localhost:8080/embed"
payload = {
    "inputs": [
        "What is machine learning?",
        "Deep learning powers AI."
    ]
}

response = requests.post(url, json=payload)
embeddings = response.json()
print(f"Generated {len(embeddings)} vectors with dimension {len(embeddings[0])}")
```

### 4. OpenAI-Compatible Embeddings (`/v1/embeddings`)

TEI exposes an OpenAI-compatible endpoint, making it a drop-in replacement for OpenAI embeddings.

#### Using `curl`
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Alibaba-NLP/gte-multilingual-base",
    "input": "Embed this text using OpenAI client interface"
  }'
```

#### Using `openai` Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="empty"  # TEI does not require an API key by default
)

response = client.embeddings.create(
    model="Alibaba-NLP/gte-multilingual-base",
    input="Embed this sentence with TEI via OpenAI SDK"
)

vector = response.data[0].embedding
print(f"Embedding dimension: {len(vector)}")
```

### 5. Using with LangChain

Hugging Face TEI provides an OpenAI-compatible `/v1/embeddings` endpoint. In LangChain, the recommended way to connect to a self-hosted TEI server is using `langchain-openai` (since `langchain-huggingface`'s `HuggingFaceEndpointEmbeddings` enforces Hugging Face Hub repo IDs and rejects custom URLs).

#### Option A: Using `langchain-openai` (Recommended)

Install the dependency:
```bash
pip install langchain-openai
```

Use `OpenAIEmbeddings` configured with your local TEI endpoint:
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:8080/v1",
    api_key="empty",  # TEI does not require an API key by default
    model="Alibaba-NLP/gte-multilingual-base",
    check_embedding_ctx_length=False
)

# Embed a single query
query_vector = embeddings.embed_query("What is machine learning?")
print(f"Query vector dimension: {len(query_vector)}")  # 768

# Embed multiple documents
doc_vectors = embeddings.embed_documents([
    "What is machine learning?",
    "Deep learning powers modern AI."
])
print(f"Generated {len(doc_vectors)} document vectors with dimension {len(doc_vectors[0])}")
```

#### Option B: Using `huggingface_hub` with LangChain `Embeddings`

If you prefer using Hugging Face's native client without `langchain-openai`:

Install dependencies:
```bash
pip install huggingface_hub langchain-core
```

Use a lightweight LangChain `Embeddings` wrapper:
```python
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

class TEIEmbeddings(Embeddings):
    """LangChain Embeddings wrapper for Hugging Face Text Embeddings Inference (TEI)."""

    def __init__(self, endpoint_url: str = "http://localhost:8080"):
        self.client = InferenceClient(endpoint_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.feature_extraction(texts)
        return response.tolist()

    def embed_query(self, text: str) -> list[float]:
        response = self.client.feature_extraction(text)
        # TEI returns a 2D array [1, dim] for a single string input
        res_list = response.tolist()
        return res_list[0] if isinstance(res_list[0], list) else res_list

# Usage
embeddings = TEIEmbeddings("http://localhost:8080")

query_vector = embeddings.embed_query("What is machine learning?")
print(f"Query vector dimension: {len(query_vector)}")  # 768

doc_vectors = embeddings.embed_documents([
    "What is machine learning?",
    "Deep learning powers modern AI."
])
print(f"Generated {len(doc_vectors)} document vectors with dimension {len(doc_vectors[0])}")
```

---

## Project Structure

```
.
├── .env.example          # Environment variables template
├── .env                  # Local environment configuration
├── .gitignore            # Git ignore rules (ignores data/ and .env)
├── docker-compose.yml    # Base Docker Compose service definition
├── docker-compose.gpu.yml# Docker Compose GPU override
├── Dockerfile            # Container build specification
├── test_api.sh           # Test script for API verification
└── README.md             # Project documentation
```