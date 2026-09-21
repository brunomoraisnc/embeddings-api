# Embeddings API

High-performance text embeddings service powered by [Hugging Face Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference).

## Overview

This project provides a production-ready setup for serving text embeddings using Hugging Face's TEI engine.
By default, it is configured with:
- **Model**: `Alibaba-NLP/gte-multilingual-base`
- **Image**: `ghcr.io/huggingface/text-embeddings-inference:cpu-1.7`
- **Platform**: `linux/amd64`
- **Data Type**: `float16`
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
MODEL_ID=Alibaba-NLP/gte-multilingual-base
DTYPE=float16
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

### Alternative: Run with `docker run`

You can also run the container directly without compose:
```bash
docker run --platform linux/amd64 \
  -p 8080:80 \
  -v $PWD/data:/data \
  --pull always \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 \
  --model-id Alibaba-NLP/gte-multilingual-base \
  --dtype float16
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

A test script is provided to verify that the service is running and all endpoints respond correctly:
```bash
./test_api.sh
# or specify a custom port:
./test_api.sh 8080
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