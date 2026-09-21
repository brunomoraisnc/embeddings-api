# GEMINI.md

This document provides project context, architecture guidelines, and operational commands for AI agents and developers working on the `embeddings-api` codebase.

---

## 1. Project Overview

`embeddings-api` is a containerized, production-ready text embeddings service powered by [Hugging Face Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference).

- **Default Model**: `Alibaba-NLP/gte-multilingual-base` (768 dimensions, 8192 context window, 70+ languages).
- **Default Image**: `ghcr.io/huggingface/text-embeddings-inference:cpu-1.7` (with `linux/amd64` platform support).
- **Default Port**: `8080` (mapped to container port `80`).
- **Protocols Exposed**:
  - TEI native REST API (`/embed`, `/tokenize`, `/info`, `/health`).
  - OpenAI-compatible embeddings API (`/v1/embeddings`).

---

## 2. Repository Structure

```
.
├── .env                  # Active environment variables (git-ignored)
├── .env.example          # Template for environment configuration
├── .gitignore            # Ignores .env, data/, and temporary files
├── docker-compose.yml    # Main Docker Compose configuration (CPU deployment)
├── docker-compose.gpu.yml# Docker Compose override for NVIDIA GPU deployment
├── Dockerfile            # Standalone container build definition
├── test_api.sh           # Basic bash test script for health and endpoint validation
├── test_api_with_langchain.py # Enriched LangChain test suite (semantic & multilingual tests)
├── README.md             # Public documentation and usage guide
└── GEMINI.md             # Agent and developer context guide
```

---

## 3. Environment & Configuration

All runtime configurations are governed by `.env`:

| Variable | Default Value | Description |
|---|---|---|
| `PORT` | `8080` | Host port mapped to container port 80. |
| `MODEL_ID` | `Alibaba-NLP/gte-multilingual-base` | Hugging Face Hub model identifier or path. |
| `DTYPE` | `float16` | Model data type precision (`float16` or `float32`). |
| `TEI_IMAGE` | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.7` | Base TEI container image. |
| `MAX_BATCH_TOKENS` | `2048` | Max tokens per batch (prevents memory spikes during warmup). |
| `MAX_CLIENT_BATCH_SIZE` | `8` | Maximum inputs a client can send in one request. |
| `HF_TOKEN` | *(empty)* | Hugging Face token (only required for private or gated models). |

---

## 4. Key Commands

### Service Management
```bash
# Start service in background
docker compose up -d

# Follow container logs
docker compose logs -f

# Check container status
docker compose ps

# Stop service
docker compose down

# Deploy with NVIDIA GPU acceleration
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Direct Docker Run
```bash
docker run --platform linux/amd64 \
  -p 8080:80 \
  -v $PWD/data:/data \
  --pull always \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 \
  --model-id Alibaba-NLP/gte-multilingual-base \
  --dtype float16 \
  --max-batch-tokens 2048 \
  --max-client-batch-size 8
```

### Verification & Testing
```bash
# 1. Quick shell test (checks /health, /info, /embed, /v1/embeddings)
./test_api.sh

# 2. Enriched LangChain test suite
python3 test_api_with_langchain.py

# 3. Test both langchain-openai and native huggingface_hub providers
python3 test_api_with_langchain.py --provider both
```

---

## 5. Important Architectural Notes & Gotchas

1. **Docker Desktop & Memory Spikes (`MAX_BATCH_TOKENS`)**:
   - By default, TEI uses `max_batch_tokens: 16384`. During model warmup on startup, this allocates multi-gigabyte attention buffers.
   - On CPU and virtualized environments (like Docker Desktop VM with ~8GB RAM), the default causes Linux OOM kills (`ExitCode: 137`).
   - Setting `MAX_BATCH_TOKENS=2048` and `MAX_CLIENT_BATCH_SIZE=8` resolves this issue.

2. **Candle CPU Backend & Intel MKL `HGEMM`**:
   - TEI falls back to its Rust Candle backend when ONNX models are not present.
   - On CPU with `Alibaba-NLP/gte-multilingual-base` and `--dtype float16`, Candle calls Intel MKL `HGEMM` which outputs `Parameter 10 was incorrect on entry to HGEMM`.
   - On NVIDIA GPUs (CUDA) or with models containing native ONNX weights (e.g., `onnx-community/gte-multilingual-base`), the engine uses ONNX Runtime (ORT) or CUDA without MKL errors.

3. **LangChain Integration**:
   - `langchain-huggingface`'s `HuggingFaceEndpointEmbeddings` explicitly rejects URLs in recent versions via Pydantic validation (`model must be a HuggingFace repo ID, not a URL`).
   - **Recommended LangChain class**: `OpenAIEmbeddings` from `langchain-openai` pointing to `base_url="http://localhost:8080/v1"` with `check_embedding_ctx_length=False`.
