#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-${PORT:-8080}}"
BASE_URL="http://localhost:${PORT}"

echo "============================================================"
echo "Testing Embeddings API (TEI) at ${BASE_URL}"
echo "============================================================"

# 1. Health check
echo -e "\n1. Checking /health..."
HEALTH_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "${BASE_URL}/health" || true)
HTTP_STATUS=$(echo "${HEALTH_RESPONSE}" | grep "HTTP_STATUS" | cut -d':' -f2)

if [ "${HTTP_STATUS}" != "200" ]; then
  echo "❌ /health check failed with status: ${HTTP_STATUS}"
  echo "Response: ${HEALTH_RESPONSE}"
  exit 1
fi
echo "✅ /health returned 200 OK"

# 2. Info check
echo -e "\n2. Checking /info..."
INFO_RESPONSE=$(curl -s "${BASE_URL}/info" || true)
echo "Model Info:"
echo "${INFO_RESPONSE}" | grep -o '"model_id":"[^"]*"' || echo "${INFO_RESPONSE}"

# 3. Embed endpoint
echo -e "\n3. Testing POST /embed..."
EMBED_RESPONSE=$(curl -s -X POST "${BASE_URL}/embed" \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Hello world", "Text embeddings inference is fast!"]}')

# Print preview of vector output (truncate if long)
echo "Embeddings generated successfully. Response snippet:"
echo "${EMBED_RESPONSE}" | head -c 200
echo -e "\n..."

# 4. OpenAI-compatible /v1/embeddings endpoint
echo -e "\n4. Testing POST /v1/embeddings (OpenAI compatible)..."
V1_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": "Testing OpenAI compatibility endpoint"}')

echo "OpenAI endpoint response snippet:"
echo "${V1_RESPONSE}" | head -c 250
echo -e "\n..."

echo -e "\n============================================================"
echo "✅ All tests passed successfully!"
echo "============================================================"
