FROM --platform=linux/amd64 ghcr.io/huggingface/text-embeddings-inference:cpu-1.7

ARG MODEL_ID=Alibaba-NLP/gte-multilingual-base
ARG DTYPE=float16

ENV MODEL_ID=${MODEL_ID}
ENV DTYPE=${DTYPE}

VOLUME ["/data"]

EXPOSE 80

ENTRYPOINT ["text-embeddings-router"]
CMD ["--model-id", "Alibaba-NLP/gte-multilingual-base", "--dtype", "float16", "--port", "80"]
