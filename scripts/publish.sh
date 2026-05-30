#!/bin/sh
set -eu

DOCKERHUB_REPO="ggalmeida0/vllm-llama31-8b"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/amd64}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not reachable. Start Docker Desktop or another daemon first." >&2
  exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "docker buildx is required for cross-platform publishing." >&2
  exit 1
fi

REMOTE_IMAGE="${DOCKERHUB_REPO}:${IMAGE_TAG}"

docker buildx build \
  --platform "${PLATFORMS}" \
  -t "${REMOTE_IMAGE}" \
  --push \
  .

echo "Pushed ${REMOTE_IMAGE} for ${PLATFORMS}"
