#!/bin/sh
set -eu

DOCKERHUB_REPO=${DOCKERHUB_REPO:-}
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

case "${PLATFORMS}" in
  *,*)
    docker buildx build \
      --platform "${PLATFORMS}" \
      --provenance=false \
      -t "${REMOTE_IMAGE}" \
      --push \
      .
    ;;
  *)
    docker buildx build \
      --platform "${PLATFORMS}" \
      --provenance=false \
      -t "${REMOTE_IMAGE}" \
      --load \
      .
    docker push "${REMOTE_IMAGE}"
    ;;
esac

echo "Pushed ${REMOTE_IMAGE} for ${PLATFORMS}"
