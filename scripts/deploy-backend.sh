#!/usr/bin/env bash
# =============================================================================
# scripts/deploy-backend.sh — manual backend redeploy to Azure Container Apps.
#
# ACR Tasks are disabled on the "Azure for Students" subscription, so the image
# is built locally (amd64) and pushed, then the Container App is rolled to it.
# Reads resource names from ../.azure-deploy.env.
#
#   ./scripts/deploy-backend.sh            # tag = git short sha
#   ./scripts/deploy-backend.sh v2         # explicit tag
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .azure-deploy.env ] || { echo "missing .azure-deploy.env"; exit 1; }
# shellcheck disable=SC1091
source .azure-deploy.env

TAG="${1:-$(git rev-parse --short HEAD)}"
IMAGE="${ACR}.azurecr.io/graphrag-api"

echo "==> az acr login ($ACR)"
az acr login -n "$ACR"

echo "==> docker build --platform linux/amd64 (target: api)  ->  $IMAGE:$TAG"
docker build --platform linux/amd64 -f backend/Dockerfile --target api \
  -t "$IMAGE:$TAG" -t "$IMAGE:latest" .

echo "==> docker push"
docker push "$IMAGE:$TAG"
docker push "$IMAGE:latest"

echo "==> az containerapp update ($APP)"
az containerapp update -n "$APP" -g "$RG" --image "$IMAGE:$TAG" \
  --query "properties.provisioningState" -o tsv

FQDN=$(az containerapp show -n "$APP" -g "$RG" \
  --query properties.configuration.ingress.fqdn -o tsv)
echo "==> https://$FQDN/api/health"
curl -fsS "https://$FQDN/api/health" && echo
