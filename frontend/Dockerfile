# syntax=docker/dockerfile:1
# =============================================================================
# frontend/Dockerfile — build the Vite/React app, serve it with nginx.
#   docker build -t graphrag-frontend --build-arg VITE_API_BASE=/api ./frontend
# =============================================================================

FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Baked into the bundle at build time. "/api" keeps calls same-origin so the
# nginx stage can reverse-proxy them to the backend (no CORS needed).
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:1.27-alpine AS runtime
# Where nginx proxies /api/* — overridden per environment (compose sets the
# service name; a standalone deploy sets the backend's public URL).
ENV BACKEND_ORIGIN=http://backend:8000
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
# nginx:alpine entrypoint runs envsubst on /etc/nginx/templates/*.template
# then launches nginx, so no CMD override is needed.
