# The SensorDyme cloud backend, as deployed (Fly.io — see fly.toml at the repo root).
#
# Build context is the REPO ROOT, not cloud/, because cloud/app.py imports
# `core.presets` at module scope and that reads presets.yaml from the repo root.
# Building from cloud/ alone produces an image that fails on the first import.
#
# What goes in and why:
#   cloud/         the service itself
#   core/          presets (a hard import), plus taxonomy/capabilities/accel,
#                  which are lazy but feed the capability sheet the AI-configure
#                  prompt is grounded in — an empty sheet makes the model decline
#                  every request as unsupported, so this is not optional weight
#   presets.yaml   the mode registry core.presets reads
#   jupiter/       the AI that /v1/ai/configure and event explanations call.
#                  cloud/app.py imports it lazily and degrades silently on
#                  ImportError, so leaving it out does not fail the build -- it
#                  ships a backend whose "Configure with AI" is a keyword table
#                  and whose explanations are always 503. Pure Python; its only
#                  dependencies (requests, pyyaml) are already installed.
#
# NOT in it: the database and the JWT secret. Those live on the Fly volume at
# /data so they survive deploys, and .dockerignore keeps the developer's copies
# out of the image.
FROM python:3.12-slim

# PYTHONUNBUFFERED so uvicorn's log lines reach `fly logs` as they happen rather
# than when a buffer fills — the difference between watching a deploy and
# guessing at one.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies before source: this layer is cached across deploys that only
# change code, which is nearly all of them.
COPY cloud/requirements.txt cloud/requirements.txt
RUN pip install --no-cache-dir -r cloud/requirements.txt

COPY cloud/ cloud/
COPY core/ core/
COPY jupiter/ jupiter/
COPY presets.yaml presets.yaml
COPY scripts/fly_import_db.py scripts/fly_import_db.py

# Defaults that make the deployed shape correct even if fly.toml is edited or
# the image is run somewhere else. fly.toml states them again, deliberately:
# these are the contract, and having them in one place only would make a
# missing volume mount look like a working server right up until a restart
# silently discarded every account.
#
# SENSORDYME_MDNS=0: mDNS is link-local. Announcing this server to a datacentre
# network reaches nothing and no one; the app finds it by its public hostname.
ENV SENSORDYME_DB=/data/sensordyme.db \
    SENSORDYME_JWT_SECRET_FILE=/data/.jwt_secret \
    SENSORDYME_MDNS=0

EXPOSE 8080

# Runs as root, which is the Fly Machines norm (a single-tenant Firecracker VM)
# and what lets the process write to the root-owned volume at /data without an
# entrypoint that chowns and drops privileges. Worth revisiting if this ever
# runs somewhere multi-tenant.
CMD ["uvicorn", "cloud.app:app", "--host", "0.0.0.0", "--port", "8080"]
