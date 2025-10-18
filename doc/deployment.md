# Greek Forge Deployment Guide

This guide covers deploying the Greek Forge API using Docker containers.

## Architecture Overview

The deployment uses a **stateless inference-only architecture**:
- Pre-trained models are bundled into the Docker image
- No database connection required at runtime
- API serves predictions using loaded models

## Prerequisites

- Docker installed and running
- Trained models (CALL and PUT) in `./models/` directory
- Models must be trained **before** building the Docker image

## Quick Start

### Option 1: Full Deployment (Recommended)

Build models, create Docker image, and run container in one command:

```bash
make docker-deploy
```

This will:
1. Train CALL and PUT models (if not already trained)
2. Build the Docker image with models bundled
3. Stop any existing container
4. Start a new container

### Option 2: Step-by-Step Deployment

```bash
# 1. Train models (if not already done)
make build-models

# 2. Build Docker image
make docker-build

# 3. Run container
make docker-run
```

## Accessing the API

Once deployed, the API is available at:
- **API Base**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Docker Commands

### Build Docker Image

```bash
make docker-build
```

This creates an image named `greek-forge-api:latest`.

**Requirements:**
- Models must exist in `./models/calls/` and `./models/puts/`
- If models are missing, the build will fail with an error message

### Run Container

```bash
make docker-run
```

Starts a container in detached mode (-d) on port 8000.

### View Logs

```bash
docker logs -f greek-forge-api
```

Or use Docker Desktop's log viewer.

### Stop Container

```bash
make docker-stop
```

Stops and removes the running container.

### Clean Up

Remove container and image:

```bash
make docker-clean
```

## Dockerfile Structure

### Stage 1: Builder
- Installs dependencies
- Copies application code
- Validates that trained models exist
- Creates a complete application environment

### Stage 2: Runtime
- Minimal runtime environment
- Copies only necessary files from builder
- Runs as non-root user (`apiuser`) for security
- Exposes port 8000
- Includes health check

## Configuration

### Port Mapping

Default port is 8000. To use a different port:

```bash
# Edit Makefile and change DOCKER_PORT
DOCKER_PORT = 8080

# Or run manually:
docker run -d --name greek-forge-api -p 8080:8000 greek-forge-api:latest
```

### Environment Variables

The container doesn't currently use environment variables, but you can add them:

```bash
docker run -d \
  --name greek-forge-api \
  -p 8000:8000 \
  -e LOG_LEVEL=debug \
  greek-forge-api:latest
```

## Production Considerations

### Security

Current implementation:
- ✅ Runs as non-root user
- ✅ No database credentials in container
- ✅ Minimal attack surface (stateless)
- ❌ No authentication/authorization
- ❌ No rate limiting
- ❌ HTTP only (no TLS)

### Health Checks

The Docker image includes a built-in health check:
- Checks `/health` endpoint every 30 seconds
- 3-second timeout
- 5-second startup grace period
- Container marked unhealthy after 3 consecutive failures

### Model Updates

To deploy updated models:

```bash
# 1. Train new models locally
make build-models

# 2. Rebuild Docker image
make docker-build

# 3. Stop old container and start new one
make docker-stop
make docker-run

# Or use docker-deploy to do all steps
make docker-deploy
```

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker logs greek-forge-api
```

Common issues:
- Missing models: Ensure `models/calls` and `models/puts` directories exist
- Port conflict: Another service using port 8000
- Resource limits: Insufficient memory/CPU

### Health Check Failing

Test manually:
```bash
curl http://localhost:8000/health
```

If endpoint is unreachable:
- Check container is running: `docker ps`
- Check port mapping: `docker port greek-forge-api`
- Check application logs: `docker logs greek-forge-api`

### Performance Issues

Monitor resource usage:
```bash
docker stats greek-forge-api
```

Increase resources if needed:
```bash
docker run -d \
  --name greek-forge-api \
  -p 8000:8000 \
  --memory="1g" \
  --cpus="2" \
  greek-forge-api:latest
```

## Alternative: Local Development

For development, run the API locally without Docker:

```bash
make serve-dev
```

This provides:
- Auto-reload on code changes
- Direct access to source code
- Easier debugging
- Faster iteration

## Support

For issues or questions:
- Check application logs: `docker logs greek-forge-api`
- Review API documentation: http://localhost:8000/docs
- Test endpoints manually using the Swagger UI