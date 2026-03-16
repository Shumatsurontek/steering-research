# Steering Arena — task runner

# Default: show available commands
default:
    @just --list

# Build the Docker image
build:
    docker build -t steering-arena .

# Run the Docker container (mount results/ for .pt files and SAE weights)
run:
    docker run --rm -it \
        -p 8000:8000 \
        -v "$(pwd)/results:/app/results" \
        --name steering-arena \
        steering-arena

# Build and run in one command
up: build run

# Stop the running container
stop:
    docker stop steering-arena 2>/dev/null || true

# Run in dev mode (no Docker — two terminals)
dev:
    @echo "Starting backend and frontend..."
    @just dev-backend &
    @just dev-frontend

# Backend only (dev mode)
dev-backend:
    cd {{justfile_directory()}} && .venv/bin/uvicorn web.api.main:app --reload --port 8000 --log-level warning --no-access-log

# Frontend only (dev mode)
dev-frontend:
    cd {{justfile_directory()}}/web/frontend && npm run dev

# Install all dependencies
install:
    cd {{justfile_directory()}} && pip install -r requirements.txt fastapi uvicorn[standard] scikit-learn sae-lens transformer-lens
    cd {{justfile_directory()}}/web/frontend && npm install

# Build frontend for production
build-frontend:
    cd {{justfile_directory()}}/web/frontend && npm run build

# Type-check frontend
check:
    cd {{justfile_directory()}}/web/frontend && npx tsc --noEmit

# Test API health
health:
    curl -s http://localhost:8000/api/health | python3 -m json.tool
