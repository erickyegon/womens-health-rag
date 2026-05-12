.PHONY: setup install lint typecheck test test-unit test-integration \
        ingest eval up down shell deploy clean help

# ── Setup ─────────────────────────────────────────────────────────────────────
setup: ## Install deps + pre-commit hooks
	uv sync --all-extras
	uv run pre-commit install
	@echo "\n✅  Environment ready. Copy .env.example to .env and fill in your keys.\n"

install: ## Install package in editable mode only
	uv pip install -e ".[dev]"

# ── Code quality ──────────────────────────────────────────────────────────────
lint: ## Run ruff linter + formatter check
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/

format: ## Auto-fix lint issues
	uv run ruff check --fix src/ tests/ scripts/
	uv run ruff format src/ tests/ scripts/

typecheck: ## Run mypy type checker
	uv run mypy src/rag/

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run full test suite with coverage
	uv run pytest tests/ -v --cov=src/rag --cov-report=term-missing --cov-report=html

test-unit: ## Run only fast unit tests (no network, no DB)
	uv run pytest tests/unit/ -v

test-integration: ## Run integration tests (needs running postgres)
	uv run pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests (needs full docker stack)
	uv run pytest tests/e2e/ -v

# ── Pipeline ──────────────────────────────────────────────────────────────────
ingest: ## Run full PDF ingestion pipeline (basic)
	uv run python scripts/ingest.py --metadata data/metadata.json

ingest-multimodal: ## Run multimodal ingestion (Docling + GPT-4o + prose) — Episode 2B
	uv run python scripts/ingest.py --multimodal --metadata data/metadata.json

ingest-docling: ## Run Docling-only ingestion (no GPT-4o vision costs)
	uv run python scripts/ingest.py --multimodal --no-vision --metadata data/metadata.json

eval: ## Run RAGAS evaluation suite
	uv run python scripts/eval.py

seed: ## Seed DB with sample data for demos
	uv run python scripts/seed_db.py

# ── Docker ────────────────────────────────────────────────────────────────────
up: ## Start full local stack (API + UI + postgres)
	docker compose up --build

up-bg: ## Start stack in background
	docker compose up --build -d

down: ## Stop stack and remove volumes
	docker compose down -v

shell: ## Open bash inside the running API container
	docker compose exec api bash

logs: ## Tail logs from all containers
	docker compose logs -f

# ── Deployment ────────────────────────────────────────────────────────────────
deploy: ## Deploy to Fly.io (production)
	fly deploy --config fly.toml

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove all build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"       -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."

# ── Help ──────────────────────────────────────────────────────────────────────
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
