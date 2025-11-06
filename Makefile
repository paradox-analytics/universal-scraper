# Universal Scraper Makefile

.PHONY: help install test lint format clean run-example deploy-apify

help:
	@echo "Universal Scraper - Available Commands"
	@echo ""
	@echo "  install          Install dependencies"
	@echo "  test             Run tests"
	@echo "  lint             Run linters"
	@echo "  format           Format code"
	@echo "  clean            Clean cache and build files"
	@echo "  run-example      Run basic example"
	@echo "  deploy-apify     Deploy to Apify platform"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=universal_scraper --cov-report=html

lint:
	flake8 universal_scraper/ --max-line-length=120
	mypy universal_scraper/ --ignore-missing-imports

format:
	black universal_scraper/ examples/
	isort universal_scraper/ examples/

clean:
	rm -rf cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-example:
	python examples/basic_usage.py

deploy-apify:
	chmod +x deploy_to_apify.sh
	./deploy_to_apify.sh

# Development helpers
dev-install:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy isort
	pip install -e .

watch-tests:
	pytest-watch tests/ -v

# Docker commands
docker-build:
	docker build -t universal-scraper:latest .

docker-run:
	docker run -it --rm \
		-e OPENAI_API_KEY=${OPENAI_API_KEY} \
		-v $(PWD)/cache:/app/cache \
		universal-scraper:latest

# Cache management
cache-stats:
	@python -c "from universal_scraper.core.code_cache import CodeCache; cache = CodeCache(); print(cache.get_stats())"

cache-clear:
	rm -rf cache/
	@echo "Cache cleared"

# Version bump
bump-patch:
	@echo "Bumping patch version..."
	# Add version bump logic here

bump-minor:
	@echo "Bumping minor version..."
	# Add version bump logic here

bump-major:
	@echo "Bumping major version..."
	# Add version bump logic here

