.PHONY: lint format test test-integration index serve eval docker-build docker-run

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -m "not integration" -v --tb=short

test-integration:
	pytest -v --tb=short

test-all:
	pytest -v --tb=short

index:
	python -m rag_ci_cd.cli index

serve:
	python -m rag_ci_cd.cli serve

eval:
	python -m rag_ci_cd.cli eval

docker-build:
	docker build -t rag-ci-cd .

docker-run:
	docker run -p 6565:6565 rag-ci-cd

install:
	pip install -e ".[dev]"
