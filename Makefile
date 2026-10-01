.PHONY: test test-integration lint build eval assets clean

test:
	pytest -q

test-integration:
	pytest -m integration

lint:
	ruff check src tests scripts evals

build:
	pip install -q build && python -m build

eval:
	python evals/run_eval.py --out evals/results/routing-eval-mock.json

assets:
	python scripts/generate_assets.py

clean:
	rm -rf dist build *.egg-info .pytest_cache .ruff_cache .coverage
