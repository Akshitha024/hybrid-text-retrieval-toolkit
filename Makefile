.PHONY: help install lint typecheck test data index search bench plots clean

DATASET ?= scifact
TOPK ?= 10

help:
	@echo "make install                          - install deps via uv"
	@echo "make lint / typecheck / test          - quality gates"
	@echo "make data DATASET=scifact             - download a BEIR dataset"
	@echo "make index DATASET=scifact            - build BM25 + dense indexes"
	@echo "make bench DATASET=scifact TOPK=10    - run all retrievers, write results/"
	@echo "make plots                            - regenerate the figure"

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -m "not slow and not needs_net"

data:
	uv run hr data fetch --dataset $(DATASET)

index:
	uv run hr index build --dataset $(DATASET)

bench:
	uv run hr bench run --dataset $(DATASET) --topk $(TOPK)

plots:
	uv run hr plots

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +


.PHONY: pdf test-artifacts
pdf:
	cd docs/_report && pandoc research_report.md -o ../research_report.pdf --pdf-engine=xelatex || echo "pandoc + xelatex required; see https://pandoc.org/installing.html"

test-artifacts:
	uv run python ../../_meta/retrofit.py "$(notdir $(CURDIR))" "$(notdir $(CURDIR))"
