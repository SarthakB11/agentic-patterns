.PHONY: help install test lint types cover demos check all

help:
	@echo "install  install the package with dev dependencies (editable)"
	@echo "test     run the test suite (offline, no API key)"
	@echo "lint     run ruff"
	@echo "types    run pyright"
	@echo "cover    run the suite under coverage and print the report"
	@echo "demos    run all twelve pattern demos offline"
	@echo "check    lint + types + test (what CI runs)"

install:
	python3 -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

types:
	pyright

cover:
	coverage run -m pytest -q
	coverage report --include="agentic_patterns/*,patterns/*"

demos:
	@for p in react planning reflection tool_use memory rag multi_agent evaluation mcp guardrails human_in_the_loop routing; do \
		echo "== $$p =="; python3 -m patterns.$$p.main > /dev/null && echo "ok" || exit 1; \
	done

check: lint types test

all: check
