# SPDX-License-Identifier: MIT

SHELL := /bin/bash
VENV_ACTIVATE := .venv/bin/activate

.PHONY: reproduce test debug clean help

reproduce:
	@echo "Running full reproduction workflow..."
	@bash ./reproduce.sh

test:
	@echo "Running test suite..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then \
		. "$(VENV_ACTIVATE)" && python -m pytest tests.py -v; \
	else \
		python -m pytest tests.py -v; \
	fi

debug:
	@echo "Running debug pipeline..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then \
		. "$(VENV_ACTIVATE)" && python main.py --debug; \
	else \
		python main.py --debug; \
	fi

clean:
	@echo "Cleaning virtualenv, outputs, and caches..."
	@rm -rf .venv checkpoints evaluation_results logs
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:
	@echo "Available targets:"
	@echo "  reproduce  Run the full end-to-end reproduction workflow via reproduce.sh"
	@echo "  test       Run the pytest suite"
	@echo "  debug      Run the pipeline in debug mode with synthetic data"
	@echo "  clean      Remove .venv, outputs, logs, and __pycache__ directories"
	@echo "  help       Print this help message"