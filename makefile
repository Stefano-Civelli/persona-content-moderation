ENV_NAME = torch
PYTHON = conda run -n $(ENV_NAME) python

.PHONY: help setup install activate run test lint format clean

help:
	@echo "Common commands:"
	@echo "  make setup     - Create conda env and install pip + deps"
	@echo "  make install   - Re-install pip deps (inside env)"
	@echo "  make run       - Run the main script"
	@echo "  make test      - Run tests with pytest"
	@echo "  make lint      - Run flake8 and black check"
	@echo "  make format    - Format code with black"
	@echo "  make clean     - Remove __pycache__, logs, etc."

setup:
	conda create -y -n $(ENV_NAME) python=3.11
	conda run -n $(ENV_NAME) pip install -r requirements.txt

install:
	conda run -n $(ENV_NAME) pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache *.log
