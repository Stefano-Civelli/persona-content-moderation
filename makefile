ENV_NAME = torch
PYTHON = conda run -n $(ENV_NAME) python

.PHONY: help setup install activate run test lint format clean run_script

help:
	@echo "Common commands:"
	@echo "  make setup     - Create conda env and install pip + deps"
	@echo "  make install   - Re-install pip deps (inside env)"
	@echo "  make run       - Run the main script"
	@echo "  make run_script script=<name> - Run a given script"
	@echo "  make test      - Run tests with pytest"
	@echo "  make lint      - Run flake8 and black check"
	@echo "  make format    - Format code with black"
	@echo "  make clean     - Remove __pycache__, logs, etc."

setup:
	conda create -y -n $(ENV_NAME) python=3.11
	conda run -n $(ENV_NAME) pip install -r requirements.txt

install:
	conda run -n $(ENV_NAME) pip install -r requirements.txt

run:
	python -m src.scripts.8_content_classification_text

run_script:
	python -m src.scripts.$(script)

a100:
	salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=48G --job-name=TinyGPU --time=01:30:00 --partition=gpu_cuda --qos=gpu --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1 --account=a_demartini srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l


clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache *.log
