SHELL := /bin/bash
PYTHON ?= $(shell which python)
CONDA_ENV := spatial-adapter
ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

.PHONY: clean install install-dev install-all build-cpp test test-cov test-fast \
        test-gpu lint format jupyter conda-env help

# Installation
install:  ## Install core library (editable)
	@$(ACTIVATE) && pip install -e .

install-dev:  ## Install with dev tools (pytest, black, etc.)
	@$(ACTIVATE) && pip install -e ".[dev]"

install-all:  ## Install everything (vision + forecasting + interactive + dev)
	@$(ACTIVATE) && pip install -e ".[all]"

# C++ extensions
build-cpp:  ## Build pybind11 C++ extensions (spatial_utils.so)
	@echo "Building C++ extensions..."
	@$(ACTIVATE) && \
	cd spatial_adapter/cpp_extensions && \
	rm -rf build spatial_utils.so && \
	mkdir -p build && cd build && \
	cmake \
	  -DCMAKE_PREFIX_PATH=$$CONDA_PREFIX \
	  -DCMAKE_INCLUDE_PATH=$$CONDA_PREFIX/include \
	  -DCMAKE_LIBRARY_PATH=$$CONDA_PREFIX/lib \
	  -DCMAKE_BUILD_TYPE=Release .. && \
	make -j$$(nproc) && \
	cd ../.. && \
	python -c "from spatial_adapter.cpp_extensions import spatial_utils; print('C++ extensions OK')"

# Testing
test:  ## Run all tests with coverage
	@$(ACTIVATE) && python -m pytest

test-cov:  ## Run tests with HTML coverage report
	@$(ACTIVATE) && python -m pytest --cov-report=html

test-fast:  ## Run non-slow tests only
	@$(ACTIVATE) && python -m pytest -m "not slow"

test-gpu:  ## Run GPU-only tests
	@$(ACTIVATE) && python -m pytest -m "gpu"

# Code quality
lint:  ## Run flake8 + mypy
	@$(ACTIVATE) && flake8 spatial_adapter/ tests/ && mypy spatial_adapter/

format:  ## Auto-format with black + isort
	@$(ACTIVATE) && black spatial_adapter/ tests/ && isort spatial_adapter/ tests/

# Environment
conda-env:  ## Create conda env + build C++ extensions
	@bash envs/conda/build_conda_env.sh
	@make build-cpp

jupyter:  ## Start Jupyter Lab
	@$(SHELL) envs/jupyter/start_jupyter_lab.sh --port 8501

# Cleanup
clean:  ## Remove build artifacts, caches, coverage
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ *.egg-info .eggs/ .pytest_cache/ htmlcov/ .coverage

# Help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
