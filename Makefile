DEFAULT_GOAL := help

SHELL=/bin/bash
VENV = .venv.make

# Detect the operating system and set the virtualenv bin directory
ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

# Get modified/added Python files from git (excluding *pb2.py files)
GIT_MODIFIED_FILES = $(shell git diff --name-only --diff-filter=d --relative packages | grep '\.py$$' | grep -v 'pb2\.py$$' | grep -v 'pb2\.pyi$$' | grep -v 'pb2_grpc\.py$$')
GIT_STAGED_FILES = $(shell git diff --cached --name-only --diff-filter=d --relative packages | grep '\.py$$' | grep -v 'pb2\.py$$' | grep -v 'pb2\.pyi$$' | grep -v 'pb2_grpc\.py$$')
GIT_UNTRACKED_FILES = $(shell git ls-files --others --exclude-standard | grep '\.py$$' | grep -v 'pb2\.py$$' | grep -v 'pb2\.pyi$$' | grep -v 'pb2_grpc\.py$$')
GIT_FILES = $(sort $(GIT_MODIFIED_FILES) $(GIT_STAGED_FILES) $(GIT_UNTRACKED_FILES))
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: $(VENV)/.venv-timestamp

$(VENV)/.venv-timestamp: uv.lock
	# Create new virtual environment if setup.py has changed
	uv venv --seed --python 3.11 $(VENV)
	cp .devcontainer/project.pth $(VENV)/lib/python3.11/site-packages
	touch $(VENV)/.venv-timestamp

testenv: $(VENV)/.testenv

$(VENV)/.testenv: $(VENV)/bin/activate
	# check uv version and use appropriate parameters
	if . $(VENV_BIN)/activate && uv sync --help | grep -q -- "--active"; then \
		. $(VENV_BIN)/activate && uv sync --active --all-packages --no-build-isolation --group dev --dev --all-extras \
			--link-mode=copy; \
	else \
		. $(VENV_BIN)/activate && uv sync --all-packages --no-build-isolation --group dev --all-extras --dev  \
			--link-mode=copy; \
	fi
	cp .devcontainer/project.pth $(VENV)/lib/python3.11/site-packages
	touch $(VENV)/.testenv
	. $(VENV_BIN)/activate && uv tool install  ruff
	. $(VENV_BIN)/activate && uv tool install pytest
	. $(VENV_BIN)/activate && uv tool install mypy
	echo "Test environment setup complete."


.PHONY: fmt
fmt: testenv ## Format Python code (only modified/added files)
ifneq ($(strip $(GIT_FILES)),)
	# Format code
	$(VENV_BIN)/ruff format $(GIT_FILES)
	# Sort imports
	. $(VENV_BIN)/activate && uv tool run ruff check --exclude *pb2.py --extend-exclude packages/raghub-interfaces/src/raghub_interfaces/protos/pb --select I --fix $(GIT_FILES)
	. $(VENV_BIN)/activate && uv tool run ruff check --exclude *pb2.py --extend-exclude packages/raghub-interfaces/src/raghub_interfaces/protos/pb --fix $(GIT_FILES)
else
	@echo "No modified/added Python files to format."
endif

.PHONY: fmt-check
fmt-check: testenv ## Check Python code formatting and style without making changes (only modified/added files)
ifneq ($(strip $(GIT_FILES)),)
	. $(VENV_BIN)/activate && uv tool run ruff format --check $(GIT_FILES)
	. $(VENV_BIN)/activate && uv tool run ruff check $(GIT_FILES)
else
	@echo "No modified/added Python files to check."
endif

.PHONY: docs
docs:
	python3 docs/make.py

.PHONY: pre-commit
pre-commit: fmt fmt-check mypy docs ## Run formatting and unit tests before committing

.PHONY: mypy
mypy: testenv## Run mypy checks (only modified/added files)
ifneq ($(strip $(GIT_FILES)),)
	. $(VENV_BIN)/activate && uv tool run mypy --config-file .mypy.ini $(GIT_FILES)
else
	@echo "No modified/added Python files to type check."
endif

.PHONY: clean
clean: ## Clean up the environment
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.coverage' -delete

.PHONY: clean-dist
clean-dist: ## Clean up the distribution
	rm -rf dist/ *.egg-info build/

.PHONY: help
help:  ## Display this help screen
	@echo "Available commands:"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort