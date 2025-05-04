DEFAULT_GOAL := help

SHELL=/bin/bash
VENV = .venv.make

# Detect the operating system and set the virtualenv bin directory
ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

# Get modified/added Python files from git
GIT_MODIFIED_FILES = $(shell git diff --name-only --diff-filter=d --relative packages | grep '\.py$$')
GIT_STAGED_FILES = $(shell git diff --cached --name-only --diff-filter=d --relative packages | grep '\.py$$')
GIT_FILES = $(sort $(GIT_MODIFIED_FILES) $(GIT_STAGED_FILES))

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: $(VENV)/.venv-timestamp

$(VENV)/.venv-timestamp: uv.lock
	# Create new virtual environment if setup.py has changed
	uv venv --seed --python 3.11 $(VENV)
	touch $(VENV)/.venv-timestamp

testenv: $(VENV)/.testenv

$(VENV)/.testenv: $(VENV)/bin/activate
	# check uv version and use appropriate parameters
	if . $(VENV_BIN)/activate && uv sync --help | grep -q -- "--active"; then \
		. $(VENV_BIN)/activate && uv sync --active --all-packages --reinstall --no-build-isolation \
			--link-mode=copy; \
	else \
		. $(VENV_BIN)/activate && uv sync --all-packages --reinstall --no-build-isolation  \
			--link-mode=copy; \
	fi
	cp .devcontainer/project.pth $(VENV)/lib/python3.11/site-packages
	touch $(VENV)/.testenv
	. $(VENV_BIN)/activate && uv tool install  ruff
	. $(VENV_BIN)/activate && uv tool install pytest
	. $(VENV_BIN)/activate && uv tool install setuptools
	. $(VENV_BIN)/activate && uv tool install mypy
	echo "Test environment setup complete."


.PHONY: fmt
fmt: testenv ## Format Python code (only modified/added files)
ifneq ($(strip $(GIT_FILES)),)
	# Format code
	$(VENV_BIN)/ruff format $(GIT_FILES)
	# Sort imports
	$(VENV_BIN)/ruff check --select I --fix $(GIT_FILES)
	$(VENV_BIN)/ruff check --fix $(GIT_FILES)
else
	@echo "No modified/added Python files to format."
endif

.PHONY: fmt-check
fmt-check: testenv ## Check Python code formatting and style without making changes (only modified/added files)
ifneq ($(strip $(GIT_FILES)),)
	$(VENV_BIN)/ruff format --check $(GIT_FILES)
	$(VENV_BIN)/ruff check --select I $(GIT_FILES)
	$(VENV_BIN)/ruff check $(GIT_FILES)
else
	@echo "No modified/added Python files to check."
endif

.PHONY: pre-commit
pre-commit: fmt-check mypy ## Run formatting and unit tests before committing

.PHONY: mypy
mypy: testenv## Run mypy checks (only modified/added files)
ifneq ($(strip $(GIT_FILES)),)
	uv tool run mypy --config-file .mypy.ini $(GIT_FILES)
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