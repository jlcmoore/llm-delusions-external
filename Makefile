# .Makefile

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

ENV_NAME ?= env-delusions
VENV_DIR ?= $(ENV_NAME)
LFS_MAX_FILE_GIB ?= 1
LFS_MAX_TOTAL_GIB ?= 10
REGULAR_MAX_FILE_MIB ?= 50

PYTHON_VENV := $(VENV_DIR)/bin/python
PYTHON_VENV_ABS := $(abspath $(PYTHON_VENV))
PIP_VENV := $(VENV_DIR)/bin/pip
CONDA_RUN := conda run -n $(ENV_NAME)
NPM_INSTALL_CMD := if [ -f package-lock.json ]; then npm ci; else npm install; fi

.PHONY: init init-js init-venv init-conda init-r setup-venv setup-conda clean pyfmt pylint jslint mdfmt clean-transcripts viewer check-sizes Rfmt Rlint

init: init-venv init-js

init-js:
	@$(NPM_INSTALL_CMD)

init-venv:
	python3 -m venv $(VENV_DIR)
	$(PIP_VENV) install -r requirements.txt
	$(PIP_VENV) install --editable .
	@echo venv > .env-type
	$(MAKE) setup-venv

init-conda:
	conda env create --file environment.yml
	@echo conda > .env-type
	$(CONDA_RUN) pip install --editable .
	$(MAKE) setup-conda

setup-venv:
	$(PYTHON_VENV) -m spacy download en_core_web_lg

setup-conda:
	$(CONDA_RUN) python -m spacy download en_core_web_lg

init-r:
	@echo "Bootstrapping project-local R environment with renv..."
	Rscript -e "if (!requireNamespace('renv', quietly = TRUE)) install.packages('renv', repos = 'https://cloud.r-project.org')" && \
	Rscript -e "renv::init(bare = TRUE)" && \
	Rscript -e "renv::install(c('arrow', 'dplyr', 'lme4', 'tidyr', 'styler', 'lintr'))"

clean:
	rm -rf $(VENV_DIR) .env-type

# Code formatting: isort (imports) + black (code style)
pyfmt:
	$(PYTHON_VENV) -m isort --profile black scripts src analysis
	$(PYTHON_VENV) -m black scripts src analysis

# Static analysis: pylint
pylint:
	$(PYTHON_VENV) -m graylint --lint "$(PYTHON_VENV_ABS) -m pylint" scripts src analysis

jslint:
	# Ensure root devDependencies (eslint, plugins) are installed
	@if [ ! -x ./node_modules/.bin/eslint ] || [ ! -x ./node_modules/.bin/prettier ]; then \
		$(NPM_INSTALL_CMD); \
	fi
	./node_modules/.bin/prettier --cache --write \
		'analysis/**/*.js' \
		'analysis/**/*.css' \
		'analysis/**/*.html' \
		'analysis/**/*.json' \
		'!analysis/data/**/*' \
		'!analysis/figures/**/*'
	./node_modules/.bin/eslint --cache --fix analysis

mdfmt:
	# Ensure Prettier is available
	@if [ ! -x ./node_modules/.bin/prettier ]; then \
		$(NPM_INSTALL_CMD); \
	fi
	./node_modules/.bin/prettier --write '**/*.md'

clean-transcripts:
	@if [ -d transcripts_de_ided ]; then \
		find transcripts_de_ided -type f \( \
			-name '*.html' \
		\) -print -delete; \
	else \
		echo "transcripts_de_ided does not exist"; \
	fi

Rfmt:
	@echo "Styling R scripts under analysis/ with styler..."
	Rscript -e "if (!requireNamespace('styler', quietly = TRUE)) install.packages('styler', repos = 'https://cloud.r-project.org'); styler::style_dir('analysis', filetype = 'R')"

Rlint:
	@echo "Linting R scripts under analysis/ with lintr..."
	Rscript -e "if (!requireNamespace('lintr', quietly = TRUE)) install.packages('lintr', repos = 'https://cloud.r-project.org'); res <- lintr::lint_dir('analysis'); print(res); if (length(res) > 0L) quit(status = 1) else quit(status = 0)"

viewer:
	# Ensure JS dependencies for the viewer are installed
	@if [ ! -d ./node_modules ]; then \
		echo "Installing JS dependencies..."; \
		$(NPM_INSTALL_CMD); \
	fi
	@echo "Starting local HTTP server for classification viewer on http://localhost:8000 ..."
	@echo "Press Ctrl+C to stop."
	@bash -c ' \
    if [ -f .env-type ] && grep -q "^conda$$" .env-type; then \
      RUN_CMD="$(CONDA_RUN) python analysis/viewer/no_cache_http_server.py"; \
	elif [ -x "$(PYTHON_VENV)" ]; then \
	  RUN_CMD="$(PYTHON_VENV) analysis/viewer/no_cache_http_server.py"; \
	else \
	  RUN_CMD="python3 analysis/viewer/no_cache_http_server.py"; \
	fi; \
	$$RUN_CMD --directory . --port 8000 & \
	SERVER_PID=$$!; \
	trap "kill $$SERVER_PID" INT TERM EXIT; \
	sleep 1; \
	open http://localhost:8000/analysis/viewer/classification_viewer.html || true; \
	wait $$SERVER_PID'

check-sizes:
	@echo "Running repository size checks (Git LFS and regular files)..."
	@LFS_MAX_FILE_GIB=$(LFS_MAX_FILE_GIB) LFS_MAX_TOTAL_GIB=$(LFS_MAX_TOTAL_GIB) REGULAR_MAX_FILE_MIB=$(REGULAR_MAX_FILE_MIB) bash scripts/check_repo_sizes.sh
