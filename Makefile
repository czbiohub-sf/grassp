PACKAGE_NAME := grassp

.PHONY: setup-develop
setup-develop:
	pip install -e .'[dev, docs, notebook]'

	pre-commit install

# The R half of the pRoloc bridge lives in a conda environment, because the Bioconductor stack it
# needs is not pip-installable. r-environment.yml declares it; the two steps after the env
# itself are the ones a declarative file cannot express, and the ones easiest to forget.
#
# Needed to run the grasspio test suite, build its vignette, or build the docs at all -- the
# Sphinx build executes docs/source/tutorials/proloc_r_tutorial.Rmd with the `ir` kernel.
R_ENV ?= grassp-r
CONDA ?= conda
# bioconductor-msnbase has no osx-arm64 build (its affyio dependency is missing there), so on
# Apple silicon the environment has to be x86_64 and run under Rosetta. Empty everywhere else.
CONDA_SUBDIR_ARG := $(shell test "`uname -s`" = Darwin && test "`uname -m`" = arm64 \
	&& echo CONDA_SUBDIR=osx-64)

.PHONY: setup-r
setup-r:
	$(CONDA_SUBDIR_ARG) $(CONDA) env create -f r-environment.yml -n $(R_ENV) \
		|| $(CONDA_SUBDIR_ARG) $(CONDA) env update -f r-environment.yml -n $(R_ENV)
	$(CONDA) run -n $(R_ENV) R CMD INSTALL --no-docs r/grasspio
	$(CONDA) run -n $(R_ENV) Rscript -e \
		'IRkernel::installspec(name = "ir", displayname = "R ($(R_ENV))")'
	@echo
	@echo "R environment '$(R_ENV)' is ready and the 'ir' Jupyter kernel is registered."
	@echo "Re-run this after changing r/grasspio so the installed copy stays current."

.PHONY: test-r
test-r:
	$(CONDA) run -n $(R_ENV) Rscript -e \
		'setwd("r/grasspio"); library(testthat); test_local()'

.PHONY: uninstall
uninstall:
	pip uninstall -y $(PACKAGE_NAME)

.PHONY: lint
lint:
	flake8 . --count --statistics --exit-zero
	black --check .
	python -m pylint $(PACKAGE_NAME)

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	pytest -v

# Most docstring examples in this package need real data and are marked `doctest: +SKIP`,
# so `make test` does not run doctests. The "unknown"-sentinel helpers in grassp/io/read.py are
# pure and dependency-light, and their examples document the one pRoloc convention Python handles
# itself, so they are worth executing.
.PHONY: test-doctest
test-doctest:
	pytest --doctest-modules $(PACKAGE_NAME)/io/read.py

.PHONY: docs
docs:
	sphinx-build -b html docs/source docs/build/html

.PHONY: setup-build
setup-build:
	pip install -e .'[build]'

.PHONY: build
build:
	python -m build

.PHONY: publish
publish:
	twine upload dist/*
