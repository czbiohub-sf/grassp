PACKAGE_NAME := grassp

.PHONY: setup-develop
setup-develop:
	pip install -e .'[dev, docs, notebook]'

	pre-commit install

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
# so `make test` does not run doctests. grassp/io/_msnset.py is pure and dependency-light,
# and its examples document the pRoloc exchange contract, so they are worth executing.
.PHONY: test-doctest
test-doctest:
	pytest --doctest-modules $(PACKAGE_NAME)/io/_msnset.py

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
