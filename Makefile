PACKAGE_NAME := some_package

.PHONY: setup-develop
setup-develop:
	pip install -e .'[dev]'
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

.PHONY: setup-build
setup-build:
	pip install -e .'[build]'

.PHONY: build
build:
	python -m build

.PHONY: publish
publish:
	twine upload dist/*
