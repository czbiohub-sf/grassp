init:
	pip install -r requirements.txt

lint:
	flake8 ./ --count --statistics --exit-zero
	python -m pylint ./

pre-commit:
	pre-commit run --all-files

test:
	pytest -v
