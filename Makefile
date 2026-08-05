.PHONY: install test lint run

install:
	python -m pip install -r requirements.txt

test:
	python -m unittest discover -s tests -v

lint:
	ruff format --check app.py signlearn tests
	ruff check app.py signlearn tests

run:
	python app.py
