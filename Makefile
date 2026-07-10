PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install test lint run generate-docs

install:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest ruff

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

run:
	$(PYTHON) -m streamlit run app.py

generate-docs:
	$(PYTHON) document_generator.py
