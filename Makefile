PYTHON ?= .venv/bin/python
ENTRYPOINT ?= benchpress.py
APP_NAME ?= benchpress

.PHONY: run build clean

run:
	$(PYTHON) $(ENTRYPOINT)

build:
	$(PYTHON) -m PyInstaller \
		--name $(APP_NAME) \
		--onefile \
		--collect-submodules tiktoken_ext \
		--collect-data tiktoken \
		$(ENTRYPOINT)

clean:
	rm -rf build dist *.spec __pycache__
