.PHONY: clean lint test docs build install cuda

all: test

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
	find . -name '.coverage' -delete
	find . -name '.pytest_cache' -delete
	rm -rf htmlcov/ .coverage

lint:
	flake8 attograd tests
	black --check attograd tests examples

format:
	black attograd tests examples

test:
	pytest --cov=attograd tests/

# Compile CUDA shared library (requires nvcc)
# Linux/WSL: -fPIC is required. Windows: -fPIC is silently ignored by MSVC, safe to keep.
cuda:
	nvcc -shared -Xcompiler -fPIC \
		-o attograd/cuda/shared_lib/vector_ops.so \
		attograd/cuda/vector_ops.cu

docs:
	cd docs && make html

build: clean
	python setup.py sdist bdist_wheel

install:
	pip install -e ".[dev]"
