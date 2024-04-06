# Contributing

TODO

## Quickstart

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg
```

## Build for Pypi

```bash
py -m build --sdist # on windows
python -m build --sdist # on linux

twine upload dist/*
```
