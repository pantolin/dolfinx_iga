# Contributing

We welcome contributions to dolfinx_iga! This document outlines how to contribute.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dolfinx_iga.git
   cd dolfinx_iga
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e .[dev]
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dolfinx_iga

# Run specific test file
pytest tests/test_bspline.py
```

## Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking

Run them manually:
```bash
black dolfinx_iga/
flake8 dolfinx_iga/
mypy dolfinx_iga/
```

Or use pre-commit (recommended):
```bash
pre-commit run --all-files
```

## Documentation

Documentation is built with Sphinx. To build locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

## Submitting Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```

2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```

6. Push to your fork:
   ```bash
   git push origin feature-name
   ```

7. Create a Pull Request on GitHub

## Guidelines

- **Tests**: All new functionality must include tests
- **Documentation**: Update docstrings and add examples
- **Type hints**: Use type hints for all new code
- **Backwards compatibility**: Avoid breaking changes when possible
- **Performance**: Consider performance implications of changes

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Python version
- dolfinx_iga version
- Minimal example reproducing the issue
- Expected vs actual behavior
