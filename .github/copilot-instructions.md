# dolfinx_iga Project Instructions

This is a Python package for Isogeometric Analysis (IGA) that extends DOLFINx with B-spline and NURBS functionality.

## Project Structure

- `dolfinx_iga/` - Main package with B-spline and NURBS implementations
- `tests/` - Comprehensive test suite using pytest
- `examples/` - Usage examples and tutorials
- `docs/` - Documentation files

## Development Workflow

1. **Testing**: Run `pytest tests/` or use the "Run Tests" task
2. **Code Quality**: Pre-commit hooks ensure consistent formatting
3. **Installation**: Package is installed in development mode with `pip install -e .`

## Key Features

- B-spline curves and surfaces with Cox-de Boor algorithms
- NURBS (rational B-splines) for exact geometry representation
- Optional Numba acceleration for performance
- Type hints and comprehensive documentation
- Integration ready for DOLFINx finite element framework

## Coding Standards

- for typing in Python files, for numpy arrays use `npt.NDArray` instead of `np.ndarray` and define `npt` as `import numpy.typing as npt`
- for typing in Python files, use `|` for union types.