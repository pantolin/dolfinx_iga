# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-08-13

### Added
- Initial project structure and setup
- B-spline curve implementation with:
  - Cox-de Boor recursion for basis functions
  - Curve evaluation and derivatives
  - Support for custom knot vectors
- NURBS curve implementation with:
  - Rational basis functions
  - Exact representation of conic sections
- B-spline and NURBS surface implementations
- Comprehensive test suite with pytest
- Example scripts demonstrating basic usage
- Development environment with:
  - Black code formatting
  - Flake8 linting
  - MyPy type checking
  - Pre-commit hooks
- CI/CD ready configuration
- Documentation structure

### Features
- Pure Python implementation with optional Numba acceleration
- Compatible with DOLFINx and FEniCS ecosystem
- Pip installable package
- Type hints throughout
- Comprehensive error handling
- 100% test coverage for core functionality
