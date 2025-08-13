# Conda Environment Setup for dolfinx_iga

## Recommended Setup

### 1. Create Conda Environment
```bash
# Create environment from file
conda env create -f environment.yml

# Or create manually
conda create -n dolfinx_iga python=3.11
conda activate dolfinx_iga
conda install -c conda-forge fenics-dolfinx fenics-basix fenics-ffcx fenics-ufl
```

### 2. Activate and Install Package
```bash
conda activate dolfinx_iga
pip install -e .
```

### 3. Alternative: Minimal Environment
```bash
# Just FEniCS + basics
conda create -n dolfinx_iga_minimal -c conda-forge python=3.11 fenics-dolfinx numpy pytest
conda activate dolfinx_iga_minimal
pip install -e .
```

## Why Conda?

- **FEniCS dependencies**: Only available through conda-forge
- **Complex C++ libraries**: Conda handles PETSc, MPI, HDF5 automatically
- **Reproducible environments**: `environment.yml` ensures consistency
- **Scientific computing**: Optimized NumPy/SciPy builds
- **Cross-platform**: Works on Linux, macOS, Windows

## Environment Management

```bash
# Export exact environment
conda env export > environment-lock.yml

# Update environment
conda env update -f environment.yml

# Remove environment
conda env remove -n dolfinx_iga
```

## Docker Alternative

For ultimate reproducibility:

```dockerfile
FROM condaforge/mambaforge:latest

COPY environment.yml .
RUN mamba env create -f environment.yml

SHELL ["conda", "run", "-n", "dolfinx_iga", "/bin/bash", "-c"]
COPY . /app
WORKDIR /app
RUN pip install -e .
```
