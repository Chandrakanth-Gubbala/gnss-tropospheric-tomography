# GNSS Tropospheric Tomography

Sanitized portfolio implementation of a GNSS tropospheric water-vapor tomography workflow.

This repository shows the core technical pattern behind a tomography project: receiver geometry, slant-path design matrices, voxel-based reconstruction, and comparison of least-squares and ART-style solvers. It uses synthetic weather fields only, so the code can be public without exposing institutional data, station files, raw GNSS products, local paths, or unpublished analysis outputs.

## What this demonstrates

- 3D voxel grid construction for atmospheric retrieval problems
- Approximate ray tracing through a local Cartesian grid
- Forward modeling of slant wet delay / integrated water-vapor observations
- Tikhonov-regularized least squares reconstruction
- Algebraic Reconstruction Technique style iterative inversion
- Vertical-profile extraction for interpretation and model comparison

## What is intentionally excluded

- Raw GNSS observations, station metadata, and institutional datasets
- Hardcoded local research paths
- Private analysis notebooks, unpublished figures, and paper-specific outputs
- Credentials, download scripts, cookies, or API tokens

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python examples/run_demo.py
```

The demo prints reconstruction metrics and writes plots to `outputs/`.

## Repository structure

```text
src/tomography.py       Core grid, ray matrix, and inversion code
examples/run_demo.py    Reproducible synthetic demo
requirements.txt        Minimal scientific Python dependencies
```

## Portfolio note

This is a public-safe reference implementation inspired by research and prototype work. It is designed to show the modeling and numerical methods without releasing confidential data or institution-specific analysis.

