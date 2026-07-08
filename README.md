# GNSS Tropospheric Tomography

Sanitized portfolio implementation of a GNSS tropospheric water-vapor tomography workflow.

This repository shows the core technical pattern behind a tomography project: receiver geometry, voxel-ray intersection matrices, synthetic slant-path observations, voxel-based reconstruction, and comparison of least-squares and ART-style solvers. It uses synthetic weather fields only, so the code can be public without exposing institutional data, station files, raw GNSS products, local paths, or unpublished analysis outputs.

The central tomography equation is:

```text
y = A x
```

`y` is the vector of slant integrated observations, `A` is the ray-by-voxel path length matrix, and `x` is the unknown voxel-level water-vapor field.

## What this demonstrates

- 3D voxel grid construction for atmospheric retrieval problems
- Voxel-ray intersection matrix construction
- Validation of row-wise path lengths against the whole model domain
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
python examples/build_matrix_from_sample.py
```

The first demo prints reconstruction metrics. The matrix demo reads a synthetic ray file, writes `A_matrix.csv`, `voxel_lookup.csv`, and `validation_lengths.csv` to `outputs/matrix_demo/`, and reports the maximum validation error.

## Sample Input Format

`examples/sample_input_data.txt` is whitespace-delimited:

```text
receiver_id x_km y_km z_km azimuth_deg elevation_deg slant_range_km
```

All sample coordinates are generic local Cartesian coordinates in kilometers. Azimuth is clockwise from north, elevation is above the local horizon, and slant range limits the finite ray segment.

## Repository structure

```text
src/tomography.py              Synthetic field, reconstruction, and solver code
src/intersection_matrix.py     Voxel-ray intersection matrix builder
examples/run_demo.py           Reproducible synthetic retrieval demo
examples/build_matrix_from_sample.py  Matrix-building demo with validation output
examples/make_sample_data.py   Synthetic ray table generator
examples/sample_input_data.txt Public synthetic ray observations
docs/method.md                 Method notes and validation logic
requirements.txt               Minimal scientific Python dependencies
```

## Portfolio note

This is a public-safe reference implementation inspired by research and prototype work. It is designed to show the modeling and numerical methods without releasing confidential data or institution-specific analysis.
