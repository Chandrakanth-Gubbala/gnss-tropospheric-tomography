# Method Notes

This repository separates two parts of the tomography workflow:

1. Matrix construction: each slant path is intersected with every voxel to produce `A`.
2. Retrieval: synthetic observations are inverted with regularized least squares or an ART-style solver.

The basic tomography equation is:

```text
y = A x
```

where:

- `y` is the vector of slant integrated observations.
- `A` contains the path length of each ray inside each voxel.
- `x` is the unknown voxel-level water-vapor field.

## Coordinate Assumptions

The public example uses a local Cartesian domain:

- `x`: east-west distance in km
- `y`: north-south distance in km
- `z`: height in km

The sample ray file stores receiver position, azimuth, elevation, and finite slant range. Azimuth is interpreted clockwise from north.

## Intersection Calculation

`src/intersection_matrix.py` uses the standard slab method for finite ray and axis-aligned box intersection. For each ray-voxel pair, it computes the entering and exiting distances along the ray. The difference between those distances is the path length assigned to that matrix cell.

## Validation Output

The matrix example writes a validation table. For each ray, it compares:

- the sum of all voxel-level intersection lengths
- the intersection length through the entire model domain treated as one large box

Small differences can occur from floating-point precision, but the synthetic example should remain near zero.

## Public-Safe Scope

This is not a release of raw GNSS data or institution-specific research outputs. The input file is synthetic, the coordinates are generic, and the code is structured to demonstrate the method without exposing private station metadata, unpublished analysis, or local data paths.

