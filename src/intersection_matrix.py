"""Voxel-ray intersection matrix builder for GNSS tomography examples.

This module is a public-safe, from-scratch implementation of the core matrix
construction step used in tomography problems. It builds the matrix A in y = Ax,
where each row is a slant ray and each column is a voxel path length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tomography import VoxelGrid


@dataclass(frozen=True)
class RayObservation:
    ray_id: int
    receiver_id: str
    start_km: np.ndarray
    azimuth_deg: float
    elevation_deg: float
    slant_range_km: float

    @property
    def direction(self) -> np.ndarray:
        return direction_from_az_el(self.azimuth_deg, self.elevation_deg)


def direction_from_az_el(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Convert azimuth/elevation to a unit vector in local ENU-like axes.

    x is east, y is north, and z is up. Azimuth is clockwise from north.
    """

    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    vector = np.array(
        [
            np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
            np.sin(el),
        ],
        dtype=float,
    )
    return vector / np.linalg.norm(vector)


def load_ray_table(path: str | Path) -> list[RayObservation]:
    """Load synthetic ray observations from a whitespace-delimited text file."""

    columns = [
        "receiver_id",
        "x_km",
        "y_km",
        "z_km",
        "azimuth_deg",
        "elevation_deg",
        "slant_range_km",
    ]
    table = pd.read_csv(path, sep=r"\s+", comment="#", names=columns)
    rays: list[RayObservation] = []

    for ray_id, row in enumerate(table.itertuples(index=False)):
        rays.append(
            RayObservation(
                ray_id=ray_id,
                receiver_id=str(row.receiver_id),
                start_km=np.array([row.x_km, row.y_km, row.z_km], dtype=float),
                azimuth_deg=float(row.azimuth_deg),
                elevation_deg=float(row.elevation_deg),
                slant_range_km=float(row.slant_range_km),
            )
        )

    return rays


def voxel_index_table(grid: VoxelGrid) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    nx, ny, nz = grid.shape

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                voxel_id = grid.flatten_index(np.array([ix]), np.array([iy]), np.array([iz]))[0]
                rows.append(
                    {
                        "voxel_id": int(voxel_id),
                        "voxel_label": f"v_{ix}_{iy}_{iz}",
                        "ix": ix,
                        "iy": iy,
                        "iz": iz,
                        "x_min_km": float(grid.x_edges[ix]),
                        "x_max_km": float(grid.x_edges[ix + 1]),
                        "y_min_km": float(grid.y_edges[iy]),
                        "y_max_km": float(grid.y_edges[iy + 1]),
                        "z_min_km": float(grid.z_edges[iz]),
                        "z_max_km": float(grid.z_edges[iz + 1]),
                    }
                )

    return pd.DataFrame(rows)


def ray_box_intersection_length(
    start_km: np.ndarray,
    direction: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    max_distance_km: float,
    epsilon: float = 1e-12,
) -> float:
    """Return path length inside an axis-aligned box using the slab method."""

    t_min = 0.0
    t_max = max_distance_km

    for axis in range(3):
        origin = start_km[axis]
        step = direction[axis]

        if abs(step) < epsilon:
            if origin < box_min[axis] or origin > box_max[axis]:
                return 0.0
            continue

        t1 = (box_min[axis] - origin) / step
        t2 = (box_max[axis] - origin) / step
        near = min(t1, t2)
        far = max(t1, t2)
        t_min = max(t_min, near)
        t_max = min(t_max, far)

        if t_max <= t_min:
            return 0.0

    return max(0.0, t_max - t_min)


def build_intersection_matrix(grid: VoxelGrid, rays: list[RayObservation]) -> tuple[np.ndarray, pd.DataFrame]:
    """Build dense A matrix and return it with a voxel lookup table."""

    voxel_table = voxel_index_table(grid)
    matrix = np.zeros((len(rays), len(voxel_table)), dtype=float)

    for voxel in voxel_table.itertuples(index=False):
        box_min = np.array([voxel.x_min_km, voxel.y_min_km, voxel.z_min_km], dtype=float)
        box_max = np.array([voxel.x_max_km, voxel.y_max_km, voxel.z_max_km], dtype=float)
        for ray in rays:
            matrix[ray.ray_id, voxel.voxel_id] = ray_box_intersection_length(
                ray.start_km,
                ray.direction,
                box_min,
                box_max,
                ray.slant_range_km,
            )

    return matrix, voxel_table


def validation_table(grid: VoxelGrid, rays: list[RayObservation], matrix: np.ndarray) -> pd.DataFrame:
    """Compare row sums in A with one whole-domain ray-box intersection."""

    domain_min = np.array([grid.x_edges[0], grid.y_edges[0], grid.z_edges[0]], dtype=float)
    domain_max = np.array([grid.x_edges[-1], grid.y_edges[-1], grid.z_edges[-1]], dtype=float)
    rows: list[dict[str, float | int | str]] = []

    for ray in rays:
        domain_length = ray_box_intersection_length(
            ray.start_km,
            ray.direction,
            domain_min,
            domain_max,
            ray.slant_range_km,
        )
        voxel_sum = float(matrix[ray.ray_id].sum())
        rows.append(
            {
                "ray_id": ray.ray_id,
                "receiver_id": ray.receiver_id,
                "domain_intersection_km": domain_length,
                "sum_voxel_lengths_km": voxel_sum,
                "absolute_error_km": abs(domain_length - voxel_sum),
                "nonzero_voxels": int(np.count_nonzero(matrix[ray.ray_id])),
            }
        )

    return pd.DataFrame(rows)


def matrix_frame(matrix: np.ndarray, voxel_table: pd.DataFrame) -> pd.DataFrame:
    labels = voxel_table.sort_values("voxel_id")["voxel_label"].tolist()
    frame = pd.DataFrame(matrix, columns=labels)
    frame.insert(0, "ray_id", np.arange(matrix.shape[0]))
    return frame


def save_outputs(
    output_dir: str | Path,
    matrix: np.ndarray,
    voxel_table: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_frame(matrix, voxel_table).to_csv(output_dir / "A_matrix.csv", index=False)
    voxel_table.to_csv(output_dir / "voxel_lookup.csv", index=False)
    validation.to_csv(output_dir / "validation_lengths.csv", index=False)

