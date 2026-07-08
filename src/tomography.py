"""Synthetic GNSS tropospheric tomography utilities.

The module keeps the public example data-free by using a local Cartesian grid
and generated observations. It mirrors the structure of a real retrieval
pipeline without embedding station files, raw observations, or private paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VoxelGrid:
    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            len(self.x_edges) - 1,
            len(self.y_edges) - 1,
            len(self.z_edges) - 1,
        )

    @property
    def size(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            0.5 * (self.x_edges[:-1] + self.x_edges[1:]),
            0.5 * (self.y_edges[:-1] + self.y_edges[1:]),
            0.5 * (self.z_edges[:-1] + self.z_edges[1:]),
        )

    def flatten_index(self, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.shape
        return (ix * ny * nz) + (iy * nz) + iz


@dataclass(frozen=True)
class Ray:
    receiver_id: str
    start_km: np.ndarray
    end_km: np.ndarray
    elevation_deg: float


def build_grid(
    x_span_km: tuple[float, float] = (-90.0, 90.0),
    y_span_km: tuple[float, float] = (-90.0, 90.0),
    z_span_km: tuple[float, float] = (0.0, 12.0),
    dx_km: float = 15.0,
    dz_km: float = 1.0,
) -> VoxelGrid:
    x_edges = np.arange(x_span_km[0], x_span_km[1] + dx_km, dx_km)
    y_edges = np.arange(y_span_km[0], y_span_km[1] + dx_km, dx_km)
    z_edges = np.arange(z_span_km[0], z_span_km[1] + dz_km, dz_km)
    return VoxelGrid(x_edges=x_edges, y_edges=y_edges, z_edges=z_edges)


def synthetic_water_vapor_field(grid: VoxelGrid, seed: int = 7) -> np.ndarray:
    """Create a smooth 3D field with vertical decay and a moisture anomaly."""

    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*grid.centers, indexing="ij")
    background = 18.0 * np.exp(-z / 2.5)
    anomaly = 5.0 * np.exp(-((x - 25.0) ** 2 + (y + 15.0) ** 2) / (2 * 32.0**2))
    shear = 1.5 * np.sin((x + y) / 70.0) * np.exp(-z / 6.0)
    small_scale = rng.normal(0.0, 0.08, size=grid.shape)
    return np.maximum(background + anomaly + shear + small_scale, 0.0).reshape(-1)


def synthetic_receivers(count: int = 8, radius_km: float = 70.0, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    jitter = rng.normal(0, 4.0, size=(count, 2))
    return pd.DataFrame(
        {
            "receiver_id": [f"RX{i + 1:02d}" for i in range(count)],
            "x_km": radius_km * np.cos(angles) + jitter[:, 0],
            "y_km": radius_km * np.sin(angles) + jitter[:, 1],
            "z_km": np.zeros(count),
        }
    )


def generate_rays(
    receivers: pd.DataFrame,
    rays_per_receiver: int = 24,
    top_height_km: float = 12.0,
    seed: int = 11,
) -> list[Ray]:
    rng = np.random.default_rng(seed)
    rays: list[Ray] = []

    for row in receivers.itertuples(index=False):
        start = np.array([row.x_km, row.y_km, row.z_km], dtype=float)
        for _ in range(rays_per_receiver):
            azimuth = rng.uniform(0, 2 * np.pi)
            elevation = rng.uniform(15, 75)
            horizontal_distance = top_height_km / np.tan(np.deg2rad(elevation))
            end = start + np.array(
                [
                    horizontal_distance * np.cos(azimuth),
                    horizontal_distance * np.sin(azimuth),
                    top_height_km,
                ]
            )
            rays.append(
                Ray(
                    receiver_id=row.receiver_id,
                    start_km=start,
                    end_km=end,
                    elevation_deg=float(elevation),
                )
            )

    return rays


def ray_path_matrix(grid: VoxelGrid, rays: list[Ray], samples_per_ray: int = 140) -> np.ndarray:
    """Approximate path length inside each voxel by sampling ray midpoints."""

    matrix = np.zeros((len(rays), grid.size), dtype=float)
    nx, ny, nz = grid.shape

    for ray_index, ray in enumerate(rays):
        vector = ray.end_km - ray.start_km
        path_length = float(np.linalg.norm(vector))
        segment_length = path_length / samples_per_ray
        fractions = (np.arange(samples_per_ray) + 0.5) / samples_per_ray
        points = ray.start_km + fractions[:, None] * vector

        ix = np.searchsorted(grid.x_edges, points[:, 0], side="right") - 1
        iy = np.searchsorted(grid.y_edges, points[:, 1], side="right") - 1
        iz = np.searchsorted(grid.z_edges, points[:, 2], side="right") - 1
        inside = (0 <= ix) & (ix < nx) & (0 <= iy) & (iy < ny) & (0 <= iz) & (iz < nz)

        indices = grid.flatten_index(ix[inside], iy[inside], iz[inside])
        np.add.at(matrix[ray_index], indices, segment_length)

    return matrix


def simulate_observations(
    path_matrix: np.ndarray,
    field: np.ndarray,
    noise_std: float = 0.35,
    seed: int = 19,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return path_matrix @ field + rng.normal(0.0, noise_std, size=path_matrix.shape[0])


def solve_tikhonov(path_matrix: np.ndarray, observations: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    lhs = path_matrix.T @ path_matrix + alpha * np.eye(path_matrix.shape[1])
    rhs = path_matrix.T @ observations
    solution = np.linalg.solve(lhs, rhs)
    return np.maximum(solution, 0.0)


def solve_art(
    path_matrix: np.ndarray,
    observations: np.ndarray,
    iterations: int = 8,
    relaxation: float = 0.35,
) -> np.ndarray:
    estimate = np.zeros(path_matrix.shape[1], dtype=float)
    row_norms = np.sum(path_matrix * path_matrix, axis=1)

    for _ in range(iterations):
        for row, target, norm in zip(path_matrix, observations, row_norms, strict=True):
            if norm <= 1e-12:
                continue
            residual = target - float(row @ estimate)
            estimate += relaxation * residual * row / norm
            estimate = np.maximum(estimate, 0.0)

    return estimate


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


def vertical_profile(grid: VoxelGrid, field: np.ndarray) -> pd.DataFrame:
    cube = field.reshape(grid.shape)
    z_centers = grid.centers[2]
    return pd.DataFrame(
        {
            "height_km": z_centers,
            "mean_value": cube.mean(axis=(0, 1)),
            "p10": np.percentile(cube, 10, axis=(0, 1)),
            "p90": np.percentile(cube, 90, axis=(0, 1)),
        }
    )


def plot_profile_comparison(
    grid: VoxelGrid,
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    output_path: str | Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    truth_profile = vertical_profile(grid, truth)
    plt.plot(truth_profile["mean_value"], truth_profile["height_km"], label="Synthetic truth", linewidth=3)

    for label, field in estimates.items():
        profile = vertical_profile(grid, field)
        plt.plot(profile["mean_value"], profile["height_km"], label=label)

    plt.xlabel("Water vapor proxy")
    plt.ylabel("Height (km)")
    plt.title("Mean vertical reconstruction profile")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True
