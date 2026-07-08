from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intersection_matrix import build_intersection_matrix, load_ray_table, save_outputs, validation_table  # noqa: E402
from tomography import build_grid  # noqa: E402


def main() -> None:
    sample_path = ROOT / "examples" / "sample_input_data.txt"
    rays = load_ray_table(sample_path)

    grid = build_grid(
        x_span_km=(-70.0, 70.0),
        y_span_km=(-70.0, 70.0),
        z_span_km=(0.0, 12.0),
        dx_km=20.0,
        dz_km=2.0,
    )
    matrix, voxels = build_intersection_matrix(grid, rays)
    validation = validation_table(grid, rays, matrix)
    save_outputs(ROOT / "outputs" / "matrix_demo", matrix, voxels, validation)

    print(f"Rays: {len(rays)}")
    print(f"Voxels: {grid.size}")
    print(f"Matrix shape: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"Max validation error: {validation['absolute_error_km'].max():.6f} km")
    print("Wrote outputs/matrix_demo/A_matrix.csv")
    print("Wrote outputs/matrix_demo/voxel_lookup.csv")
    print("Wrote outputs/matrix_demo/validation_lengths.csv")


if __name__ == "__main__":
    main()

