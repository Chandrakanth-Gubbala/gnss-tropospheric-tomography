from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tomography import (  # noqa: E402
    build_grid,
    generate_rays,
    plot_profile_comparison,
    ray_path_matrix,
    rmse,
    simulate_observations,
    solve_art,
    solve_tikhonov,
    synthetic_receivers,
    synthetic_water_vapor_field,
)


def main() -> None:
    grid = build_grid(dx_km=18.0, dz_km=1.5)
    truth = synthetic_water_vapor_field(grid)
    receivers = synthetic_receivers(count=9)
    rays = generate_rays(receivers, rays_per_receiver=28)
    matrix = ray_path_matrix(grid, rays)
    observations = simulate_observations(matrix, truth)

    tikhonov = solve_tikhonov(matrix, observations, alpha=1.2)
    art = solve_art(matrix, observations, iterations=10, relaxation=0.28)

    print(f"Receivers: {len(receivers)}")
    print(f"Rays: {len(rays)}")
    print(f"Voxels: {grid.size}")
    print(f"Tikhonov RMSE: {rmse(truth, tikhonov):.3f}")
    print(f"ART RMSE: {rmse(truth, art):.3f}")

    plotted = plot_profile_comparison(
        grid,
        truth,
        {"Tikhonov": tikhonov, "ART": art},
        ROOT / "outputs" / "profile_comparison.png",
    )
    if plotted:
        print("Wrote outputs/profile_comparison.png")
    else:
        print("Skipped plot because matplotlib is not installed.")


if __name__ == "__main__":
    main()
