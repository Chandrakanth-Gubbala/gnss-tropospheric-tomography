from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    rng = np.random.default_rng(42)
    receivers = pd.DataFrame(
        {
            "receiver_id": ["RX01", "RX02", "RX03", "RX04", "RX05"],
            "x_km": [-55.0, 42.0, 50.0, -48.0, 0.0],
            "y_km": [-45.0, -48.0, 38.0, 42.0, 0.0],
            "z_km": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    rows = []
    for receiver in receivers.itertuples(index=False):
        for _ in range(6):
            elevation = rng.uniform(20.0, 60.0)
            rows.append(
                {
                    "receiver_id": receiver.receiver_id,
                    "x_km": receiver.x_km,
                    "y_km": receiver.y_km,
                    "z_km": receiver.z_km,
                    "azimuth_deg": rng.uniform(0.0, 360.0),
                    "elevation_deg": elevation,
                    "slant_range_km": 12.0 / np.sin(np.deg2rad(elevation)),
                }
            )

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_input_data.txt"
    pd.DataFrame(rows).round(3).to_csv(output_path, sep=" ", index=False, header=False)
    print(f"Wrote {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

