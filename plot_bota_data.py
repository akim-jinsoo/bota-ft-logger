"""Plot Bota force/torque CSV logs.

Usage:
    python plot_bota_data.py bota_readings_YYYY-MM-DD_HH-MM-SS.csv

The script will:
    - Load the CSV written by `Bota-Rokubi-Logger.py`.
    - Create a time axis starting at t = 0 s.
    - Normalize Fx, Fy, Fz, Mx, My, Mz so that the first
      sample is zero (baseline subtraction).
    - Show a figure with two subplots: forces (top) and
      torques (bottom).
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


FORCE_COLUMNS = ["Fx", "Fy", "Fz"]
TORQUE_COLUMNS = ["Mx", "My", "Mz"]


def load_bota_csv(path: Path):
    """Load Bota CSV and return time array and data columns.

    Returns:
        t: list[float]          -- time in seconds, starting at 0
        data: dict[str, list]   -- keys are column names such as
                                   "Fx", "Fy", "Fz", "Mx", "My", "Mz".
    """

    t_raw = []
    columns = {name: [] for name in FORCE_COLUMNS + TORQUE_COLUMNS}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV file has no header; is this a Bota log?")

        for row in reader:
            try:
                t_raw.append(float(row["host_time_s"]))
                for name in columns:
                    columns[name].append(float(row[name]))
            except (KeyError, ValueError) as exc:
                # Skip malformed rows
                print(f"Skipping malformed row: {exc}")
                continue

    if not t_raw:
        raise RuntimeError("No valid data rows found in CSV.")

    t0 = t_raw[0]
    t = [ti - t0 for ti in t_raw]
    return t, columns


def normalize_to_zero(data: dict[str, list[float]]):
    """Normalize each channel so that its first value becomes zero.

    This subtracts the value at index 0 from all subsequent samples.
    """

    norm = {}
    for name, values in data.items():
        if not values:
            norm[name] = []
            continue
        offset = values[0]
        norm[name] = [v - offset for v in values]
    return norm


def plot_force_torque(t, forces, torques, title: str | None = None):
    fig, (ax_f, ax_m) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    for name in FORCE_COLUMNS:
        ax_f.plot(t, forces[name], label=name)
    ax_f.set_ylabel("Force [N] (normalized)")
    ax_f.grid(True, linestyle="--", alpha=0.3)
    ax_f.legend(loc="best")

    for name in TORQUE_COLUMNS:
        ax_m.plot(t, torques[name], label=name)
    ax_m.set_ylabel("Torque [Nm] (normalized)")
    ax_m.set_xlabel("Time [s]")
    ax_m.grid(True, linestyle="--", alpha=0.3)
    ax_m.legend(loc="best")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot normalized Bota force/torque data from CSV.")
    parser.add_argument(
        "csv_file",
        help="Path to CSV file produced by Bota-Rokubi-Logger.py",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    path = Path(args.csv_file)
    if not path.is_file():
        raise SystemExit(f"CSV file not found: {path}")

    t, data = load_bota_csv(path)
    data_norm = normalize_to_zero(data)

    forces = {name: data_norm[name] for name in FORCE_COLUMNS}
    torques = {name: data_norm[name] for name in TORQUE_COLUMNS}

    plot_force_torque(t, forces, torques, title=path.name)


if __name__ == "__main__":
    main()
