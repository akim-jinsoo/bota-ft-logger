"""Plot Bota force/torque CSV logs.

Usage:
    python plot_bota_data.py bota_readings_YYYY-MM-DD_HH-MM-SS.csv
    python plot_bota_data.py bota_readings_YYYY-MM-DD_HH-MM-SS.csv --smooth 20

The script will:
    - Load the CSV written by `Bota-Rokubi-Logger.py`.
    - Create a time axis starting at t = 0 s.
    - Normalize Fx, Fy, Fz, Mx, My, Mz so that the first
      sample is zero (baseline subtraction).
    - Optionally smooth the data with a moving average filter.
    - Show a figure with two subplots: forces (top) and
      torques (bottom).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


FORCE_COLUMNS = ["Fx", "Fy", "Fz"]
TORQUE_COLUMNS = ["Mx", "My", "Mz"]


def load_bota_csv(path: Path, use_device_time: bool = True):
    """Load Bota CSV and return time array and data columns.

    Args:
        path: Path to CSV file
        use_device_time: If True, use device_timestamp (smooth, exact sampling rate).
                        If False, use host_time_s (subject to OS buffering jitter).

    Returns:
        t: list[float]          -- time in seconds, starting at 0
        data: dict[str, list]   -- keys are column names such as
                                   "Fx", "Fy", "Fz", "Mx", "My", "Mz".
    """

    t_raw = []
    device_timestamps = []
    columns = {name: [] for name in FORCE_COLUMNS + TORQUE_COLUMNS}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV file has no header; is this a Bota log?")

        for row in reader:
            try:
                t_raw.append(float(row["host_time_s"]))
                device_timestamps.append(int(row["device_timestamp"]))
                for name in columns:
                    columns[name].append(float(row[name]))
            except (KeyError, ValueError) as exc:
                # Skip malformed rows
                print(f"Skipping malformed row: {exc}")
                continue

    if not t_raw:
        raise RuntimeError("No valid data rows found in CSV.")

    if use_device_time:
        # Use device timestamp: convert uint32 counter to seconds
        # The sensor timestep is 0.00001953125 * SINC_LENGTH
        # With SINC_LENGTH=256, timestep = 0.005 seconds = 5 ms
        timestep = 0.005  # seconds per device timestamp increment
        t0_device = device_timestamps[0]
        t = [(dt - t0_device) * timestep for dt in device_timestamps]
    else:
        # Use host time (subject to OS buffering)
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


def smooth_data(data: dict[str, list[float]], window: int):
    """Apply moving average smoothing to each channel.

    Args:
        data: Dictionary of signal arrays
        window: Window size for moving average (must be > 0)

    Returns:
        Smoothed data dict with same keys, values will be shorter by (window-1)
    """
    if window <= 1:
        return data

    smoothed = {}
    for name, values in data.items():
        if len(values) < window:
            # Not enough data to smooth
            smoothed[name] = values
            continue
        # Use uniform convolution for moving average
        kernel = np.ones(window) / window
        smoothed[name] = np.convolve(values, kernel, mode='valid').tolist()
    return smoothed


def clip_data(t: list[float], data: dict[str, list[float]], t_start: float, t_end: float):
    """Clip time and data arrays to specified time range.

    Args:
        t: Time array
        data: Dictionary of signal arrays (same length as t)
        t_start: Start time (inclusive)
        t_end: End time (inclusive)

    Returns:
        Tuple of (clipped_time, clipped_data)
    """
    t_arr = np.array(t)
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(f"No data found in time range [{t_start}, {t_end}]")

    t_clipped = [t[i] for i in indices]
    data_clipped = {name: [values[i] for i in indices] for name, values in data.items()}

    return t_clipped, data_clipped


def plot_force_torque(t, forces, torques, title: str | None = None, smooth_window: int | None = None):
    fig, (ax_f, ax_m) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    for name in FORCE_COLUMNS:
        ax_f.plot(t, forces[name], label=name)
    ylabel_f = "Force [N] (normalized)"
    if smooth_window and smooth_window > 1:
        ylabel_f += f" [smoothed, window={smooth_window}]"
    ax_f.set_ylabel(ylabel_f)
    ax_f.grid(True, linestyle="--", alpha=0.3)
    ax_f.legend(loc="best")

    for name in TORQUE_COLUMNS:
        ax_m.plot(t, torques[name], label=name)
    ylabel_m = "Torque [Nm] (normalized)"
    if smooth_window and smooth_window > 1:
        ylabel_m += f" [smoothed, window={smooth_window}]"
    ax_m.set_ylabel(ylabel_m)
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
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        metavar="N",
        help="Apply moving average smoothing with window size N (default: 1, no smoothing). "
             "Try values like 10-50 to reduce quantization steps.",
    )
    parser.add_argument(
        "--clip",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="Clip data to time range [START, END] seconds (e.g., --clip 21 22.5)",
    )
    parser.add_argument(
        "--use-host-time",
        action="store_true",
        help="Use host_time_s instead of device_timestamp (default: use device time for smooth sampling)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    path = Path(args.csv_file)
    if not path.is_file():
        raise SystemExit(f"CSV file not found: {path}")

    # Use device timestamp by default (smooth), or host time if requested
    use_device_time = not args.use_host_time
    t, data = load_bota_csv(path, use_device_time=use_device_time)
    data_norm = normalize_to_zero(data)

    # Apply smoothing if requested
    if args.smooth > 1:
        data_norm = smooth_data(data_norm, args.smooth)
        # Truncate time array to match smoothed data length
        # (convolution with mode='valid' shortens output by window-1)
        t = t[args.smooth - 1:]

    # Apply time clipping if requested
    if args.clip:
        t_start, t_end = args.clip
        t, data_norm = clip_data(t, data_norm, t_start, t_end)

    forces = {name: data_norm[name] for name in FORCE_COLUMNS}
    torques = {name: data_norm[name] for name in TORQUE_COLUMNS}

    plot_force_torque(t, forces, torques, title=path.name, smooth_window=args.smooth)


if __name__ == "__main__":
    main()
