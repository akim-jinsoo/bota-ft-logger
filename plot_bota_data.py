"""Plot Bota force/torque CSV logs.

USAGE MODES:
============

1. Single-trial mode (plot one CSV file):
        python plot_bota_data.py bota_readings_YYYY-MM-DD_HH-MM-SS.csv [options]

2. Multi-trial overlay mode (overlay trials from one batch directory):
        python plot_bota_data.py --batch "0 degrees" --channel Fz [options]

3. Multi-batch comparison mode (compare averages across multiple configurations):
        python plot_bota_data.py --compare "0 degrees" "30 degrees" "-30 degrees" --channel Fz [options]

PARAMETERS:
===========

Positional:
        csv_file                Path to a single CSV file (single-trial mode only).

Mode Selection:
        --batch DIR             Directory containing multiple CSV trials to overlay.
        --compare DIR [DIR ...] Compare multiple batch directories on one plot.
                                Shows average (and optional envelope) for each batch.

Channel Selection:
        --channel NAME          Plot only one channel: Fx, Fy, Fz, Mx, My, or Mz.
                                Required for --align, --peak, --envelope, --compare.

Alignment & Clipping:
        --align                 Align trials by onset of movement (when signal starts changing).
                                Uses derivative threshold detection. Requires --channel.
        --clip START END        Clip to absolute time range [START, END] seconds.
                                Time resets to 0 at the clip start.
        --peak BEFORE AFTER     Clip around the peak value of the channel.
                                E.g., --peak 1 2 clips 1s before and 2s after peak.
                                Peak is set to t=0. Requires --channel.

Data Transformation:
        --flip                  Flip Y-axis values (negate all data: + becomes -, - becomes +).
        --smooth N              Moving average smoothing with window size N samples.
                                Default: 1 (no smoothing). Try 5-20 for noisy data.

Plotting Options:
        --envelope              Show shaded min/max envelope around the mean.
                                In --compare mode, shows envelope for each batch.
        --max-trials N          Max trials to load per batch directory. Default: 5.

EXAMPLES:
=========

# Plot single file with all channels:
python plot_bota_data.py "0 degrees/bota_readings_2025-12-10_16-56-36.csv"

# Overlay 5 trials from one batch, aligned by Fz onset:
python plot_bota_data.py --batch "0 degrees" --channel Fz --align

# Same, but show average with min/max envelope:
python plot_bota_data.py --batch "0 degrees" --channel Fz --align --envelope

# Clip around peak, flip values, smooth:
python plot_bota_data.py --batch "0 degrees" --channel Fz --align --flip --peak 1 1 --smooth 10

# Compare 5 angle configurations on one plot:
python plot_bota_data.py --compare "0 degrees" "30 degrees" "-30 degrees" "60 degrees" "-60 degrees" \\
        --channel Fz --align --flip --peak 1 1 --envelope

TUNABLE CONSTANTS (edit in code):
=================================
        ONSET_NOISE_MULTIPLIER  Threshold multiplier for onset detection (default: 17.0)
        ONSET_MIN_FRACTION      Minimum fraction of max derivative for onset (default: 0.3)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


FORCE_COLUMNS = ["Fx", "Fy", "Fz"]
TORQUE_COLUMNS = ["Mx", "My", "Mz"]

# Alignment onset detection thresholds (adjust these for noisy data)
ONSET_NOISE_MULTIPLIER = 17.0  # Derivative must exceed this × noise floor
ONSET_MIN_FRACTION = 0.3       # Or this fraction of max derivative, whichever is larger


def load_bota_csv(path: Path):
    """Load Bota CSV and return time array and data columns.

    Args:
        path: Path to CSV file

    Returns:
        t: list[float]          -- time in seconds, starting at 0
        data: dict[str, list]   -- keys are column names such as
                                   "Fx", "Fy", "Fz", "Mx", "My", "Mz".
    """

    device_timestamps = []
    columns = {name: [] for name in FORCE_COLUMNS + TORQUE_COLUMNS}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV file has no header; is this a Bota log?")

        for row in reader:
            try:
                device_timestamps.append(int(row["device_timestamp"]))
                for name in columns:
                    columns[name].append(float(row[name]))
            except (KeyError, ValueError) as exc:
                # Skip malformed rows
                print(f"Skipping malformed row: {exc}")
                continue

    if not device_timestamps:
        raise RuntimeError("No valid data rows found in CSV.")

    # device_timestamp is in microseconds, convert to seconds
    t0 = device_timestamps[0]
    t = [(ts - t0) / 1_000_000.0 for ts in device_timestamps]
    
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


def flip_data(data: dict[str, list[float]]):
    """Flip (negate) all values in each channel."""
    return {name: [-v for v in values] for name, values in data.items()}


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
        Tuple of (clipped_time, clipped_data) - time is reset so clipped region starts at 0
    """
    t_arr = np.array(t)
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(f"No data found in time range [{t_start}, {t_end}]")

    t_clipped = [t[i] for i in indices]
    # Reset time so the clipped region starts at 0
    t0 = t_clipped[0]
    t_clipped = [tv - t0 for tv in t_clipped]
    
    data_clipped = {name: [values[i] for i in indices] for name, values in data.items()}

    return t_clipped, data_clipped


def clip_around_peak(
    t: list[float],
    data: dict[str, list[float]],
    channel: str,
    before: float,
    after: float,
):
    """Clip data around the peak (max absolute value) of a channel.

    Args:
        t: Time array
        data: Dictionary of signal arrays
        channel: Which channel to find the peak in
        before: Seconds to include before the peak
        after: Seconds to include after the peak

    Returns:
        Tuple of (clipped_time, clipped_data) - time is reset so peak is at t=0
    """
    if channel not in data:
        raise ValueError(f"Channel {channel} not found in data.")

    y = np.array(data[channel])
    idx_peak = int(np.argmax(np.abs(y)))
    t_peak = t[idx_peak]

    t_start = t_peak - before
    t_end = t_peak + after

    # Clip to this range
    t_arr = np.array(t)
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(f"No data found in peak range [{t_start}, {t_end}]")

    t_clipped = [t[i] for i in indices]
    # Reset time so the peak is at t=0
    t_clipped = [tv - t_peak for tv in t_clipped]

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


def load_and_preprocess_trial(
    path: Path,
    smooth_window: int = 1,
    clip: tuple[float, float] | None = None,
):
    """Load one Bota CSV trial and apply normalization, optional smoothing and clipping.

    Returns:
        t: list[float] (starts at 0)
        data_norm: dict[str, list[float]]
    """

    t, data = load_bota_csv(path)
    data_norm = normalize_to_zero(data)

    # Smoothing
    if smooth_window > 1:
        data_norm = smooth_data(data_norm, smooth_window)
        # Truncate time array to match smoothed data length
        t = t[smooth_window - 1 :]

    # Optional time clipping
    if clip is not None:
        t_start, t_end = clip
        t, data_norm = clip_data(t, data_norm, t_start, t_end)

    return t, data_norm


def align_and_trim_trials(
    trials: list[tuple[list[float], dict[str, list[float]]]],
):
    """Align and trim multiple trials to a common time window.

    Assumes each trial's time array starts at 0. We:
        - Compute the minimum duration across trials.
        - Clip each trial to [0, min_duration].

    Args:
        trials: List of (t, data) for each trial.

    Returns:
        aligned_t: list[float]  -- common time array (taken from first trial, clipped)
        aligned_data: list[dict[str, list[float]]] -- list of data dicts per trial, clipped.
    """

    if not trials:
        raise ValueError("No trials provided for alignment.")

    # Compute duration for each trial
    durations = [trial_t[-1] for trial_t, _ in trials if trial_t]
    if not durations:
        raise ValueError("No valid time data in trials.")
    min_duration = min(durations)

    aligned_t = None
    aligned_data_list: list[dict[str, list[float]]] = []

    for idx, (trial_t, trial_data) in enumerate(trials):
        if not trial_t:
            raise ValueError(f"Trial {idx} has empty time array.")

        # Clip this trial to [0, min_duration]
        t_clipped, data_clipped = clip_data(trial_t, trial_data, 0.0, min_duration)

        # Use the first trial's clipped time as the common time array
        if aligned_t is None:
            aligned_t = t_clipped
        else:
            # Ensure same length by truncating to min length
            min_len = min(len(aligned_t), len(t_clipped))
            aligned_t = aligned_t[:min_len]
            for key in data_clipped:
                data_clipped[key] = data_clipped[key][:min_len]

        aligned_data_list.append(data_clipped)

    # Final consistency: ensure all data dicts have arrays of same length as aligned_t
    final_len = len(aligned_t)
    for d in aligned_data_list:
        for key in d:
            if len(d[key]) > final_len:
                d[key] = d[key][:final_len]

    return aligned_t, aligned_data_list


def compute_derivative(t: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    """Compute numerical derivative dy/dt using central differences.

    Returns:
        t_mid: time points for derivative (len = len(t) - 2)
        dy_dt: derivative values
    """
    if len(t) < 3 or len(y) < 3:
        return [], []

    t_arr = np.array(t, dtype=float)
    y_arr = np.array(y, dtype=float)
    dt = np.diff(t_arr)
    # avoid division by zero
    dt[dt == 0.0] = np.min(dt[dt > 0.0]) if np.any(dt > 0.0) else 1.0

    # central difference: dy/dt at interior points
    dy = y_arr[2:] - y_arr[:-2]
    dt_mid = t_arr[2:] - t_arr[:-2]
    dt_mid[dt_mid == 0.0] = np.min(dt_mid[dt_mid > 0.0]) if np.any(dt_mid > 0.0) else 1.0

    dy_dt = dy / dt_mid
    t_mid = t_arr[1:-1]
    return t_mid.tolist(), dy_dt.tolist()


def align_trials_by_derivative_peak(
    trials: list[tuple[list[float], dict[str, list[float]]]],
    channel: str,
):
    """Align trials by onset of movement in a given channel.

    For each trial, find when the signal first starts moving (derivative exceeds
    a threshold based on the noise floor). This aligns trials by the start of
    the event rather than the peak.

    Returns:
        List of (shifted_t, data) tuples, one per trial.
    """

    if not trials:
        raise ValueError("No trials provided for derivative-based alignment.")

    shifted_trials: list[tuple[list[float], dict[str, list[float]]]] = []
    
    for trial_t, trial_data in trials:
        if channel not in trial_data:
            raise ValueError(f"Channel {channel} not in trial data.")
        
        t_mid, dy_dt = compute_derivative(trial_t, trial_data[channel])
        if not t_mid:
            raise ValueError("Not enough samples to compute derivative.")
        
        dy_dt_arr = np.abs(dy_dt)
        
        # Estimate noise floor from first 10% of data (assumed to be baseline)
        baseline_samples = max(10, len(dy_dt_arr) // 10)
        noise_floor = np.std(dy_dt_arr[:baseline_samples])
        
        # Threshold: derivative must exceed noise_multiplier × noise floor
        # or min_fraction of max derivative, whichever is larger
        threshold = max(
            ONSET_NOISE_MULTIPLIER * noise_floor,
            ONSET_MIN_FRACTION * np.max(dy_dt_arr)
        )
        
        # Find first index where derivative exceeds threshold
        onset_indices = np.where(dy_dt_arr > threshold)[0]
        if len(onset_indices) == 0:
            # Fallback to peak if no clear onset found
            idx_onset = int(np.argmax(dy_dt_arr))
        else:
            idx_onset = onset_indices[0]
        
        t_onset = t_mid[idx_onset]

        # Shift time so that this onset is at t = 0
        shifted_t = [ti - t_onset for ti in trial_t]
        shifted_trials.append((shifted_t, trial_data))

    # Now normalize so the earliest time across all trials is 0
    min_time = min(min(t) for t, _ in shifted_trials)
    normalized_trials: list[tuple[list[float], dict[str, list[float]]]] = []
    for trial_t, trial_data in shifted_trials:
        normalized_t = [ti - min_time for ti in trial_t]
        normalized_trials.append((normalized_t, trial_data))

    return normalized_trials


def plot_multi_trials(
    trials: list[tuple[list[float], dict[str, list[float]]]],
    title: str | None = None,
    smooth_window: int | None = None,
    channel: str | None = None,
):
    """Overlay multiple trials, each with its own time array.

    If `channel` is provided, only that channel is plotted (one subplot).
    Otherwise, this falls back to plotting all forces and torques.
    """

    num_trials = len(trials)

    if channel is not None:
        # Single-channel overlay
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for trial_idx, (t, data) in enumerate(trials, start=1):
            if channel not in data:
                raise ValueError(f"Channel {channel} not found in trial data.")
            ax.plot(t, data[channel], label=f"trial {trial_idx}", alpha=0.7)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{channel} (normalized)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        plt.show()
        return

    # Default: all forces and torques (original behavior for multi-trial mode)
    fig, (ax_f, ax_m) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # Plot forces
    for trial_idx, (t, data) in enumerate(trials, start=1):
        label_suffix = f" (trial {trial_idx})" if num_trials > 1 else ""
        for name in FORCE_COLUMNS:
            ax_f.plot(t, data[name], label=f"{name}{label_suffix}", alpha=0.6)

    ylabel_f = "Force [N] (normalized)"
    if smooth_window and smooth_window > 1:
        ylabel_f += f" [smoothed, window={smooth_window}]"
    ax_f.set_ylabel(ylabel_f)
    ax_f.grid(True, linestyle="--", alpha=0.3)
    ax_f.legend(loc="best", fontsize="small", ncol=2)

    # Plot torques
    for trial_idx, (t, data) in enumerate(trials, start=1):
        label_suffix = f" (trial {trial_idx})" if num_trials > 1 else ""
        for name in TORQUE_COLUMNS:
            ax_m.plot(t, data[name], label=f"{name}{label_suffix}", alpha=0.6)

    ylabel_m = "Torque [Nm] (normalized)"
    if smooth_window and smooth_window > 1:
        ylabel_m += f" [smoothed, window={smooth_window}]"
    ax_m.set_ylabel(ylabel_m)
    ax_m.set_xlabel("Time [s]")
    ax_m.grid(True, linestyle="--", alpha=0.3)
    ax_m.legend(loc="best", fontsize="small", ncol=2)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    plt.show()


def plot_envelope(
    trials: list[tuple[list[float], dict[str, list[float]]]],
    channel: str,
    title: str | None = None,
):
    """Plot average of trials with shaded min/max envelope.

    Interpolates all trials to a common time axis, then computes mean, min, max
    at each time point.

    Args:
        trials: List of (t, data) tuples for each trial
        channel: Which channel to plot (e.g., 'Fz')
        title: Optional plot title
    """
    if not trials:
        raise ValueError("No trials to plot.")

    # Find common time range (intersection of all trials)
    t_min = max(min(t) for t, _ in trials)
    t_max = min(max(t) for t, _ in trials)

    if t_min >= t_max:
        raise ValueError("Trials do not overlap in time.")

    # Create common time axis with fine resolution
    # Use the average sample rate from first trial
    t0, _ = trials[0]
    if len(t0) > 1:
        avg_dt = (t0[-1] - t0[0]) / (len(t0) - 1)
    else:
        avg_dt = 0.001
    num_points = int((t_max - t_min) / avg_dt) + 1
    t_common = np.linspace(t_min, t_max, num_points)

    # Interpolate each trial onto common time axis
    interpolated = []
    for trial_t, trial_data in trials:
        if channel not in trial_data:
            raise ValueError(f"Channel {channel} not found in trial data.")
        y_interp = np.interp(t_common, trial_t, trial_data[channel])
        interpolated.append(y_interp)

    # Stack into 2D array: (num_trials, num_points)
    stacked = np.array(interpolated)

    # Compute statistics
    y_mean = np.mean(stacked, axis=0)
    y_min = np.min(stacked, axis=0)
    y_max = np.max(stacked, axis=0)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Shaded envelope (min to max)
    ax.fill_between(t_common, y_min, y_max, alpha=0.3, label="min/max range")

    # Average line
    ax.plot(t_common, y_mean, linewidth=2, label=f"mean (n={len(trials)})")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{channel} (normalized)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    plt.show()


def compute_batch_stats(
    trials: list[tuple[list[float], dict[str, list[float]]]],
    channel: str,
):
    """Compute mean, min, max for a batch of trials on a common time axis.

    Returns:
        t_common: Common time array
        y_mean: Mean values
        y_min: Min values
        y_max: Max values
    """
    if not trials:
        raise ValueError("No trials to process.")

    # Find common time range (intersection of all trials)
    t_min = max(min(t) for t, _ in trials)
    t_max = min(max(t) for t, _ in trials)

    if t_min >= t_max:
        raise ValueError("Trials do not overlap in time.")

    # Create common time axis
    t0, _ = trials[0]
    if len(t0) > 1:
        avg_dt = (t0[-1] - t0[0]) / (len(t0) - 1)
    else:
        avg_dt = 0.001
    num_points = int((t_max - t_min) / avg_dt) + 1
    t_common = np.linspace(t_min, t_max, num_points)

    # Interpolate each trial onto common time axis
    interpolated = []
    for trial_t, trial_data in trials:
        if channel not in trial_data:
            raise ValueError(f"Channel {channel} not found in trial data.")
        y_interp = np.interp(t_common, trial_t, trial_data[channel])
        interpolated.append(y_interp)

    stacked = np.array(interpolated)
    y_mean = np.mean(stacked, axis=0)
    y_min = np.min(stacked, axis=0)
    y_max = np.max(stacked, axis=0)

    return t_common, y_mean, y_min, y_max


def plot_multi_batch_envelope(
    batch_data: list[tuple[str, list[float], list[float], list[float], list[float]]],
    channel: str,
    show_envelope: bool = True,
    title: str | None = None,
):
    """Plot multiple batches with their averages and optional envelopes.

    Args:
        batch_data: List of (label, t_common, y_mean, y_min, y_max) for each batch
        channel: Channel name for axis label
        show_envelope: Whether to show shaded min/max envelope
        title: Optional plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Use a colormap for distinct colors
    colors = plt.cm.tab10.colors

    for idx, (label, t_common, y_mean, y_min, y_max) in enumerate(batch_data):
        color = colors[idx % len(colors)]

        # Shaded envelope
        if show_envelope:
            ax.fill_between(t_common, y_min, y_max, alpha=0.2, color=color)

        # Average line
        ax.plot(t_common, y_mean, linewidth=2, label=label, color=color)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{channel} (normalized)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot normalized Bota force/torque data from CSV.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        help="Path to CSV file produced by Bota-Rokubi-Logger.py (single-trial mode). "
             "Omit this and use --angle-dir for multi-trial overlay mode.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Directory containing multiple Bota CSV trials to overlay (multi-trial mode).",
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
        "--peak",
        nargs=2,
        type=float,
        metavar=("BEFORE", "AFTER"),
        help="Clip around the peak value: --peak 1 1 clips 1s before and 1s after the peak (requires --channel).",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Only plot a single channel (e.g., Fx, Fy, Fz, Mx, My, Mz) in multi-trial mode.",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="In multi-trial mode, align trials by peak derivative of the chosen channel.",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip the Y-axis values (negate all data: positive becomes negative and vice versa).",
    )
    parser.add_argument(
        "--envelope",
        action="store_true",
        help="Plot average of trials with shaded min/max envelope (requires --channel).",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        metavar="DIR",
        help="Compare multiple batch directories on one plot (e.g., --compare '0 degrees' '30 degrees' '-30 degrees').",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=5,
        help="Maximum number of trials to overlay in multi-trial mode (default: 5).",
    )
    return parser.parse_args()


def process_batch_dir(batch_dir: Path, args):
    """Process a single batch directory and return aligned/clipped trials.

    Args:
        batch_dir: Path to the batch directory
        args: Parsed command line arguments

    Returns:
        List of (t, data) tuples for the processed trials
    """
    if not batch_dir.is_dir():
        raise SystemExit(f"Batch directory not found: {batch_dir}")

    csv_paths = sorted(batch_dir.glob("bota_readings_*.csv"))
    if not csv_paths:
        raise SystemExit(f"No bota_readings_*.csv files found in {batch_dir}")

    csv_paths = csv_paths[: args.max_trials]

    trials: list[tuple[list[float], dict[str, list[float]]]] = []

    for path in csv_paths:
        t, data_norm = load_and_preprocess_trial(
            path,
            smooth_window=args.smooth,
            clip=None,
        )
        if args.flip:
            data_norm = flip_data(data_norm)
        trials.append((t, data_norm))

    # Align by derivative peak if requested
    if args.align:
        if args.channel is None:
            raise SystemExit("--align requires --channel to specify which signal to use.")
        trials = align_trials_by_derivative_peak(trials, channel=args.channel)

    # Optional clip after alignment
    if args.clip:
        t_start, t_end = args.clip
        clipped_trials: list[tuple[list[float], dict[str, list[float]]]] = []
        for trial_t, trial_data in trials:
            t_clipped, data_clipped = clip_data(trial_t, trial_data, t_start, t_end)
            clipped_trials.append((t_clipped, data_clipped))
        trials = clipped_trials

    # Optional clip around peak
    if args.peak:
        if args.channel is None:
            raise SystemExit("--peak requires --channel to specify which signal to find the peak in.")
        before, after = args.peak
        clipped_trials: list[tuple[list[float], dict[str, list[float]]]] = []
        for trial_t, trial_data in trials:
            t_clipped, data_clipped = clip_around_peak(
                trial_t, trial_data, args.channel, before, after
            )
            clipped_trials.append((t_clipped, data_clipped))
        trials = clipped_trials

    return trials


def main():
    args = _parse_args()

    # Multi-batch comparison mode
    if args.compare:
        if args.channel is None:
            raise SystemExit("--compare requires --channel to specify which signal to plot.")

        batch_data: list[tuple[str, list[float], list[float], list[float], list[float]]] = []

        for batch_name in args.compare:
            batch_dir = Path(batch_name)
            trials = process_batch_dir(batch_dir, args)

            # Compute statistics for this batch
            t_common, y_mean, y_min, y_max = compute_batch_stats(trials, args.channel)
            batch_data.append((batch_dir.name, t_common, y_mean, y_min, y_max))

        plot_multi_batch_envelope(
            batch_data,
            channel=args.channel,
            show_envelope=args.envelope,
            title=f"Comparison: {args.channel} across {len(batch_data)} configurations",
        )
        return

    # Single-batch multi-trial mode
    if args.batch:
        batch_dir = Path(args.batch)
        aligned_trials = process_batch_dir(batch_dir, args)

        # Plot: envelope mode or overlay mode
        batch_label = batch_dir.name
        if args.envelope:
            if args.channel is None:
                raise SystemExit("--envelope requires --channel to specify which signal to plot.")
            plot_envelope(
                aligned_trials,
                channel=args.channel,
                title=f"Batch: {batch_label} (n={len(aligned_trials)} trials)",
            )
        else:
            plot_multi_trials(
                aligned_trials,
                title=f"Batch: {batch_label} (n={len(aligned_trials)} trials)",
                smooth_window=args.smooth,
                channel=args.channel,
            )
        return

    # Single-trial mode (existing behavior)
    if not args.csv_file:
        raise SystemExit(
            "You must provide either a CSV file (single-trial mode) or --batch (multi-trial mode)."
        )

    path = Path(args.csv_file)
    if not path.is_file():
        raise SystemExit(f"CSV file not found: {path}")

    t, data = load_bota_csv(path)
    data_norm = normalize_to_zero(data)

    # Apply flip if requested
    if args.flip:
        data_norm = flip_data(data_norm)

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
    try:
        main()
    except KeyboardInterrupt:
        pass  # Exit quietly on Ctrl+C
