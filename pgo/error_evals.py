import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import islice
from scipy.stats import linregress

sys.path.append(os.getcwd())
sys.path.append("/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo")
[sys.path.append(i) for i in [".", ".."]]

from pose_graph_optimization.utils import *
from src.utils.pose import (
    get_global_and_relative_gt_trackings,
    matrix_to_pose_vector,
    pose_vector_to_matrix
)
import pgo


LABELS = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_all_relative_trackings(input_pred_path, input_gt_path, nr_of_scans=None, start=1):

    data = os.listdir(input_pred_path)
    data = islice(data, start - 1, start + nr_of_scans - 1) if nr_of_scans is not None else data

    pred_inbetween_all = []
    gt_inbetween_all = []

    for i, el in enumerate(tqdm(data, desc="Loading scans", total=nr_of_scans)):

        sweep_path = os.path.join(input_pred_path, el, "export.h5")
        if not os.path.isfile(sweep_path):
            continue

        with h5py.File(sweep_path, "r") as f_pred:

            nr_of_frames = len(f_pred["images"])

            pred_inbetween = np.array(f_pred["pred_tracking_loc"][:nr_of_frames])

        gt_file = os.path.join(input_gt_path, f"{el}.h5")

        with h5py.File(gt_file, "r") as f_gt:

            gt = np.array(f_gt["tracking"][:nr_of_frames])
            _, gt_inbetween = get_global_and_relative_gt_trackings(gt)

        pred_inbetween_all.append(pred_inbetween)
        gt_inbetween_all.append(gt_inbetween)

    pred_inbetween_all = np.concatenate(pred_inbetween_all, axis=0)
    gt_inbetween_all = np.concatenate(gt_inbetween_all, axis=0)

    return pred_inbetween_all, gt_inbetween_all


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def compute_pose_statistics(pred, gt):

    pred_tracking = np.stack([matrix_to_pose_vector(T) for T in pred])
    gt_tracking = np.stack([matrix_to_pose_vector(T) for T in gt])

    errors_real = pred_tracking - gt_tracking
    errors_abs = np.abs(errors_real)

    stats = {
        "pred_tracking": pred_tracking,
        "gt_tracking": gt_tracking,
        "errors_real": errors_real,
        "errors_abs": errors_abs,
        "mean_gt": gt_tracking.mean(axis=0),
        "mean_pred": pred_tracking.mean(axis=0),
        "mean_gt_abs": np.abs(gt_tracking).mean(axis=0),
        "mean_pred_abs": np.abs(pred_tracking).mean(axis=0),
        "mean_abs_error": errors_abs.mean(axis=0),
        "mean_signed_error": errors_real.mean(axis=0),
        "std_error": errors_real.std(axis=0),
    }

    return stats


def print_statistics(stats):

    print("=== Error statistics ===")

    for i, label in enumerate(LABELS):
        print(f"{label}:")
        print(f"  Mean GT       : {stats['mean_gt'][i]:.4f}")
        print(f"  Mean pred     : {stats['mean_pred'][i]:.4f}")
        print(f"  Mean GT abs   : {stats['mean_gt_abs'][i]:.4f}")
        print(f"  Mean pred abs : {stats['mean_pred_abs'][i]:.4f}")
        print(f"  MAE           : {stats['mean_abs_error'][i]:.4f}")
        print(f"  Mean error    : {stats['mean_signed_error'][i]:.4f}")
        print(f"  Std deviation : {stats['std_error'][i]:.4f}")
        print()


# -----------------------------------------------------------------------------
# Distribution analysis
# -----------------------------------------------------------------------------
def plot_error_histograms(stats, bins=60):

    errors_real = stats["errors_real"]

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(6):
        ax_ = ax.flatten()[i]
        ax_.hist(errors_real[:, i], bins=bins)
        ax_.set_title(LABELS[i])
        ax_.set_xlabel("Error")
        ax_.set_ylabel("Count")

    fig.tight_layout()
    fig.canvas.manager.set_window_title("Error histograms")


def plot_qq_plots(stats):

    import scipy.stats as scipy_stats

    errors_real = stats["errors_real"]

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(6):
        scipy_stats.probplot(errors_real[:, i], dist="norm", plot=ax.flatten()[i])
        ax.flatten()[i].set_title(LABELS[i])
    # sortiere Punkte, weise jedem Punkt ein Quantil zu, berechne Idealverteilung anhand von Standardabweichung und Mittelwert, plotte Idealquantile und tatsaechliche Werte

    fig.tight_layout()
    fig.canvas.manager.set_window_title("QQ plots error")


# -----------------------------------------------------------------------------
# Largest 5 % errors
# -----------------------------------------------------------------------------
def get_largest_error_indices(stats, quantile=0.95):

    errors_abs = stats["errors_abs"]

    thresholds = np.quantile(errors_abs, quantile, axis=0)

    largest_indices = [
        np.where(errors_abs[:, i] >= thresholds[i])[0]
        for i in range(6)
    ]

    return largest_indices, thresholds


def plot_values(values, indices=None, labels=None, title="Values"):

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(6):
        ax_ = ax.flatten()[i]

        if indices is not None:
            idx = indices[i]
        else:
            idx = range(values[0].shape[0]-1)

        for value_set, label in zip(values, labels):
            ax_.plot(value_set[idx, i], label=label, alpha=0.8)

        ax_.set_title(LABELS[i])
        ax_.set_xlabel("Selected sample index")

    ax.flatten()[-1].legend()

    fig.tight_layout()
    fig.canvas.manager.set_window_title(title)


# -----------------------------------------------------------------------------
# Bias analysis
# -----------------------------------------------------------------------------
def plot_bias(stats):

    gt_tracking = stats["gt_tracking"]
    errors_real = stats["errors_real"]
    pred_tracking = stats["pred_tracking"]

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(6):

        x = gt_tracking[:, i]
        y = errors_real[:, i]
        
        result = linregress(x, y)

        x_line = np.linspace(x.min(), x.max(), 100)

        ax_ = ax.flatten()[i]

        ax_.scatter(gt_tracking[:, i], errors_real[:, i], s=5, alpha=0.5)
        ax_.axhline(0, color="red", linestyle="--")

        ax_.plot(x_line, result.slope * x_line + result.intercept, color="orange")

        corr = np.corrcoef(gt_tracking[:, i], errors_real[:, i])[0, 1]

        ax_.set_title(f"{LABELS[i]} (corr={corr:.2f})")
        ax_.set_xlabel("Ground truth")
        ax_.set_ylabel("Pred - GT")

        print(f"DOF {i}")
        print(f"  slope     = {result.slope:.6f}")
        print(f"  intercept = {result.intercept:.6f}")
        print(f"  R²        = {result.rvalue**2:.3f}")
        print()

    fig.tight_layout()
    fig.canvas.manager.set_window_title("Bias analysis")


def estimate_gt(pred):

    SLOPE = np.array([ # empiric values
        -0.084166,
        -0.343315,
        -0.155954,
        -0.765020,
        -0.671620,
        -0.612082,
    ])
    INTERCEPT = np.array([
        -0.002105,
        -0.000186,
        0.000503,
        0.001229,
        -0.002765,
        0.001885,
    ])

    gt_est_vector = np.stack([matrix_to_pose_vector(T) for T in pred])
    gt_est = np.zeros(pred.shape)
    gt_est[0] = np.eye(4)

    for i in range(1, gt_est_vector.shape[0], 1):

        gt_est_vector[i] = (gt_est_vector[i] - INTERCEPT) / (1.0 + SLOPE)
        gt_est[i] = pose_vector_to_matrix(gt_est_vector[i])

    return gt_est


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():

    input_pred_path = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run/scans"
    input_gt_path = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec24_converted"

    pred_all, gt_all = load_all_relative_trackings(
        input_pred_path=input_pred_path,
        input_gt_path=input_gt_path,
        nr_of_scans=None,
        start=1,
    )

    pred_stats = compute_pose_statistics(pred_all, gt_all)
    
    print_statistics(pred_stats)
    # -> Fehler sind im Prinzip mittelwertfrei
    # -> Groessere Fehler bei groesseren Werten
    # -> Groesste Fehler bei y und roll (ergibt Sinn, da y Dimension kleiner ist als x Dimension -> weniger Info bei y, mehr Fehler bei x roll)
    # -> Generell sind Winkelfehler viel groesser als T Fehler im Vergleich zu Werten (ergibt Sinn, da Translation recht eindeutig ist)

    # largest_indices, thresholds = get_largest_error_indices(pred_stats, quantile=0.95)

    # print("Largest 5% thresholds:")
    # for label, thr in zip(LABELS, thresholds):
    #     print(f"  {label}: {thr:.4f}")

    # plot_error_histograms(stats)
    # plot_qq_plots(stats)
    # -> errors are basically gaussian (only really small and really big errors are off)
    
    # plot_values(
    #     values=[
    #         pred_stats["errors_real"],
    #     ],
    #     labels=[
    #         "GT",
    #         "Pred",
    #         "Abs error",
    #     ],
    #     title="pred errors"
    # )

    # plot_bias(pred_stats)
    # -> negative correlation between GT and Pred for all DoFs (more for angles than for translation)
    # -> DualTrack tends to underestimate values
    gt_est = estimate_gt(pred_all)
    gt_est_stats = compute_pose_statistics(gt_est, gt_all)
    print_statistics(gt_est_stats)

    # Is linear regression helpful?
    diff_est_pred = np.stack([matrix_to_pose_vector(T) for T in gt_est])-np.stack([matrix_to_pose_vector(T) for T in pred_all])
    diff_pred_gt = np.stack([matrix_to_pose_vector(T) for T in pred_all])-np.stack([matrix_to_pose_vector(T) for T in gt_all])

    for i in range(6):
        print(np.corrcoef(diff_est_pred[:, i], diff_pred_gt[:, i])[0, 1])
    # -> no

    old_error = np.abs(np.stack([matrix_to_pose_vector(T) for T in pred_all]) - np.stack([matrix_to_pose_vector(T) for T in gt_all]))
    new_error = np.abs(np.stack([matrix_to_pose_vector(T) for T in gt_est]) - np.stack([matrix_to_pose_vector(T) for T in gt_all]))

    improved = new_error < old_error

    print(improved.mean())
    # -> no :(

    # plot_bias(gt_est_stats)
    # plot_values(
    #     values=[
    #         gt_est_stats["errors_real"],
    #     ],
    #     labels=[
    #         "GT",
    #         "Pred",
    #         "Abs error"
    #     ],
    #     title="est gt errors"
    # )

    # pgo.plot_motion_vs_error(pred_all, gt_all, title="Scatter plot errors")
    # -> the larger GT the larger error, especially for angles

    plt.show()


if __name__ == "__main__":
    main()