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
from pgo import plot_motion_vs_error
from src.utils.pose import (
    get_global_and_relative_gt_trackings,
    matrix_to_pose_vector,
    pose_vector_to_matrix
)


LABELS = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]


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


def plot_bias(stats):

    gt_tracking = stats["gt_tracking"]
    errors_real = stats["errors_real"]
    pred_tracking = stats["pred_tracking"]

    x_label = "GT"
    y_label = "Pred - GT"

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(6):

        # x = abs(gt_tracking)[:, i]
        # y = abs(errors_real)[:, i]
        x = gt_tracking[:, i]
        y = errors_real[:, i]
        
        result = linregress(x, y)

        x_line = np.linspace(x.min(), x.max(), 100)

        ax_ = ax.flatten()[i]

        ax_.scatter(x, y, s=5, alpha=0.5)
        ax_.axhline(0, color="red", linestyle="--")

        ax_.plot(x_line, result.slope * x_line + result.intercept, color="orange")

        corr = np.corrcoef(x, y)[0, 1]

        ax_.set_title(f"{LABELS[i]} (corr={corr:.2f})")
        ax_.set_xlabel(x_label)
        ax_.set_ylabel(y_label)

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


def variance_statistics(stats, num_bins=10):

    gt_tracking = stats["gt_tracking"]
    errors_real = stats["errors_real"]

    labels = [
        "Tx", "Ty", "Tz",
        "Rx", "Ry", "Rz"
    ]

    for dof in range(6):

        print(f"\n{'=' * 70}")
        print(labels[dof])
        print(f"{'=' * 70}")

        gt_abs = np.abs(gt_tracking[:, dof])

        # Quantile bins -> approximately same number of samples per bin
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(gt_abs, quantiles)

        # Remove duplicate edges (can happen if many identical values exist)
        bin_edges = np.unique(bin_edges)

        for b in range(len(bin_edges) - 1):

            lower = bin_edges[b]
            upper = bin_edges[b + 1]

            if b == len(bin_edges) - 2:
                mask = (gt_abs >= lower) & (gt_abs <= upper) # -> true false mask
            else:
                mask = (gt_abs >= lower) & (gt_abs < upper)
                print(mask)

            err = errors_real[mask, dof] # -> get values of indices for which mask is true

            if len(err) == 0:
                continue

            gt_bin = gt_abs[mask]

            print(
                f"Bin {b + 1:2d}: "
                f"median GT={np.median(gt_bin):7.4f}  "
                f"[{lower:7.4f}, {upper:7.4f}]  "
                f"N={len(err):5d}  "
                f"mean={err.mean():9.5f}  "
                f"mean_abs={np.abs(err).mean():9.5f}  "
                f"std={err.std():9.5f}"
            )
    

def la_improvement_stats(pred_all, gt_all, gt_est, num_quantiles=10):

    pred = np.stack([matrix_to_pose_vector(T) for T in pred_all])
    gt = np.stack([matrix_to_pose_vector(T) for T in gt_all])
    est = np.stack([matrix_to_pose_vector(T) for T in gt_est])

    old_error = np.abs(pred - gt)
    new_error = np.abs(est - gt)
    gt_abs = np.abs(gt)
    pred_abs = np.abs(pred)

    labels = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]

    for dof in range(6):

        print(f"\n{labels[dof]}")

        edges = np.quantile(
            gt_abs[:, dof],
            np.linspace(0, 1, num_quantiles + 1)
        )
        edges = np.unique(edges)

        for q in range(len(edges) - 1):

            lower = edges[q]
            upper = edges[q + 1]

            if q == len(edges) - 2:
                mask = (
                    (gt_abs[:, dof] >= lower) &
                    (gt_abs[:, dof] <= upper)
                )
            else:
                mask = (
                    (gt_abs[:, dof] >= lower) &
                    (gt_abs[:, dof] < upper)
                )

            if not np.any(mask):
                continue

            old = old_error[mask, dof]
            new = new_error[mask, dof]
            gt_mag = gt_abs[mask, dof]
            pred_mag = pred_abs[mask, dof]

            improved = new < old
            worsened = new > old

            percentage = 100.0 * improved.mean()

            if np.any(improved):
                improvement_rate = (
                    (old[improved] - new[improved]) /
                    np.maximum(old[improved], 1e-12)
                ).mean() * 100.0

                improved_gt_error = (
                    new[improved] /
                    np.maximum(gt_mag[improved], 1e-12)
                ).mean() * 100.0

                improved_pred_error = (
                                    new[improved] /
                                    np.maximum(pred_mag[improved], 1e-12)
                                ).mean() * 100.0
            else:
                improvement_rate = np.nan
                improved_gt_error = np.nan

            if np.any(worsened):
                worsening_rate = (
                    (new[worsened] - old[worsened]) /
                    np.maximum(old[worsened], 1e-12)
                ).mean() * 100.0

                worsened_gt_error = (
                    new[worsened] /
                    np.maximum(gt_mag[worsened], 1e-12)
                ).mean() * 100.0

                worsened_pred_error = (
                                    new[worsened] /
                                    np.maximum(pred_mag[worsened], 1e-12)
                                ).mean() * 100.0
            else:
                worsening_rate = np.nan
                worsened_gt_error = np.nan

            print(
                f"Q{q+1:2d} "
                f"[{lower:.4f}, {upper:.4f}] "
                f"N={mask.sum():5d} | "
                f"Improved={percentage:6.2f}% | "
                f"Avg improvement={improvement_rate:6.2f}% | "
                f"Avg worsening={worsening_rate:6.2f}% | "
                f"Improved err/GT={improved_gt_error:6.2f}% | "
                f"Worsened err/GT={worsened_gt_error:6.2f}% | "
                f"Improved err/pred={improved_pred_error:6.2f}% | "
                f"Worsened err/pred={worsened_pred_error:6.2f}%"
            )

def main():

    # -----------------------------------------------------------------------------
    # data loading
    # -----------------------------------------------------------------------------
    input_pred_path = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run/scans"
    input_gt_path = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec24_converted"

    # input_pred_path = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_25_val/validation_run/scans"
    # input_gt_path = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec25_converted"

    pred_all, gt_all = load_all_relative_trackings(
        input_pred_path=input_pred_path,
        input_gt_path=input_gt_path,
        nr_of_scans=None,
        start=1,
    )

    # -----------------------------------------------------------------------------
    # pred statistics
    # -----------------------------------------------------------------------------
    pred_stats = compute_pose_statistics(pred_all, gt_all)
    
    # print_statistics(pred_stats)
    # -> Fehler sind im Prinzip mittelwertfrei
    # -> Groessere Fehler bei groesseren Werten
    # -> grosse Standardabweichung

    # -> Groesste Fehler bei y und roll (ergibt Sinn, da y Dimension kleiner ist als x Dimension -> weniger Info bei y, mehr Fehler bei x roll)
    # -> Generell sind Winkelfehler viel groesser als T Fehler im Vergleich zu Werten (ergibt Sinn, da Translation recht eindeutig ist)


    # -----------------------------------------------------------------------------
    # errors
    # -----------------------------------------------------------------------------
    largest_indices, thresholds = get_largest_error_indices(pred_stats, quantile=0.95)

    print("Largest 5% thresholds:")
    for label, thr in zip(LABELS, thresholds):
        print(f"  {label}: {thr:.4f}")

    # plot_error_histograms(pred_stats)
    # plot_qq_plots(pred_stats)
    # # -> errors are basically gaussian (only really small and really big errors are off)
    
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

    # plot_motion_vs_error(pred_all, gt_all, title="Scatter plot errors")
    # # -> the larger GT the larger error, especially for angles

    # variance_statistics(pred_stats)
    # -> abs mean der Fehler steigt ueber Wertebereich von GT gar nicht bis moderat an, erst bei sehr grossen Werten steigt es stark
    # -> std dasselbe
    # -> staerkerer Anstieg bei Rotation
    # -> Wieder: besonders grosse Werte bedeuten grosse Fehler


    # -----------------------------------------------------------------------------
    # linear approximation
    # -----------------------------------------------------------------------------
    plot_bias(pred_stats)
    # -> negative correlation between GT and Error for all DoFs (more for angles than for translation)
    # -> x y z is fine, pitch and yaw abs estimations too big (negative values too small, positive too big), no drift in neither
    # gt_est = estimate_gt(pred_all)
    # gt_est_stats = compute_pose_statistics(gt_est, gt_all)
    # print_statistics(gt_est_stats)

    # Did linear approximation improve anything?
    # la_improvement_stats(pred_all, gt_all, gt_est, 10)
    # -> improvement especially for large GT values and translation
    # -> improvement of about 20 - 50%
    # -> worsening of several 100 to 1000 percent
    # -> worsened is way bigger than GT compared to improved, though regarding pred the distinction is not that great, so simple filtering of worsenings by magnitude not possible 

    # plot_bias(gt_est_stats)
    # plot_bias(pred_stats)
    # plot_values(
    #     values=[
    #         gt_est_stats["gt_tracking"],
    #         gt_est_stats["pred_tracking"],
    #     ],
    #     labels=[
    #         "GT",
    #         "Pred",
    #         "Abs error"
    #     ],
    #     title="est gt errors"
    # )

    plt.show()


if __name__ == "__main__":
    main()