import argparse
import os
from pathlib import Path
import sys

import h5py
from itertools import islice
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy.stats import linregress, probplot

sys.path.append(os.getcwd())
sys.path.append("/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo")
[sys.path.append(i) for i in [".", ".."]]

from src.utils.pose import (
    get_global_and_relative_gt_trackings,
    matrix_to_pose_vector,
    pose_vector_to_matrix,
)

LABELS = ["Tx", "Ty", "Tz", "Roll", "Pitch", "Yaw"]


def load_array(path, dataset=None):

    path = Path(path)

    if path.suffix == ".npy":

        return np.load(path)
    
    if path.suffix == ".npz":

        data = np.load(path)
        key = dataset or data.files[0]
        return data[key]
    
    if path.suffix in {".h5", ".hdf5"}:

        with h5py.File(path, "r") as handle:

            if dataset is None:

                dataset = next(key for key in handle if isinstance(handle[key], h5py.Dataset))

            return np.asarray(handle[dataset])
        
    raise ValueError(f"Unsupported pose-array file: {path}")


def load_all_relative_trackings(input_pred_path, input_gt_path, nr_of_scans=None, start=1):

    data = os.listdir(input_pred_path)
    data = islice(data, start - 1, start + nr_of_scans - 1) if nr_of_scans is not None else data

    pred_inbetween_all = []
    gt_inbetween_all = []

    for el in data:

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

    if not pred_inbetween_all:

        raise ValueError(f"No scan data found in {input_pred_path}")
    
    return np.concatenate(pred_inbetween_all, axis=0), np.concatenate(gt_inbetween_all, axis=0)


def compute_pose_statistics(gt, pred):

    gt_vectors = np.stack([matrix_to_pose_vector(pose) for pose in gt])
    pred_vectors = np.stack([matrix_to_pose_vector(pose) for pose in pred])

    signed_error = pred_vectors - gt_vectors
    absolute_error = np.abs(signed_error)

    return {
        "gt": gt_vectors,
        "pred": pred_vectors,
        "errors_real": signed_error,
        "errors_abs": absolute_error,
        "mean_gt": gt_vectors.mean(axis=0),
        "mean_pred": pred_vectors.mean(axis=0),
        "mean_gt_abs": np.abs(gt_vectors).mean(axis=0),
        "mean_pred_abs": np.abs(pred_vectors).mean(axis=0),
        "mean_abs_error": absolute_error.mean(axis=0),
        "mean_signed_error": signed_error.mean(axis=0),
        "std_error": signed_error.std(axis=0),
    }


def print_statistics(stats):

    lines = ["=== Error statistics ==="]

    for index, label in enumerate(LABELS):
        
        lines.extend([
            f"{label}:",
            f"  Mean GT       : {stats['mean_gt'][index]:.4f}",
            f"  Mean pred     : {stats['mean_pred'][index]:.4f}",
            f"  Mean GT abs   : {stats['mean_gt_abs'][index]:.4f}",
            f"  Mean pred abs : {stats['mean_pred_abs'][index]:.4f}",
            f"  MAE           : {stats['mean_abs_error'][index]:.4f}",
            f"  Mean error    : {stats['mean_signed_error'][index]:.4f}",
            f"  Std deviation : {stats['std_error'][index]:.4f}",
        ])
    return "\n".join(lines)


def error_histograms(stats, bins):

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for index, axis in enumerate(axes.flat):

        axis.hist(stats["errors_real"][:, index], bins=bins)
        axis.set_title(LABELS[index])
        axis.set_xlabel("Error")
        axis.set_ylabel("Count")

    fig.tight_layout()
    return fig


def qq_plots(stats):

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for index, axis in enumerate(axes.flat):

        probplot(stats["errors_real"][:, index], dist="norm", plot=axis)
        axis.set_title(LABELS[index])

    fig.tight_layout()
    return fig


def largest_error_indices(stats, quantile):

    errors_abs = stats["errors_abs"]
    thresholds = np.quantile(errors_abs, quantile, axis=0)
    indices = [np.where(errors_abs[:, index] >= thresholds[index])[0] for index in range(6)]
    lines = [f"=== Largest errors at quantile {quantile:.3f} ==="]

    for label, threshold, values in zip(LABELS, thresholds, indices):

        lines.append(f"{label}: threshold={threshold:.6f}, count={len(values)}")

    return indices, "\n".join(lines)


def values_plot(stats, indices):

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for index, axis in enumerate(axes.flat):

        selected = indices[index] if indices is not None else np.arange(len(stats["gt"]))
        axis.plot(selected, stats["gt"][selected, index], label="GT")
        axis.plot(selected, stats["pred"][selected, index], label="pred")
        axis.set_title(LABELS[index])
        axis.set_xlabel("Sample index")

    axes.flat[-1].legend()
    fig.tight_layout()
    return fig


def fit_bias(gt, pred, mode="gt"):

    if mode not in {"gt", "pred"}:
        raise ValueError(f"Unsupported bias mode: {mode}")

    slopes = np.empty(len(LABELS))
    intercepts = np.empty(len(LABELS))
    thresholds = np.empty((2, len(LABELS)))
    values = gt if mode == "gt" else pred

    for index in range(len(LABELS)):

        error = pred[:, index] - gt[:, index]
        result = linregress(values[:, index], error)
        slopes[index] = result.slope
        intercepts[index] = result.intercept

        # Find the outermost values that still violate the expected error sign.
        negative_values = values[:, index] < 0
        positive_values = values[:, index] > 0
        negative_violations = negative_values & (error < 0)
        positive_violations = positive_values & (error > 0)
        thresholds[0, index] = (
            values[negative_violations, index].min()
            if np.any(negative_violations)
            else 0.0
        )
        thresholds[1, index] = (
            values[positive_violations, index].max()
            if np.any(positive_violations)
            else 0.0
        )

    return slopes, intercepts, thresholds


def bias_plot(stats, mode="gt"):

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    lines = ["=== Bias statistics ==="]
    gt = stats["gt"]
    pred = stats["pred"]
    slopes, intercepts, thresholds = fit_bias(gt, pred, mode)

    for index, axis in enumerate(axes.flat):
        x = gt[:, index] if mode == "gt" else pred[:, index]
        y = stats["errors_real"][:, index]
        x_line = np.linspace(x.min(), x.max(), 100)
        axis.scatter(x, y, s=5, alpha=0.5)
        axis.axhline(0, color="red", linestyle="--")
        axis.plot(x_line, slopes[index] * x_line + intercepts[index], color="orange")
        axis.axvline(
            thresholds[0, index],
            color="blue",
            linestyle=":",
            label="negative threshold",
        )
        axis.axvline(
            thresholds[1, index],
            color="green",
            linestyle=":",
            label="positive threshold",
        )
        axis.set_title(f"{LABELS[index]} (corr={np.corrcoef(x, y)[0, 1]:.2f})")
        axis.set_xlabel(mode)
        axis.set_ylabel("pred - GT")
        axis.legend(fontsize="small")
        lines.append(
            f"{LABELS[index]} ({mode}): slope={slopes[index]:.6f}, "
            f"intercept={intercepts[index]:.6f}, "
            f"negative_threshold={thresholds[0, index]:.6f}, "
            f"positive_threshold={thresholds[1, index]:.6f}"
        )

    fig.tight_layout()
    return fig, "\n".join(lines)


def variance_statistics(stats, num_bins):

    lines = ["=== Variance statistics ==="]
    gt = stats["gt"]
    errors = stats["errors_real"]

    for dof, label in enumerate(LABELS):

        lines.append(label)
        edges = np.unique(np.quantile(np.abs(gt[:, dof]), np.linspace(0, 1, num_bins + 1)))

        for bin_index in range(len(edges) - 1):

            lower, upper = edges[bin_index:bin_index + 2]

            if bin_index == len(edges) - 2:

                mask = (np.abs(gt[:, dof]) >= lower) & (np.abs(gt[:, dof]) <= upper)

            else:

                mask = (np.abs(gt[:, dof]) >= lower) & (np.abs(gt[:, dof]) < upper)

            values = errors[mask, dof]

            if len(values):
                
                lines.append(f"  bin={bin_index + 1} range=[{lower:.5f}, {upper:.5f}] N={len(values)} mean={values.mean():.6f} mean_abs={np.abs(values).mean():.6f} std={values.std():.6f}")

    return "\n".join(lines)


def estimate_gt(gt, pred, mode="gt"):

    slope, intercept, _ = fit_bias(
        np.stack([matrix_to_pose_vector(pose) for pose in gt]),
        np.stack([matrix_to_pose_vector(pose) for pose in pred]),
        mode,
    )
    pred_vectors = np.stack([matrix_to_pose_vector(pose) for pose in pred])
    estimated = np.empty_like(pred)
    estimated[0] = gt[0]

    for index in range(1, len(pred_vectors)):

        if mode == "gt":
            estimated_vector = (pred_vectors[index] - intercept) / (1.0 + slope)
        else:
            estimated_error = slope * pred_vectors[index] + intercept
            estimated_vector = pred_vectors[index] - estimated_error
        estimated[index] = pose_vector_to_matrix(estimated_vector)

    return estimated


def apply_threshold_correction(pred, slopes, intercepts, thresholds, correction_factor, mode="pred"):

    if mode != "pred":
        raise ValueError("Threshold correction requires thresholds defined in prediction space")

    pred_vectors = np.stack([matrix_to_pose_vector(pose) for pose in pred])
    estimated_gt = (pred_vectors - intercepts) / (1.0 + slopes)
    corrected_vectors = pred_vectors.copy()

    for dof in range(len(LABELS)):

        negative = (pred_vectors[:, dof] < thresholds[0, dof])
        positive = (pred_vectors[:, dof] > thresholds[1, dof])
        correction = correction_factor * estimated_gt[:, dof]
        # corrected_vectors[negative | positive, dof] += correction[negative | positive]
        corrected_vectors[negative | positive, dof] += correction_factor*corrected_vectors[negative | positive, dof]

    return np.stack([pose_vector_to_matrix(vector) for vector in corrected_vectors])


def linear_approximation_statistics(
    gt,
    pred,
    num_quantiles,
    mode="gt",
    estimated_tracking=None,
    title="Linear approximation",
):

    estimated_tracking = estimated_tracking if estimated_tracking is not None else estimate_gt(gt, pred, mode)
    estimated_vectors = np.stack([matrix_to_pose_vector(pose) for pose in estimated_tracking])
    gt_vectors = np.stack([matrix_to_pose_vector(pose) for pose in gt])
    pred_vectors = np.stack([matrix_to_pose_vector(pose) for pose in pred])

    old_error = np.abs(pred_vectors - gt_vectors)
    new_error = np.abs(estimated_vectors - gt_vectors)
    values_abs = np.abs(gt_vectors if mode == "gt" else pred_vectors)
    gt_abs = np.abs(gt_vectors)
    pred_abs = np.abs(pred_vectors)
    lines = [f"=== {title} statistics ==="]

    for dof, label in enumerate(LABELS):
        
        edges = np.unique(np.quantile(values_abs[:, dof], np.linspace(0, 1, num_quantiles + 1)))
        lines.append(label)

        for index in range(len(edges) - 1):

            lower, upper = edges[index:index + 2]
            mask = (values_abs[:, dof] >= lower) & ((values_abs[:, dof] <= upper) if index == len(edges) - 2 else (values_abs[:, dof] < upper))

            if not np.any(mask):

                continue

            old = old_error[mask, dof]
            new = new_error[mask, dof]
            gt_magnitude = gt_abs[mask, dof]
            pred_magnitude = pred_abs[mask, dof]

            improved = new < old
            worsened = new > old

            if np.any(improved):

                improvement_rate = ((old[improved] - new[improved]) / np.maximum(old[improved], 1e-12)).mean() * 100.0
                improved_gt_error = (new[improved] / np.maximum(gt_magnitude[improved], 1e-12)).mean() * 100.0
                improved_pred_error = (new[improved] / np.maximum(pred_magnitude[improved], 1e-12)).mean() * 100.0
                
            else:

                improvement_rate = np.nan
                improved_gt_error = np.nan
                improved_pred_error = np.nan

            if np.any(worsened):

                worsening_rate = ((new[worsened] - old[worsened]) / np.maximum(old[worsened], 1e-12)).mean() * 100.0
                worsened_gt_error = (new[worsened] / np.maximum(gt_magnitude[worsened], 1e-12)).mean() * 100.0
                worsened_pred_error = (new[worsened] / np.maximum(pred_magnitude[worsened], 1e-12)).mean() * 100.0

            else:

                worsening_rate = np.nan
                worsened_gt_error = np.nan
                worsened_pred_error = np.nan

            lines.append(
                f"  bin={index + 1} [{lower:.4f}, {upper:.4f}] N={mask.sum()} | "
                f"Improved={100 * improved.mean():.2f}% | "
                f"Avg improvement={improvement_rate:.2f}% | "
                f"Avg worsening={worsening_rate:.2f}% | "
                f"Improved err/GT={improved_gt_error:.2f}% | "
                f"Worsened err/GT={worsened_gt_error:.2f}% | "
                f"Improved err/pred={improved_pred_error:.2f}% | "
                f"Worsened err/pred={worsened_pred_error:.2f}%"
            )
    return "\n".join(lines)


def save_figure(fig, output_dir, name):
    
    fig.savefig(Path(output_dir) / f"{name}.png", dpi=150)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    input_pred_path = config.inputs.input_pred_path
    input_gt_path = config.inputs.input_gt_path
    nr_of_scans = config.inputs.get("nr_of_scans")
    start = config.inputs.get("start", 1)

    pred_all, gt_all = load_all_relative_trackings(
        input_pred_path=input_pred_path,
        input_gt_path=input_gt_path,
        nr_of_scans=nr_of_scans,
        start=start,
    )
    stats = compute_pose_statistics(gt_all, pred_all)
    quantile = float(config.parameters.get("quantile", 0.95))
    bins = int(config.parameters.get("bins", 60))
    num_bins = int(config.parameters.get("num_bins", 10))
    num_quantiles = int(config.parameters.get("num_quantiles", 10))
    correction_factor = float(config.parameters.get("correction_factor", 0.3))

    largest_indices, largest_text = largest_error_indices(stats, quantile)
    bias_gt_figure, bias_gt_text = bias_plot(stats, "gt")
    bias_pred_figure, bias_pred_text = bias_plot(stats, "pred")
    slopes, intercepts, thresholds = fit_bias(stats["gt"], stats["pred"], "pred")
    corrected_pred = apply_threshold_correction(
        pred_all,
        slopes,
        intercepts,
        thresholds,
        correction_factor,
        "pred",
    )
    figures = {
        "error_histograms": error_histograms(stats, bins),
        "error_qq_plots": qq_plots(stats),
        "values": values_plot(stats, largest_indices),
        "bias_gt": bias_gt_figure,
        "bias_pred": bias_pred_figure,
    }
    text_sections = [
        print_statistics(stats),
        largest_text,
        variance_statistics(stats, num_bins),
        bias_gt_text,
        bias_pred_text,
        linear_approximation_statistics(gt_all, pred_all, num_quantiles, mode="gt"),
        linear_approximation_statistics(gt_all, pred_all, num_quantiles, mode="pred"),
    ]
    text_sections.append(
        linear_approximation_statistics(
            gt_all,
            pred_all,
            num_quantiles,
            mode="pred",
            estimated_tracking=corrected_pred,
            title=f"Threshold correction (pred, factor={correction_factor:.3f})",
        )
    )
    text = "\n\n".join(text_sections)

    output_dir = Path(config.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.txt").write_text(text + "\n", encoding="utf-8")
    for name, figure in figures.items():
        save_figure(figure, output_dir, name)

    print(f"Wrote {output_dir / 'results.txt'}")


if __name__ == "__main__":
    main()
