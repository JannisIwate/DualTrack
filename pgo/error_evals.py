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

LABELS = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]


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


def compute_pose_statistics(first, second):
    first_vectors = np.stack([matrix_to_pose_vector(pose) for pose in first])
    second_vectors = np.stack([matrix_to_pose_vector(pose) for pose in second])
    signed_error = first_vectors - second_vectors
    absolute_error = np.abs(signed_error)
    return {
        "first": first_vectors,
        "second": second_vectors,
        "errors_real": signed_error,
        "errors_abs": absolute_error,
        "mean_second": second_vectors.mean(axis=0),
        "mean_first": first_vectors.mean(axis=0),
        "mean_second_abs": np.abs(second_vectors).mean(axis=0),
        "mean_first_abs": np.abs(first_vectors).mean(axis=0),
        "mean_abs_error": absolute_error.mean(axis=0),
        "mean_signed_error": signed_error.mean(axis=0),
        "std_error": signed_error.std(axis=0),
    }


def print_statistics(stats):
    lines = ["=== Error statistics ==="]
    for index, label in enumerate(LABELS):
        lines.extend([
            f"{label}:",
            f"  Mean second   : {stats['mean_second'][index]:.4f}",
            f"  Mean first    : {stats['mean_first'][index]:.4f}",
            f"  Mean second abs: {stats['mean_second_abs'][index]:.4f}",
            f"  Mean first abs: {stats['mean_first_abs'][index]:.4f}",
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
        selected = indices[index] if indices is not None else np.arange(len(stats["first"]))
        axis.plot(selected, stats["first"][selected, index], label="first")
        axis.plot(selected, stats["second"][selected, index], label="second")
        axis.set_title(LABELS[index])
        axis.set_xlabel("Sample index")
    axes.flat[-1].legend()
    fig.tight_layout()
    return fig


def bias_plot(stats):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    lines = ["=== Bias statistics ==="]
    for index, axis in enumerate(axes.flat):
        x = stats["second"][:, index]
        y = stats["errors_real"][:, index]
        result = linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        axis.scatter(x, y, s=5, alpha=0.5)
        axis.axhline(0, color="red", linestyle="--")
        axis.plot(x_line, result.slope * x_line + result.intercept, color="orange")
        axis.set_title(f"{LABELS[index]} (corr={np.corrcoef(x, y)[0, 1]:.2f})")
        axis.set_xlabel("Second")
        axis.set_ylabel("First - Second")
        lines.append(f"{LABELS[index]}: slope={result.slope:.6f}, intercept={result.intercept:.6f}, R2={result.rvalue ** 2:.3f}")
    fig.tight_layout()
    return fig, "\n".join(lines)


def variance_statistics(stats, num_bins):
    lines = ["=== Variance statistics ==="]
    second = stats["second"]
    errors = stats["errors_real"]
    for dof, label in enumerate(LABELS):
        lines.append(label)
        edges = np.unique(np.quantile(np.abs(second[:, dof]), np.linspace(0, 1, num_bins + 1)))
        for bin_index in range(len(edges) - 1):
            lower, upper = edges[bin_index:bin_index + 2]
            if bin_index == len(edges) - 2:
                mask = (np.abs(second[:, dof]) >= lower) & (np.abs(second[:, dof]) <= upper)
            else:
                mask = (np.abs(second[:, dof]) >= lower) & (np.abs(second[:, dof]) < upper)
            values = errors[mask, dof]
            if len(values):
                lines.append(f"  bin={bin_index + 1} range=[{lower:.5f}, {upper:.5f}] N={len(values)} mean={values.mean():.6f} mean_abs={np.abs(values).mean():.6f} std={values.std():.6f}")
    return "\n".join(lines)


def estimate_second(first):
    slope = np.array([-0.084166, -0.343315, -0.155954, -0.765020, -0.671620, -0.612082])
    intercept = np.array([-0.002105, -0.000186, 0.000503, 0.001229, -0.002765, 0.001885])
    vectors = np.stack([matrix_to_pose_vector(pose) for pose in first])
    estimated = np.empty_like(first)
    estimated[0] = np.eye(4)
    for index in range(1, len(vectors)):
        estimated[index] = pose_vector_to_matrix((vectors[index] - intercept) / (1.0 + slope))
    return estimated


def linear_approximation_statistics(first, second, num_quantiles):
    estimated_vectors = np.stack([matrix_to_pose_vector(pose) for pose in estimate_second(first)])
    first_vectors = np.stack([matrix_to_pose_vector(pose) for pose in first])
    second_vectors = np.stack([matrix_to_pose_vector(pose) for pose in second])
    old_error = np.abs(first_vectors - second_vectors)
    new_error = np.abs(estimated_vectors - second_vectors)
    lines = ["=== Linear approximation statistics ==="]
    for dof, label in enumerate(LABELS):
        edges = np.unique(np.quantile(np.abs(second_vectors[:, dof]), np.linspace(0, 1, num_quantiles + 1)))
        lines.append(label)
        for index in range(len(edges) - 1):
            lower, upper = edges[index:index + 2]
            mask = (np.abs(second_vectors[:, dof]) >= lower) & ((np.abs(second_vectors[:, dof]) <= upper) if index == len(edges) - 2 else (np.abs(second_vectors[:, dof]) < upper))
            if not np.any(mask):
                continue
            improved = new_error[mask, dof] < old_error[mask, dof]
            lines.append(f"  bin={index + 1} N={mask.sum()} improved={100 * improved.mean():.2f}%")
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
    stats = compute_pose_statistics(pred_all, gt_all)
    quantile = float(config.parameters.get("quantile", 0.95))
    bins = int(config.parameters.get("bins", 60))
    num_bins = int(config.parameters.get("num_bins", 10))
    num_quantiles = int(config.parameters.get("num_quantiles", 10))

    largest_indices, largest_text = largest_error_indices(stats, quantile)
    bias_figure, bias_text = bias_plot(stats)
    figures = {
        "error_histograms": error_histograms(stats, bins),
        "qq_plots": qq_plots(stats),
        "values": values_plot(stats, largest_indices),
        "bias": bias_figure,
    }
    text = "\n\n".join([
        print_statistics(stats),
        largest_text,
        variance_statistics(stats, num_bins),
        bias_text,
        linear_approximation_statistics(pred_all, gt_all, num_quantiles),
    ])

    output_dir = Path(config.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.txt").write_text(text + "\n", encoding="utf-8")
    for name, figure in figures.items():
        save_figure(figure, output_dir, name)

    print(f"Wrote {output_dir / 'results.txt'}")


if __name__ == "__main__":
    main()
