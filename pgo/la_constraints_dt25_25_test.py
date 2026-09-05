"""Grid-search LA constraint parameters with PGO for TUSREC25 / DualTrack25.

This evaluates the same setup as ``pgo.py`` with ``pgo.options`` containing
``noise_constraints``. For every quantile/scale pair, LA transforms are added
as extra pose-graph constraints and the graph is optimized for every scan.

Example:
    python pgo/pose_graph_optimization/la_constraints_dt25_25_test.py \
        -c pgo/experiments/la_replace_pgo/dt25/tusrec25/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
PGO_ROOT = REPO_ROOT / "pgo"
sys.path.insert(0, str(PGO_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from pgo import (  # noqa: E402
    cfg_require,
    get_ddf_metrics,
    get_drift_metrics,
    get_linear_approximation_profile,
    get_scan_list,
    linear_approximation,
    load_scan_data,
    extract_positions,
    plot_trajectories,
)
from pose_graph_optimization.graph import PoseGraph  # noqa: E402
from pose_graph_optimization.utils import (  # noqa: E402
    accumulated_to_inbetween,
    gtsam_to_numpy,
)


GRID_VALUES = tuple(round(index / 10, 1) for index in range(11))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="CSV output path; defaults next to the config.",
    )
    return parser.parse_args()


def load_scans(config: object) -> list[dict]:
    input_pred = cfg_require(config, "dirs.input_pred")
    scan_names, _ = get_scan_list(config, input_pred)
    scans = []

    for scan_index, scan_name in enumerate(scan_names):
        scan = load_scan_data(input_pred, scan_name, scan_index, config)
        if scan is not None:
            scans.append(scan)

    if not scans:
        raise RuntimeError(f"No scans found below {input_pred}")

    return scans


def optimize_scan(
    scan: dict,
    slopes: list[float],
    intercepts: list[float],
    quantile: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred_acc = scan["pred_acc"]
    pred_inbetween = scan["pred_inbetween"]
    pred_graph = PoseGraph(
        poses=pred_acc,
        constraints=pred_inbetween,
        initial_pose=pred_acc[0],
    )

    pred_inbetween_la = linear_approximation(
        pred_inbetween,
        quantile,
        scale,
        slopes,
        intercepts,
    )

    for node_index, transform in enumerate(pred_inbetween_la[1:-1]):
        pred_graph.add_constraint(
            node_i=node_index,
            node_j=node_index + 1,
            transform=transform,
        )

    _, _, optimized = pred_graph.build_graph()
    optimized_acc = gtsam_to_numpy(optimized)
    optimized_inbetween = accumulated_to_inbetween(optimized_acc)
    return optimized_acc, optimized_inbetween


def evaluate_combination(
    scans: list[dict],
    slopes: list[float],
    intercepts: list[float],
    quantile: float,
    scale: float,
) -> dict[str, float]:
    drift_metrics = []
    ddf_metrics = []

    for scan in scans:
        optimized_acc, optimized_inbetween = optimize_scan(
            scan,
            slopes,
            intercepts,
            quantile,
            scale,
        )
        drift_metrics.append(
            get_drift_metrics(torch.as_tensor(scan["gt_acc"]), optimized_acc)
        )
        ddf_metrics.append(
            get_ddf_metrics(
                optimized_acc,
                optimized_inbetween,
                scan["gt_acc"],
                scan["gt_inbetween"],
                scan["calibration_matrix"],
                scan["image_shape_hw"],
                mode="5pt-landmark",
            )
        )

    return {
        "quantile": quantile,
        "scale": scale,
        "fdr": float(np.mean([metric["final_drift_rate"] for metric in drift_metrics])),
        "gpe": float(np.mean([metric["avg_global_displacement_error"] for metric in ddf_metrics])),
        "lpe": float(np.mean([metric["avg_local_displacement_error"] for metric in ddf_metrics])),
        "tusrec_final_score": float(np.mean([metric["tusrec_final_score"] for metric in ddf_metrics])),
    }


def write_results(output_path: Path, results: list[dict[str, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def save_best_plots(
    scans: list[dict],
    profile: tuple[list[float], list[float]],
    result: dict[str, float],
    metric_name: str,
    output_dir: Path,
) -> None:
    for scan_index, scan in enumerate(scans):
        optimized_acc, _ = optimize_scan(
            scan,
            *profile,
            result["quantile"],
            result["scale"],
        )
        trajectories = [
            extract_positions(scan["gt_acc"]),
            extract_positions(scan["pred_acc"]),
            extract_positions(optimized_acc),
        ]
        fig = plot_trajectories(
            trajectories,
            labels=["GT", "Initial estimated", "Optimized"],
            colors=["blue", "red", "black"],
            title=(
                f"Best LA configuration by {metric_name.upper()} | "
                f"quantile={result['quantile']:.1f}, scale={result['scale']:.1f}"
            ),
        )
        output_path = output_dir / f"la_best_{metric_name}_scan_{scan_index}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def main() -> None:
    args = parse_arguments()
    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)

    profile = get_linear_approximation_profile(config)
    scans = load_scans(config)
    output_path = args.output or args.config.parent / "la_constraints_grid_search.csv"

    print(f"Loaded {len(scans)} scans")
    print(f"Testing {len(GRID_VALUES) ** 2} quantile/scale combinations with PGO")
    start_time = time.perf_counter()
    results = [
        evaluate_combination(scans, *profile, quantile, scale)
        for quantile in GRID_VALUES
        for scale in GRID_VALUES
    ]
    write_results(output_path, results)

    print(f"Saved all results to {output_path}")
    output_dir = Path(__file__).resolve().parent
    for metric_name in ("fdr", "gpe", "lpe"):
        best = min(results, key=lambda result: result[metric_name])
        print(
            f"best {metric_name.upper()}: "
            f"quantile={best['quantile']:.1f}, "
            f"scale={best['scale']:.1f}, "
            f"value={best[metric_name]:.6f}"
        )
        save_best_plots(scans, profile, best, metric_name, output_dir)
    print(f"Elapsed: {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()
