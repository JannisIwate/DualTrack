import os
import shutil
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .utils import mat4_to_pose3


def pose_error(T_gt: Any, T_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    
    delta = T_gt.between(T_pred)

    t_err = np.asarray(delta.translation())
    r_err = np.asarray(delta.rotation().matrix())

    return t_err, r_err


def avg_trajectory_error(
    transforms_1: Sequence[np.ndarray],
    transforms_2: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    
    if len(transforms_1) != len(transforms_2):
        raise ValueError("Inputs must have the same length")

    avg_t_err = np.zeros(3, dtype=float)
    avg_r_err = np.zeros((3, 3), dtype=float)

    for T1, T2 in zip(transforms_1, transforms_2):
        T_gt = mat4_to_pose3(T1)
        T_pred = mat4_to_pose3(T2)

        t_err, r_err = pose_error(T_gt, T_pred)

        avg_t_err += t_err
        avg_r_err += r_err

    avg_t_err /= len(transforms_1)
    avg_r_err /= len(transforms_1)

    return avg_t_err, avg_r_err


def save_results(
    output_dir: str,
    initial: Sequence[np.ndarray],
    optimized: Sequence[np.ndarray],
    number_of_scans: int = 0,
    number_of_failed_scans: int = 0,
    graph: Any = None,
    metrics_original: Sequence[dict[str, float]] | None = None,
    metrics_after_pgo: Sequence[dict[str, float]] | None = None,
    ir_metrics: dict[str, Sequence[float]] | None = None,
    la_metrics: Sequence[dict[str, float]] | None = None,
    loop_closure_metrics: Sequence[dict[str, float]] | None = None,
    figs_individual: dict | None = None,
    figs_general: dict | None = None,
) -> None:

    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError:
        print("Directory not found, creating new.")

    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics.txt")

    with open(metrics_path, "w") as f:

        f.write("initial:\n\n")
        f.write(f"  number of scans: {number_of_scans}\n")
        f.write(f"  number of failed scans: {number_of_failed_scans}\n\n")

        if len(metrics_original[0]) > 0:

            for metrics in metrics_original:
                metrics_df = pd.DataFrame(metrics).mean()

                for key, value in metrics_df.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

        if len(metrics_after_pgo[0]) > 0:

            f.write("after pgo:\n\n")

            for metrics in metrics_after_pgo:
                metrics_df = pd.DataFrame(metrics).mean()

                for key, value in metrics_df.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

        if len(ir_metrics[0]) > 0:

            f.write("image registration:\n\n")
            # print(ir_metrics)
            
            for metrics in ir_metrics:

                if len(metrics) > 0:

                    if isinstance(metrics, dict):
                                                        
                        max_len = max((len(v) for v in metrics.values()))
                        metrics = {
                            k: v + [np.nan] * (max_len - len(v)) if isinstance(v, list) else v
                            for k, v in metrics.items()
                        }

                    ir_df = pd.DataFrame(metrics)
                    numeric_cols = ir_df.select_dtypes(include="number").columns
                    non_numeric_cols = ir_df.columns.difference(numeric_cols)

                    for col in non_numeric_cols:          # z.B. "metric": "mi" -> einmalig

                        f.write(f"  {col}: {ir_df[col].iloc[0]}\n")

                    for key, value in ir_df[numeric_cols].mean().items():   # ALLE numerischen Spalten mitteln
                        
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
            # breakpoint()
                    
        if la_metrics:

            f.write("linear approximation:\n\n")
            la_df = pd.DataFrame(la_metrics).mean()
            for key, value in la_df.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        if loop_closure_metrics:

            f.write("loop closure:\n\n")
            lc_df = pd.DataFrame(loop_closure_metrics).mean()
            for key, value in lc_df.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    if graph is not None and initial is not None and optimized is not None:
        graph_path = os.path.join(output_dir, "graph.h5")

        with h5py.File(graph_path, "w") as f:
            graph_group = f.create_group("graph")
            graph_group.attrs["num_factors"] = graph.size()

            f.create_dataset(
                "initial",
                data=np.asarray(initial),
                compression="gzip",
            )

            f.create_dataset(
                "optimized",
                data=np.asarray(optimized),
                compression="gzip",
            )

    if figs_general:

        for fig_name, fig in figs_general.items():

            fig.savefig(os.path.join(output_dir, fig_name))

    if figs_individual:

        for sweep_name, fig_collection in figs_individual.items():

            figs_dir = os.path.join(output_dir, sweep_name)
            os.makedirs(figs_dir, exist_ok=True)

            for fig_name, fig in fig_collection.items():

                fig.savefig(os.path.join(figs_dir, fig_name))

def print_avg_metrics(metrics_list: Sequence[dict[str, float]]) -> None:

    for metrics in metrics_list:
        avg_metrics_df = pd.DataFrame(metrics).mean()

        for key, value in avg_metrics_df.items():
            print(f"  {key}: {value:.4f}")