import os
from os import path
import pandas as pd
import h5py
from .utils import mat4_to_pose3
import numpy as np


def pose_error(T_gt, T_pred):

    delta = T_gt.between(T_pred)
    t_err = delta.translation()
    r_err = delta.rotation().matrix()

    return t_err, r_err


def avg_trajectory_error(transforms_1, transforms_2):

    if len(transforms_1) != len(transforms_2):
        raise ValueError("Inputs must have the same length")

    avg_t_err, avg_r_err = np.zeros(3), np.zeros(shape=(3, 3))

    for el in zip(transforms_1, transforms_2):
        T_gt = mat4_to_pose3(el[0])
        T_pred = mat4_to_pose3(el[1])
        t_err, r_err = pose_error(T_gt, T_pred)
        avg_t_err += t_err
        avg_r_err += r_err
    avg_t_err /= len(transforms_1)
    avg_r_err /= len(transforms_1)

    return avg_t_err, avg_r_err


def save_results(
    output_dir,
    graph,
    initial,
    optimized,
    metrics_original=[],
    metrics_after_pgo=[],
):
    ## metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(
        output_dir,
        "metrics.txt",
    )

    with open(metrics_path, "w") as f:

        f.write(
                "initial:\n\n"
            )

        for metrics in metrics_original:
            metrics_df = pd.DataFrame(metrics).mean()

            for key, value in metrics_df.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write(
                "after pgo:\n\n"
            )

        for metrics in metrics_after_pgo:
            metrics_df = pd.DataFrame(metrics).mean()

            for key, value in metrics_df.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    ## graph
    graph_path = os.path.join(
        output_dir,
        "graph.h5",
    )

    with h5py.File(graph_path, "w") as f:

        graph_group = f.create_group("graph")
        graph_group.attrs["num_factors"] = graph.size()

        f.create_dataset(
            "initial",
            data=np.asarray(initial),
            compression="gzip"
        )

        f.create_dataset(
            "optimized",
            data=np.asarray(optimized),
            compression="gzip"
        )

    print(f"Saved results to {output_dir}")

def print_avg_metrics(metrics_list):

        for metrics in metrics_list:

            avg_metrics_df = pd.DataFrame(metrics).mean()

            for key, value in avg_metrics_df.items():
                print(f"  {key}: {value:.4f}")
            print("\n")