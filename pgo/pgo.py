import torch
import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.append(os.getcwd())
sys.path.append("/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo")
[sys.path.append(i) for i in [".", ".."]]

from pose_graph_optimization.graph import *
from pose_graph_optimization.error_metrics import *
from pose_graph_optimization.utils import *
from pose_graph_optimization.loop_closure import detect_loop_closures
from src.utils.pose import (
    get_drift_metrics,
    get_ddf_metrics,
    get_global_and_relative_gt_trackings,
)
from pose_graph_optimization.image_registration import register


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=str
    )

    return parser.parse_args()


def main():

    args = parse_arguments()
    config = OmegaConf.load(args.config)

    OmegaConf.resolve(config)

    input_pred = config.dirs.input_pred
    input_gt = config.dirs.input_gt
    output_dir = config.dirs.output_dir

    if not input_pred:
        raise ValueError("dirs.input_pred must be specified in the config file.")
    
    if not input_gt:
        raise ValueError("dirs.input_gt must be specified in the config file.")

    if not output_dir:
        raise ValueError("dirs.output_dir must be specified in the config file.")

    drift_metrics_original = []
    drift_metrics_after_pgo = []

    ddf_metrics_original = []
    ddf_metrics_after_pgo = []

    for el in os.listdir(input_pred):

        sweep_path = os.path.join(input_pred, el, "export.h5")

        if not os.path.isfile(sweep_path):
            continue

        with h5py.File(sweep_path, "r") as f:

            # load data
            pred_acc = np.array(f["pred_tracking"])

            if input_gt:

                gt_file = os.path.join(input_gt, f"{el}.h5")

                with h5py.File(gt_file, "r") as f_gt:
                    gt = np.array(f_gt["tracking"])
                    gt_acc, gt_inbetween = get_global_and_relative_gt_trackings(gt)

            else:

                gt_acc = np.array(f["gt_tracking"])
                gt_inbetween = compute_inbetween_transforms(gt_acc)

            # load auxiliary data
            calibration_matrix = np.round(np.array(f["pixel_to_image"]), 4)
            fvs = np.array(f["fvs"])
            frames = np.array(f["images"])
            dimensions = np.array(f["dimensions"])

        image_shape_hw = tuple(dimensions[:2])

        # reformat
        pred_inbetween = compute_inbetween_transforms(pred_acc)

        pred_acc_torch = torch.from_numpy(pred_acc).float()
        gt_acc_torch = torch.from_numpy(gt_acc).float()

        pred_inbetween_torch = torch.from_numpy(pred_inbetween).float()
        gt_inbetween_torch = torch.from_numpy(gt_inbetween).float()

        # build graph
        pred_graph = PoseGraph(
            poses=pred_acc_torch,
            constraints=pred_inbetween_torch,
            initial_pose=pred_acc_torch[0],
        )

        # loop closure constraints
        if "loop_closure" in config:

            loop_closures = detect_loop_closures(
                feature_vectors=fvs,
                frames=frames,
                transforms=pred_inbetween_torch,
                pixel_to_image=calibration_matrix,
                method=config.loop_closure.method,
                stepsize=config.loop_closure.stepsize,
                temporal_offset=config.loop_closure.temporal_offset,
                threshold=config.loop_closure.threshold,
                n_neighbors=config.loop_closure.n_neighbors
            )

            for lc in loop_closures:

                i = lc["source_idx"]
                j = lc["target_idx"]
                transform = lc["transform"]
                score = lc["combined_score"]

                pred_graph.add_constraint(
                    i,
                    j,
                    transform,
                    registration_noise_model(confidence=score, ref_sigma=config.loop_closure.ref_values_sigma),
                )

        if "image_registration" in config:
            # Implement image registration logic here
            pass

        if "optical_flow" in config:
            # Implement optical flow logic here
            pass

        # optimize graph
        pred_graph_gtsam, _, pred_optimized = pred_graph.build_graph()

        optimized_pred = gtsam_values_to_torch(pred_optimized).numpy()

        # drift metrics
        drift_metrics_pred_vs_gt = get_drift_metrics(
            gt_acc_torch.numpy(),
            pred_acc_torch.numpy(),
        )

        drift_metrics_optimized_vs_gt = get_drift_metrics(
            gt_acc_torch.numpy(),
            optimized_pred,
        )

        drift_metrics_original.append(drift_metrics_pred_vs_gt)
        drift_metrics_after_pgo.append(drift_metrics_optimized_vs_gt)

        # ddf metrics
        ddf_metrics_pred_vs_gt = get_ddf_metrics(
            pred_acc_torch,
            pred_inbetween_torch,
            gt_acc_torch,
            gt_inbetween_torch,
            calibration_matrix,
            image_shape_hw,
            mode="5pt-landmark",
        )

        ddf_metrics_optimized_vs_gt = get_ddf_metrics(
            optimized_pred,
            compute_inbetween_transforms(optimized_pred),
            gt_acc_torch,
            gt_inbetween_torch,
            calibration_matrix,
            image_shape_hw,
            mode="5pt-landmark",
        )

        ddf_metrics_original.append(ddf_metrics_pred_vs_gt)
        ddf_metrics_after_pgo.append(ddf_metrics_optimized_vs_gt)

    print("\nAvg drift metrics (initial pred vs optimized pred):\n")
    print_avg_metrics([drift_metrics_original, drift_metrics_after_pgo]
    )

    print("\nAvg DDF metrics (initial pred vs optimized pred):\n")
    print_avg_metrics([ddf_metrics_original, ddf_metrics_after_pgo])

    save_results(
        output_dir=output_dir,
        graph=pred_graph_gtsam,
        initial=pred_acc_torch,
        optimized=optimized_pred,
        metrics_original=[drift_metrics_original, ddf_metrics_original],
        metrics_after_pgo=[drift_metrics_after_pgo, ddf_metrics_after_pgo]
    )


if __name__ == "__main__":
    main()
