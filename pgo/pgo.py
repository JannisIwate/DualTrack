import torch
import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from itertools import islice

sys.path.append(os.getcwd())
sys.path.append("/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo")
[sys.path.append(i) for i in [".", ".."]]

from pose_graph_optimization.graph import *
from pose_graph_optimization.error_metrics import *
from pose_graph_optimization.utils import *
from pose_graph_optimization.loop_closure import detect_loop_closures
from pose_graph_optimization.image_registration import sample_random_pairs, sample_pairs_by_step, register
from src.utils.pose import (
    get_drift_metrics,
    get_ddf_metrics,
    get_global_and_relative_gt_trackings,
)


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

    data = os.listdir(input_pred)

    nr_of_scans = OmegaConf.select(config, "general.nr_scans")
    data = islice(data, nr_of_scans) if nr_of_scans is not None else data

    for el in tqdm(data, desc="Working", total=nr_of_scans):

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
            initial_pose=pred_acc_torch[0]
        )

        # loop closure constraints
        if "loop_closure" in config:

            loop_closures = detect_loop_closures(
                feature_vectors=fvs,
                frames=frames,
                transforms=pred_inbetween_torch,
                **config.loop_closure
            )

            # print(f"nr of valid LCs found: {len(loop_closures)}")

            for lc in loop_closures:

                i = lc["source_idx"]
                j = lc["target_idx"]
                transform = lc["transform"]
                score = lc["combined_score"]

                pred_graph.add_constraint(
                    i,
                    j,
                    transform,
                    registration_noise_model(confidence=score, ref_sigma=config.general.ref_values_sigma)
                )

        # IR constraints
        if "image_registration" in config:

            STEP = 100
            idc1, idc2, frames_1, frames_2 = sample_pairs_by_step(frames, STEP)
            nr_valid_irs = 0

            for i, _ in enumerate(idc1):

                T, confidence, valid = register(frame_i=frames_1[i],
                                                frame_j=frames_2[i],
                                                transforms=pred_inbetween[i:STEP-1],
                                                **config.image_registration
                                                )
                
                # print(f"confidence: {confidence}")
                # print(f"transform model: {pred_inbetween_torch[i]}")
                # print(f"transform IR: {T}")

                if valid:
                    
                    print(f"valid confidence: {confidence}")
        
                    pred_graph.add_constraint(
                        idc1[i],
                        idc2[i],
                        T,
                        registration_noise_model(confidence=confidence, ref_sigma=config.general.ref_values_sigma)
                    )
                    nr_valid_irs = nr_valid_irs + 1
            
            # print(f"percentage of valid IRs: {(nr_valid_irs / len(idc1)) * 100}%")

        if "optical_flow" in config:
            # Implement optical flow logic here
            pass

        if "trajectory_smoothing":
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

        #break

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
