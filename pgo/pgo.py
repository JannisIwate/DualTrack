import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from itertools import islice
from matplotlib import pyplot as plt
import time

sys.path.append(os.getcwd())
sys.path.append("/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo")
[sys.path.append(i) for i in [".", ".."]]

from pose_graph_optimization.graph import *
from pose_graph_optimization.error_metrics import *
from pose_graph_optimization.utils import *
from pose_graph_optimization.loop_closure import detect_loop_closures
from pose_graph_optimization.image_registration import sample_pairs_by_step, register
from src.utils.pose import get_drift_metrics, get_ddf_metrics, get_global_and_relative_gt_trackings, plot_pose_differences
from src.evaluator import plot_pose_differences


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

    ## init
    # arguments
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

    # variables
    drift_metrics_original = []
    drift_metrics_after_pgo = []

    ddf_metrics_original = []
    ddf_metrics_after_pgo = []

    # data
    data = os.listdir(input_pred)

    nr_of_scans = OmegaConf.select(config, "general.nr_scans")
    data = islice(data, nr_of_scans) if nr_of_scans is not None else data

    ## apply pgo to data scans
    for el in tqdm(data, desc="Working", total=nr_of_scans):

        # load file
        sweep_path = os.path.join(input_pred, el, "export.h5")

        if not os.path.isfile(sweep_path):
            continue

        with h5py.File(sweep_path, "r") as f:
            
            # load scan data
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

        # build graph
        # smoothing edges
        if "trajectory_smoothing" in config:
            
            # TODO: implement trajectory smoothing

            pred_graph = PoseGraph(
                poses=pred_acc,
                constraints=pred_inbetween,
                initial_pose=pred_acc[0],
                odom_noise_sigma=res["weights"][:-1]
            )
        else:
            pred_graph = PoseGraph(
                poses=pred_acc,
                constraints=pred_inbetween,
                initial_pose=pred_acc[0]
            )

        # LC constraints
        if "loop_closure" in config:

            loop_closures = detect_loop_closures(
                feature_vectors=fvs,
                frames=frames,
                transforms=pred_inbetween,
                image_registration_cfg = config.image_registration,
                **config.loop_closure
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
                    registration_noise_model(confidence=score, ref_sigma=config.general.ref_values_sigma)
                )

        # IR constraints
        if "image_registration" in config:

            STEP = 100 # register every STEP frames
            idc1, idc2, ir_ref_transforms, ir_gt_transforms = sample_pairs_by_step(frames, pred_acc, gt_acc, STEP)
            nr_valid_irs = 0

            metric_before_list = []
            metric_before_gt_list = []
            metric_before_pred_list = []
            metric_after_list = []
            ir_execution_time_list = []
            ir_transforms = []

            for i, _ in enumerate(frames[idc1]):

                start_time = time.time() # measure IR execution time
                (transform_ir,
                 confidence,
                 valid,
                 metric_before_identity,
                 metric_before_gt,
                 metric_before_pred,
                 metric_after
                ) = register(frame_i=frames[idc1][i],
                            frame_j=frames[idc2][i],
                            ref_transform=ir_ref_transforms[i],
                            gt_transform=ir_gt_transforms[i],
                            **config.image_registration
                            )
                ir_execution_time = time.time() - start_time
                
                metric_before_list.append(metric_before_identity)
                metric_before_gt_list.append(metric_before_gt)
                metric_before_pred_list.append(metric_before_pred)
                metric_after_list.append(metric_after)
                ir_execution_time_list.append(ir_execution_time)
                ir_transforms.append(transform_ir)

                if valid:
                    pred_graph.add_constraint(
                        idc1[i],
                        idc2[i],
                        transform_ir,
                        registration_noise_model(confidence=confidence, ref_sigma=config.general.ref_values_sigma)
                    )
                    nr_valid_irs = nr_valid_irs + 1
                #break
            
            # print(f"percentage of valid IRs: {(nr_valid_irs / len(idc1)) * 100}%")
            print(f"avg ir metric before: {np.average(np.array(metric_before_list))}")
            print(f"avg ir metric before (gt): {np.average(np.array(metric_before_gt_list))}")
            print(f"avg ir metric before (pred): {np.average(np.array(metric_before_pred_list))}")
            print(f"avg ir metric after: {np.average(np.array(metric_after_list))}")
            print(f"avg ir execution time (s): {np.average(np.array(ir_execution_time_list))}")
            
            # plot_pose_differences(ir_gt_transforms, ir_ref_transforms)
            # plot_pose_differences(ir_gt_transforms, ir_transforms)
            # plt.show()
            #breakpoint()

        if "optical_flow" in config:
            # Implement optical flow logic here
            pass
            
        # optimize graph
        pred_graph_gtsam, _, pred_optimized = pred_graph.build_graph()

        optimized_pred = gtsam_to_numpy(pred_optimized)

        # drift metrics
        drift_metrics_pred_vs_gt = get_drift_metrics(
            gt_acc,
            pred_acc,
        )

        drift_metrics_optimized_vs_gt = get_drift_metrics(
            gt_acc,
            optimized_pred,
        )

        drift_metrics_original.append(drift_metrics_pred_vs_gt)
        drift_metrics_after_pgo.append(drift_metrics_optimized_vs_gt)

        # ddf metrics
        ddf_metrics_pred_vs_gt = get_ddf_metrics(
            pred_acc,
            pred_inbetween,
            gt_acc,
            gt_inbetween,
            calibration_matrix,
            image_shape_hw,
            mode="5pt-landmark",
        )

        ddf_metrics_optimized_vs_gt = get_ddf_metrics(
            optimized_pred,
            compute_inbetween_transforms(optimized_pred),
            gt_acc,
            gt_inbetween,
            calibration_matrix,
            image_shape_hw,
            mode="5pt-landmark",
        )

        ddf_metrics_original.append(ddf_metrics_pred_vs_gt)
        ddf_metrics_after_pgo.append(ddf_metrics_optimized_vs_gt)

        #break

    # process results
    print("\nAvg drift metrics (initial pred vs optimized pred):\n")
    print_avg_metrics([drift_metrics_original, drift_metrics_after_pgo]
    )

    print("\nAvg DDF metrics (initial pred vs optimized pred):\n")
    print_avg_metrics([ddf_metrics_original, ddf_metrics_after_pgo])

    save_results(
        output_dir=output_dir,
        graph=pred_graph_gtsam,
        initial=pred_acc,
        optimized=optimized_pred,
        metrics_original=[drift_metrics_original, ddf_metrics_original],
        metrics_after_pgo=[drift_metrics_after_pgo, ddf_metrics_after_pgo]
    )

    # fig = plot_pose_differences(optimized_pred, gt_acc)
    # plt.show()
    # plot_trajectories([extract_positions(inbetween_to_accumulated(np.array(ir_gt_transforms))),
    #                    extract_positions(inbetween_to_accumulated(np.array(ir_ref_transforms))),
    #                    extract_positions(inbetween_to_accumulated(np.array(ir_transforms)))],
    #                     labels=["GT", "Initial estimated", "IR"],
    #                     colors=["blue", "red", "black"])


def plot_trajectories(trajectories, labels=None, colors=None):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = len(trajectories)

    # defaults
    if labels is None:
        labels = [f"traj_{i}" for i in range(n)]
    if colors is None:
        colors = [None] * n

    for i, (xs, ys, zs) in enumerate(trajectories):
        ax.plot(xs, ys, zs,
                label=labels[i],
                color=colors[i])

        ax.scatter(xs[0], ys[0], zs[0], color=colors[i])
        ax.scatter(xs[-1], ys[-1], zs[-1], color=colors[i], marker="s")

    ax.set_title("Pose Graph Trajectories")
    ax.legend()
    plt.show()


def extract_positions(values:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    xs, ys, zs = [], [], []

    for el in values:

        xs.append(el[0, 3])
        ys.append(el[1, 3])
        zs.append(el[2, 3])

    return np.array(xs), np.array(ys), np.array(zs)


if __name__ == "__main__":
    main()
