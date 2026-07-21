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
from scipy.stats import norm

sys.path.append(os.getcwd())
sys.path.append("/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo")
[sys.path.append(i) for i in [".", ".."]]

from pose_graph_optimization.graph import *
from pose_graph_optimization.error_metrics import *
from pose_graph_optimization.utils import *
from pose_graph_optimization.loop_closure import detect_loop_closures
from pose_graph_optimization.image_registration import sample_pairs_by_step, register
from src.utils.pose import get_drift_metrics, get_ddf_metrics, get_global_and_relative_gt_trackings, plot_pose_differences, pose_vector_to_matrix, matrix_to_pose_vector
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

    # --------------------------------------------------------
    # init
    # --------------------------------------------------------

    def cfg_has(path: str) -> bool:
        return OmegaConf.select(config, path) is not None


    def cfg_require(path: str):
        value = OmegaConf.select(config, path)
        if value is None:
            raise ValueError(f"Missing required config entry: {path}")
        return value
    

    def cfg_get(name:str):
        return OmegaConf.select(config, name) if cfg_has(name) else None
    

    # arguments
    args = parse_arguments()
    config = OmegaConf.load(args.config)

    OmegaConf.resolve(config)

    input_pred = cfg_require("dirs.input_pred")
    # if OmegaConf.select(config, ) is None:
    #     raise ValueError("Missing required config entry: dirs.input_pred")
    # input_pred = config.dirs.input_pred

    execute_pgo = False
    noise_constraints = False

    if cfg_has("general.options") and "pgo" in cfg_get("general.options"):
        execute_pgo = True
    if cfg_has("general.options") and "noise_constraints" in cfg_get("general.options"):
        noise_constraints = True
    # if OmegaConf.select(config, "general.pgo") is not None and \
    #        OmegaConf.select(config, "general.pgo"):
    #     execute_pgo = True
    # else:
    #     execute_pgo = False

    # variables
    drift_metrics_original = []
    drift_metrics_after_pgo = []
    drift_metrics_after_ir = []

    ddf_metrics_original = []
    ddf_metrics_after_pgo = []
    ddf_metrics_after_ir = []

    ir_transforms_all = {}
    ir_metrics_all = {}

    figs_individual = {}
    figs_general = {}

    counter = 0

    # data
    data = os.listdir(input_pred)
    nr_of_scans = cfg_get("general.nr_scans")
    start = cfg_get("general.start_scan")
    if start == None:
        start = 1
    data = islice(data, start-1, start+nr_of_scans-1) if nr_of_scans is not None else data       

    # --------------------------------------------------------
    # PGO
    # --------------------------------------------------------

    for i, el in enumerate(tqdm(data, desc="Working", total=nr_of_scans)):

        ## load data
        # load file
        sweep_path = os.path.join(input_pred, el, "export.h5")
        sweep_name = f"sweep_{i}"

        if not os.path.isfile(sweep_path):
            continue

        with h5py.File(sweep_path, "r") as f:

            nr_of_frames= cfg_get("general.nr_frames")
            if nr_of_frames is None:
                nr_of_frames = len(f["images"])
            else:
                nr_of_frames += 1
            
            # load scan data
            pred_acc = np.array(f["pred_tracking_glob"][:nr_of_frames]) # starts with identity, normalized acc world coords
            pred_inbetween = np.array(f["pred_tracking_loc"][:nr_of_frames])

            if cfg_has("dirs.input_gt"):
                gt_file = os.path.join(config.dirs.input_gt, f"{el}.h5")

                with h5py.File(gt_file, "r") as f_gt:
                    gt = np.array(f_gt["tracking"][:nr_of_frames]) # acc gt poses in arbitrary world coords
                    gt_acc, gt_inbetween = get_global_and_relative_gt_trackings(gt) # get normalized acc gt (first pose is identity) and relative gt
                    # first transform of gt_acc is identity (or rather almost due to numerics)
                    # first transform of gt_inbetween is identity as first frame is first frame
                    # gt_inbetween is Ti->j, forward
            else:
                gt_acc = np.array(f["gt_tracking"][:nr_of_frames]) # same here as above, transforms already normalized
                gt_inbetween = compute_inbetween_transforms(gt_acc)

            # load auxiliary data
            calibration_matrix = np.round(np.array(f["pixel_to_image"]), 4)
            fvs = np.array(f["fvs"])
            frames = np.array(f["images"][:nr_of_frames])
            dimensions = np.array(f["dimensions"])

            # from PIL import Image
            # for i in range(0, 30, 10):
            #     np.save(f"frame_{i}", frames[i])
            #     np.save(f"ref_transform_{i}", pred_inbetween[i+1])
            #     np.save(f"gt_transform_{i}", gt_inbetween[i+1])

            #     img = Image.fromarray(frames[i])
            #     img.save(f"image_{i}.jpg")
            # breakpoint()

        image_shape_hw = tuple(dimensions[:2])
        image_shape_hw = (image_shape_hw[1], image_shape_hw[0])

        ## build graph
        if execute_pgo:
            pred_graph = PoseGraph(
                poses=pred_acc,
                constraints=pred_inbetween,
                initial_pose=pred_acc[0]
            )

            # LC constraints
            if cfg_has("loop_closure"):

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
            
            # OF
            if cfg_has("optical_flow"):
                # Implement optical flow logic here
                pass

        # IR (constraints or just IR)
        if cfg_has("image_registration"):

            ir_metrics_all = {
                "metric": str,
                "metric_before": [],
                "metric_before_gt": [],
                "metric_before_pred": [],
                "metric_after": [],
                "ir_execution_time": [],
            }

            ir_transforms_all = {
                "ir_transforms": [],
                "ir_gt_transforms": [],
                "ir_ref_transforms": [],
            }

            # get transforms for non-adjacent frames (STEP > 1)
            STEP = cfg_get("image_registration.step")
            if STEP == None:
                STEP = 1
            idc1, idc2, ir_ref_transforms_inbetween, ir_gt_transforms_inbetween = sample_pairs_by_step(frames, pred_acc, gt_acc, STEP) # first element is identity
            nr_valid_irs = 0

            ir_frames_1 = frames[idc1]
            ir_frames_2 = frames[idc2]
            ir_transforms_inbetween = np.tile(np.eye(4), (ir_ref_transforms_inbetween.shape[0], 1, 1))

            for i, _ in enumerate(idc1):

                start_time = time.time() # measure IR execution time
                (transform_ir,
                 confidence,
                 valid,
                 metric_before_identity,
                 metric_before_gt,
                 metric_before_pred,
                 metric_after
                ) = register(frame_i=ir_frames_1[i],
                            frame_j=ir_frames_2[i],
                            ref_transform=ir_ref_transforms_inbetween[i+1], # skip first identity element
                            gt_transform=ir_gt_transforms_inbetween[i+1],
                            **config.image_registration
                            )
                ir_execution_time = time.time() - start_time
                
                ir_metrics_all["metric"] = config.image_registration.sitk.metric # single value is added, individual scans still distinguishable
                ir_metrics_all["metric_before"].append(metric_before_identity)
                ir_metrics_all["metric_before_gt"].append(metric_before_gt)
                ir_metrics_all["metric_before_pred"].append(metric_before_pred)
                ir_metrics_all["metric_after"].append(metric_after)
                ir_metrics_all["ir_execution_time"].append(ir_execution_time)
                ir_transforms_inbetween[i+1] = transform_ir # skip first identity element
                
                # if valid and execute_pgo:
                if execute_pgo:
                    
                    ref_sigma = cfg_get("general.ref_values_sigma")
                    if ref_sigma == None:
                        ref_sigma = 1e-2

                    stride = cfg_get("general.counter")
                    if stride == None:
                        stride = 1

                    counter += 1
                    
                    if counter % stride == 0:
                        pred_graph.add_constraint(
                            idc1[i],
                            idc2[i],
                            transform_ir,
                            registration_noise_model(confidence=confidence, ref_sigma=ref_sigma)
                        )
                    nr_valid_irs = nr_valid_irs + 1
            
            # create plots
            plot_cfg = cfg_get("plot")

            if plot_cfg is not None:

                if "plot_ir_pose_differences" in plot_cfg:
                    
                    figs_individual[f"{sweep_name}"] = {} # populate sweep key with sub dict first
                    figs_individual[f"{sweep_name}"]["ir_pose_diffs_ref_gt"] = plot_pose_differences(ir_ref_transforms_inbetween, ir_gt_transforms_inbetween, title="GT vs Pred") # gt is blue -> general direction is fine
                    plt.close()
                    figs_individual[f"{sweep_name}"]["ir_pose_diffs_ir_gt"] = plot_pose_differences(ir_transforms_inbetween, ir_gt_transforms_inbetween, title="GT vs IR") # -> general direction is fine but sometimes very big errors
                    plt.close()

                if "plot_ir_trajectories" in plot_cfg:

                    figs_individual[f"{sweep_name}"]["ir_trajectories_gt_ref_ir"] = plot_trajectories([extract_positions(inbetween_to_accumulated(ir_gt_transforms_inbetween[1:])), # skip first identity element
                                                                                                        extract_positions(inbetween_to_accumulated(ir_ref_transforms_inbetween[1:])),
                                                                                                        extract_positions(inbetween_to_accumulated(ir_transforms_inbetween[1:]))],
                                                                                                        labels=["GT", "Initial estimated", "IR"],
                                                                                                        colors=["blue", "red", "black"])
                    # plt.show()
                    # breakpoint()
                    plt.close()

                # STEP = 1, ein Scan
                # -> IR passt von den Richtungen her einigermassen, allerdings ist gibt es viel mehr Drift
                # -> Fehler bei beiden und allen 6DoF sind weitgehend mittelwertfrei
                # -> GT Trajektorie "zittert", IR auch, pred ist glatter (vor allem Winkel)
            
            # store results
            ir_transforms_all["ir_transforms"].append(ir_transforms_inbetween[1:]) # ignore first identity transform
            ir_transforms_all["ir_gt_transforms"].append(ir_gt_transforms_inbetween[1:])
            ir_transforms_all["ir_ref_transforms"].append(ir_ref_transforms_inbetween[1:])
            ir_transforms_acc = inbetween_to_accumulated(ir_transforms_inbetween[1:])
            ir_gt_transforms_acc = inbetween_to_accumulated(ir_gt_transforms_inbetween[1:])

            drift_metrics_ir_vs_gt = get_drift_metrics(
                ir_gt_transforms_acc,
                ir_transforms_acc,
            )
            drift_metrics_after_ir.append(drift_metrics_ir_vs_gt)

            ddf_metrics_ir = get_ddf_metrics(
                ir_transforms_acc,
                ir_transforms_inbetween,
                ir_gt_transforms_acc,
                ir_gt_transforms_inbetween,
                calibration_matrix,
                image_shape_hw,
                mode="5pt-landmark",
            )
            ddf_metrics_after_ir.append(ddf_metrics_ir)

        # noise contraints
        if noise_constraints:

            mean = 0.0 # empirical values
            STD = np.array([
                0.07656708383374906,
                0.02681774961755883,
                0.09021679400248785,
                0.04206508373865844,
                0.033058318883716194,
                0.03825069374999717,
            ])
            SLOPE = np.array([
                -0.084166,
                -0.343315,
                -0.155954,
                -0.765020,
                -0.671620,
                -0.612082,
            ])
            INTERCEPT = np.array([                -0.002105,
                -0.000186,
                0.000503,
                0.001229,
                -0.002765,
                0.001885,
            ])

            lower = norm.ppf(0.05, loc=mean, scale=STD) # filter really small and really big values 
            upper = norm.ppf(0.95, loc=mean, scale=STD)

            for i in range(1, pred_inbetween.shape[0] - 1, 1): # shape (X, 4, 4)

                # noise_vector = np.random.normal(mean, STD)
                # noise_vector = np.clip(noise_vector, lower, upper)
                pred_inbetween_vector = matrix_to_pose_vector(pred_inbetween[i])
                # vector = pred_inbetween_vector + noise_vector

                # pred_vector_corrected = 0.3*(pred_inbetween_vector - INTERCEPT) / (1.0 + SLOPE)
                alpha = 1.0

                expected_error = SLOPE * pred_inbetween_vector + INTERCEPT
                pred_vector_corrected = pred_inbetween_vector - alpha * expected_error

                transform = pose_vector_to_matrix(pred_vector_corrected)

                pred_graph.add_constraint(
                            node_i=i,
                            node_j=i + 1,
                            transform=transform
                        )
            
        ## optimize graph
        pred_graph_gtsam = None
        optimized_pred = None

        if execute_pgo:
            
            pred_graph_gtsam, _, pred_optimized = pred_graph.build_graph()
            optimized_pred = gtsam_to_numpy(pred_optimized)

        ## metrics
        # drift metrics
        gt_acc = torch.tensor(gt_acc) # to match 
        drift_metrics_pred_vs_gt = get_drift_metrics(
            gt_acc,
            pred_acc,
        )
        drift_metrics_original.append(drift_metrics_pred_vs_gt)

        if execute_pgo:

            drift_metrics_optimized_vs_gt = get_drift_metrics(
                gt_acc,
                optimized_pred,
            )
            drift_metrics_after_pgo.append(drift_metrics_optimized_vs_gt)

        # ddf metrics
        ddf_metrics_pred_vs_gt = get_ddf_metrics(
            pred_acc,
            pred_inbetween,
            gt_acc,
            gt_inbetween,
            torch.tensor(calibration_matrix),
            image_shape_hw,
            mode="5pt-landmark",
        )
        ddf_metrics_original.append(ddf_metrics_pred_vs_gt)

        if execute_pgo:

            ddf_metrics_optimized_vs_gt = get_ddf_metrics(
                optimized_pred,
                compute_inbetween_transforms(optimized_pred),
                gt_acc,
                gt_inbetween,
                calibration_matrix,
                image_shape_hw,
                mode="5pt-landmark",
            )
            ddf_metrics_after_pgo.append(ddf_metrics_optimized_vs_gt)

        #break

    # --------------------------------------------------------
    # process results
    # --------------------------------------------------------    
    if cfg_has("image_registration") and \
        cfg_has("plot") and \
        "plot_ir_error_magnitudes" in cfg_get("plot"):

        figs_general["ir_error_mags_ref_gt"] = plot_motion_vs_error(
            np.concatenate(ir_transforms_all["ir_ref_transforms"], axis=0), # transform to np array and combine
            np.concatenate(ir_transforms_all["ir_gt_transforms"], axis=0),
            title="Ref vs GT"
        )
        plt.close()

        figs_general["ir_error_mags_ir_gt"] = plot_motion_vs_error(
            np.concatenate(ir_transforms_all["ir_transforms"], axis=0), # transform to np array and combine
            np.concatenate(ir_transforms_all["ir_gt_transforms"], axis=0),
            title="IR vs GT"
        )
        plt.close()

    if cfg_has("dirs.output_dir"):

        save_results(
            output_dir=f"{config.dirs.output_dir}/results",
            graph=pred_graph_gtsam,
            initial=pred_acc,
            optimized=optimized_pred,
            metrics_original=[drift_metrics_original, ddf_metrics_original],
            metrics_after_pgo=[drift_metrics_after_pgo, ddf_metrics_after_pgo],
            ir_metrics=[ir_metrics_all, drift_metrics_after_ir, ddf_metrics_after_ir],
            figs_individual=figs_individual,
            figs_general=figs_general
        )

    # plt.show()


def largest_pose_errors(est: np.ndarray, gt: np.ndarray, n: int = 5) -> tuple[
                                                                            np.ndarray,
                                                                            np.ndarray,
                                                                            np.ndarray,
                                                                            np.ndarray,
                                                                            np.ndarray,
                                                                            np.ndarray,
                                                                        ]:

    if est.shape != gt.shape:
        raise ValueError("est and gt must have the same shape.")

    def rotation_to_angles(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        pitch = np.arctan2(
            -R[:, 2, 0],
            np.sqrt(R[:, 2, 1] ** 2 + R[:, 2, 2] ** 2),
        )

        roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])

        return roll, pitch, yaw

    def angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:

        return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

    # Translation errors
    x_error = np.abs(est[:, 0, 3] - gt[:, 0, 3])
    y_error = np.abs(est[:, 1, 3] - gt[:, 1, 3])
    z_error = np.abs(est[:, 2, 3] - gt[:, 2, 3])

    # Rotation errors
    roll_est, pitch_est, yaw_est = rotation_to_angles(est[:, :3, :3])
    roll_gt, pitch_gt, yaw_gt = rotation_to_angles(gt[:, :3, :3])

    roll_error = angle_error(roll_est, roll_gt)
    pitch_error = angle_error(pitch_est, pitch_gt)
    yaw_error = angle_error(yaw_est, yaw_gt)

    idx_x = np.argsort(x_error)[-n:][::-1]
    idx_y = np.argsort(y_error)[-n:][::-1]
    idx_z = np.argsort(z_error)[-n:][::-1]

    idx_roll = np.argsort(roll_error)[-n:][::-1]
    idx_pitch = np.argsort(pitch_error)[-n:][::-1]
    idx_yaw = np.argsort(yaw_error)[-n:][::-1]

    return (
        idx_x,
        idx_y,
        idx_z,
        idx_roll,
        idx_pitch,
        idx_yaw,
    )


def plot_motion_vs_error(est: np.ndarray, gt: np.ndarray, title:str = "Error magnitudes"):

    if est.shape != gt.shape:
        raise ValueError("est and gt must have the same shape.")

    # ---------- Translation ----------
    gt_translation = gt[:, :3, 3]
    est_translation = est[:, :3, 3]

    gt_translation_mag = np.linalg.norm(gt_translation, axis=1)
    translation_error = np.linalg.norm(est_translation - gt_translation, axis=1)

    t_correlation = np.corrcoef(gt_translation_mag, translation_error)[1, 0]

    # ---------- Rotation ----------
    R_gt = gt[:, :3, :3]
    R_est = est[:, :3, :3]
    # print(R_gt.shape, R_est.shape)
    # breakpoint()
    # # Relative rotation
    # print(len(R_gt), len(R_est))
    R_err = np.matmul(np.transpose(R_gt, (0, 2, 1)), R_est)

    # Rotation magnitude of GT
    trace_gt = np.trace(R_gt, axis1=1, axis2=2)
    gt_rotation = np.arccos(
        np.clip((trace_gt - 1.0) / 2.0, -1.0, 1.0)
    )

    # Rotation error
    trace_err = np.trace(R_err, axis1=1, axis2=2)
    rotation_error = np.arccos(
        np.clip((trace_err - 1.0) / 2.0, -1.0, 1.0)
    )

    # Convert to degrees
    gt_rotation_deg = np.degrees(gt_rotation)
    rotation_error_deg = np.degrees(rotation_error)

    r_correlation = np.corrcoef(gt_rotation_deg, rotation_error_deg)[1, 0]

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(gt_translation_mag, translation_error, s=10)
    axes[0].set_title(f"Translation, pearson correlation: {t_correlation:.2f}")
    axes[0].set_xlabel("GT translation magnitude [mm]")
    axes[0].set_ylabel("Translation error [mm]")
    axes[0].grid(True)

    axes[1].scatter(gt_rotation_deg, rotation_error_deg, s=10)
    axes[1].set_title(f"Rotation, pearson correlation: {r_correlation:.2f}")
    axes[1].set_xlabel("GT rotation angle [°]")
    axes[1].set_ylabel("Rotation error [°]")
    axes[1].grid(True)

    fig.canvas.manager.set_window_title(title)

    plt.tight_layout()

    return fig


def plot_trajectories(trajectories, labels=None, colors=None, title:str = "Trajectories"):

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
    fig.canvas.manager.set_window_title(title)
    # plt.show()
    # breakpoint()

    return fig


def extract_positions(values:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    xs, ys, zs = [], [], []

    for el in values:

        xs.append(el[0, 3])
        ys.append(el[1, 3])
        zs.append(el[2, 3])

    return np.array(xs), np.array(ys), np.array(zs)


if __name__ == "__main__":
    main()