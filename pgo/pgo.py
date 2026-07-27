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
from pose_graph_optimization.image_registration import register_2d, register_3d, sample_pairs_by_step, sample_sliding_windows
from error_evals import estimate_gt
from src.utils.pose import get_drift_metrics, get_ddf_metrics, get_global_and_relative_gt_trackings, plot_pose_differences, pose_vector_to_matrix, matrix_to_pose_vector
from src.evaluator import plot_pose_differences


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, type=str)

    return parser.parse_args()


# ==========================================================================
# config helpers
# ==========================================================================

def cfg_has(config, path: str) -> bool:

    return OmegaConf.select(config, path) is not None


def cfg_require(config, path: str):

    value = OmegaConf.select(config, path)

    if value is None:

        raise ValueError(f"Missing required config entry: {path}")
    
    return value


def cfg_get(config, path: str):

    return OmegaConf.select(config, path) if cfg_has(config, path) else None


def get_execution_flags(config) -> dict:

    options = cfg_get(config, "general.options") or []

    return {
        "pgo": "pgo" in options,
        "noise_constraints": "noise_constraints" in options,
        "la_replace": "la_replace" in options,
    }


def init_results() -> dict:

    return {
        "pgo": {
            "drift_metrics_original": [], "drift_metrics_after_pgo": [],
            "ddf_metrics_original": [], "ddf_metrics_after_pgo": [],
            "graph": None, "initial": None, "optimized": None,
        },
        "ir": {
            "metrics": {},
            "transforms": {"ir_transforms": [], "ir_gt_transforms": [], "ir_ref_transforms": []},
            "drift_metrics_after_ir": [], "ddf_metrics_after_ir": [],
            "figs_individual": {}, "figs_general": {},
        },
    }


# ==========================================================================
# other helpers
# ==========================================================================

def largest_pose_errors(est: np.ndarray, gt: np.ndarray, n: int = 5) -> tuple[
                                                                            np.ndarray, np.ndarray, np.ndarray,
                                                                            np.ndarray, np.ndarray, np.ndarray,
                                                                        ]:
    if est.shape != gt.shape:

        raise ValueError("est and gt must have the same shape.")

    def rotation_to_angles(R: np.ndarray):

        pitch = np.arctan2(-R[:, 2, 0], np.sqrt(R[:, 2, 1] ** 2 + R[:, 2, 2] ** 2))
        roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])

        return roll, pitch, yaw

    def angle_error(a, b):

        return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

    x_error = np.abs(est[:, 0, 3] - gt[:, 0, 3])
    y_error = np.abs(est[:, 1, 3] - gt[:, 1, 3])
    z_error = np.abs(est[:, 2, 3] - gt[:, 2, 3])

    roll_est, pitch_est, yaw_est = rotation_to_angles(est[:, :3, :3])
    roll_gt, pitch_gt, yaw_gt = rotation_to_angles(gt[:, :3, :3])

    roll_error = angle_error(roll_est, roll_gt)
    pitch_error = angle_error(pitch_est, pitch_gt)
    yaw_error = angle_error(yaw_est, yaw_gt)

    return (
        np.argsort(x_error)[-n:][::-1], np.argsort(y_error)[-n:][::-1], np.argsort(z_error)[-n:][::-1],
        np.argsort(roll_error)[-n:][::-1], np.argsort(pitch_error)[-n:][::-1], np.argsort(yaw_error)[-n:][::-1],
    )


def extract_positions(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    xs, ys, zs = [], [], []

    for el in values:

        xs.append(el[0, 3])
        ys.append(el[1, 3])
        zs.append(el[2, 3])

    return np.array(xs), np.array(ys), np.array(zs)


# ==========================================================================
# plotting
# ==========================================================================

def plot_motion_vs_error(est: np.ndarray, gt: np.ndarray, title: str = "Error magnitudes"):

    if est.shape != gt.shape:

        raise ValueError("est and gt must have the same shape.")

    gt_translation = gt[:, :3, 3]
    est_translation = est[:, :3, 3]
    gt_translation_mag = np.linalg.norm(gt_translation, axis=1)
    translation_error = np.linalg.norm(est_translation - gt_translation, axis=1)
    t_correlation = np.corrcoef(gt_translation_mag, translation_error)[1, 0]

    R_gt, R_est = gt[:, :3, :3], est[:, :3, :3]
    R_err = np.matmul(np.transpose(R_gt, (0, 2, 1)), R_est)

    trace_gt = np.trace(R_gt, axis1=1, axis2=2)
    gt_rotation = np.arccos(np.clip((trace_gt - 1.0) / 2.0, -1.0, 1.0))

    trace_err = np.trace(R_err, axis1=1, axis2=2)
    rotation_error = np.arccos(np.clip((trace_err - 1.0) / 2.0, -1.0, 1.0))

    gt_rotation_deg = np.degrees(gt_rotation)
    rotation_error_deg = np.degrees(rotation_error)
    r_correlation = np.corrcoef(gt_rotation_deg, rotation_error_deg)[1, 0]

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


def plot_trajectories(trajectories, labels=None, colors=None, title: str = "Trajectories"):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = len(trajectories)
    labels = labels or [f"traj_{i}" for i in range(n)]
    colors = colors or [None] * n

    for i, (xs, ys, zs) in enumerate(trajectories):

        ax.plot(xs, ys, zs, label=labels[i], color=colors[i])
        ax.scatter(xs[0], ys[0], zs[0], color=colors[i])
        ax.scatter(xs[-1], ys[-1], zs[-1], color=colors[i], marker="s")

    ax.set_title("Pose Graph Trajectories")
    ax.legend()
    fig.canvas.manager.set_window_title(title)

    return fig


# ==========================================================================
# data loading
# ==========================================================================

def get_scan_list(config, input_pred: str):

    data = os.listdir(input_pred)
    nr_of_scans = cfg_get(config, "general.nr_scans")
    start = cfg_get(config, "general.start_scan") or 1

    if nr_of_scans is not None:

        data = list(islice(data, start - 1, start + nr_of_scans - 1))

    return data, nr_of_scans


def load_scan_data(input_pred: str, el: str, sweep_index: int, config):

    sweep_path = os.path.join(input_pred, el, "export.h5")

    if not os.path.isfile(sweep_path):

        return None

    with h5py.File(sweep_path, "r") as f:

        nr_of_frames = cfg_get(config, "general.nr_frames")
        nr_of_frames = len(f["images"]) if nr_of_frames is None else nr_of_frames + 1

        pred_acc = np.array(f["pred_tracking_glob"][:nr_of_frames])  # normalized acc world poses
        pred_inbetween = np.array(f["pred_tracking_loc"][:nr_of_frames])  # relative poses Ti->j

        if cfg_has(config, "dirs.input_gt"):

            with h5py.File(os.path.join(config.dirs.input_gt, f"{el}.h5"), "r") as f_gt:

                gt = np.array(f_gt["tracking"][:nr_of_frames])
                gt_acc, gt_inbetween = get_global_and_relative_gt_trackings(gt)
        else:

            gt_acc = np.array(f["gt_tracking"][:nr_of_frames])
            gt_inbetween = compute_inbetween_transforms(gt_acc)

        calibration_matrix = np.round(np.array(f["pixel_to_image"]), 4)
        fvs = np.array(f["fvs"])
        frames = np.array(f["images"][:nr_of_frames])
        h, w = np.array(f["dimensions"])[:2]

    return {
        "sweep_name": f"sweep_{sweep_index}",
        "pred_acc": pred_acc, "pred_inbetween": pred_inbetween,
        "gt_acc": gt_acc, "gt_inbetween": gt_inbetween,
        "calibration_matrix": calibration_matrix, "fvs": fvs, "frames": frames,
        "image_shape_hw": (w, h),
    }


# ==========================================================================
# PGO
# ==========================================================================

def init_pose_graph(pred_acc: np.ndarray, pred_inbetween: np.ndarray, flags: dict):

    if flags["la_replace"]:

        pred_inbetween = linear_approximation(pred_inbetween, 0.9, 0.8)
        # pred_inbetween = estimate_gt(pred_inbetween)
        pred_acc = inbetween_to_accumulated(pred_inbetween[1:])
        # -> doesn't help as worsening outweighs improvement

    pred_graph = PoseGraph(poses=pred_acc, constraints=pred_inbetween, initial_pose=pred_acc[0])

    return pred_graph, pred_acc, pred_inbetween


def run_loop_closure(pred_graph, fvs: np.ndarray, frames: np.ndarray, pred_inbetween: np.ndarray, config):

    loop_closures = detect_loop_closures(
        feature_vectors=fvs, frames=frames, transforms=pred_inbetween,
        image_registration_cfg=config.image_registration, **config.loop_closure,
    )

    for lc in loop_closures:

        pred_graph.add_constraint(
            lc["source_idx"], lc["target_idx"], lc["transform"],
            registration_noise_model(confidence=lc["combined_score"], ref_sigma=config.general.ref_values_sigma),
        )


def run_optical_flow(pred_graph, config):

    # TODO
    pass


def run_noise_constraints(pred_graph, pred_inbetween: np.ndarray, config):

    pred_inbetween_la = linear_approximation(pred_inbetween, 0.9, 0.8)

    for i, transform in enumerate(pred_inbetween_la[1:-1]):

        pred_graph.add_constraint(node_i=i, node_j=i + 1, transform=transform)


def get_noise(pred_inbetween):

    pred_vector = np.stack([matrix_to_pose_vector(i) for i in pred_inbetween])  # (N, 6)

    last_gt_bin_ranges = np.array([
        [0.4370, 1.4032], [0.0721, 0.2460], [0.3952, 1.2951],
        [0.0718, 0.6201], [0.0634, 0.2451], [0.0769, 0.5950],
    ])  # (6, 2)

    noise = np.full_like(pred_vector, 1e-2, dtype=float)

    for dof in range(6):

        lower, upper = last_gt_bin_ranges[dof]
        mask = (np.abs(pred_vector[:, dof]) >= lower) & (np.abs(pred_vector[:, dof]) <= upper)
        noise[mask, dof] = 1e-1

    return noise


def linear_approximation(
    pred_inbetween,
    quantile=0.9,
    scale=0.8,
    max_factor=10.0,
):

    SLOPE = np.array([
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

    # LA correction for large values, as these benefit the most from it
    pred_inbetween_vector = np.stack(
        [matrix_to_pose_vector(T) for T in pred_inbetween]
    )

    for dof in range(pred_inbetween_vector.shape[1]):

        # only last quantile
        lower = np.quantile(
            np.abs(pred_inbetween_vector[:, dof]),
            quantile,
        )

        mask = np.abs(pred_inbetween_vector[:, dof]) >= lower

        pred = pred_inbetween_vector[mask, dof]

        # la
        la = (
            scale *
            (pred - INTERCEPT[dof]) /
            (1.0 + SLOPE[dof])
        )

        # only keep reasonable corrections (doesn't help)
        ratio = np.abs(la) / np.maximum(np.abs(pred), 1e-8)
        valid = ratio <= max_factor

        pred_inbetween_vector[mask, dof] = np.where(
            valid,
            la,
            pred,
        )

        break  # only x

    return np.stack(
        [pose_vector_to_matrix(v) for v in pred_inbetween_vector]
    )


# ==========================================================================
# IR
# ==========================================================================

def register_frame_pairs(idc1, idc2, frames_1, frames_2, ir_ref, ir_gt, config, counter=0):

    ir_metrics = {
        "metric": config.image_registration.sitk.metric,
        "metric_before": [], "metric_before_gt": [], "metric_before_pred": [],
        "metric_after": [], "ir_execution_time": [],
    }
    confidences = []
    ir_transforms = np.tile(np.eye(4), (ir_ref.shape[0], 1, 1))

    for i, _ in enumerate(idc1):

        start_time = time.time()
        (transform_ir,
         confidence,
         valid,
         m_before_id,
         m_before_gt,
         m_before_pred,
         m_after) = register_2d(frame_i=frames_1[i],
                             frame_j=frames_2[i],
                             ref_transform=ir_ref[i + 1], # skip identity
                             gt_transform=ir_gt[i + 1],
                             **config.image_registration)

        confidences.append(confidence)
        ir_metrics["metric_before"].append(m_before_id)
        ir_metrics["metric_before_gt"].append(m_before_gt)
        ir_metrics["metric_before_pred"].append(m_before_pred)
        ir_metrics["metric_after"].append(m_after)
        ir_metrics["ir_execution_time"].append(time.time() - start_time)
        ir_transforms[i + 1] = transform_ir

    return ir_metrics, ir_transforms, counter, confidences


def register_volumes(
    windows,
    first_window_start,
    pred_acc,
    pred_inbetween,
    config,
    counter,
):
    ir_metrics = {
        "metric": config.image_registration.sitk.metric,
        "metric_before": [],
        "metric_before_gt": [],
        "metric_before_pred": [],
        "metric_after": [],
        "ir_execution_time": [],
    }

    # Reference transforms are used by default.
    # Only frames that become the centre of a window are overwritten.
    ir_transforms = np.copy(pred_inbetween)

    window_size = windows.shape[1]

    half_window = window_size // 2

    first_registered = first_window_start + half_window

    for i, window in enumerate(windows):

        start_time = time.time()
        ref_idx_start = first_window_start + i
        ref_idx_end = ref_idx_start + window_size

        (
            ir_transform,
            m_before_id,
            m_before_gt,
            m_before_pred,
            m_after,
        ) = register_3d(
            window,
            pred_acc[ref_idx_start:ref_idx_end],
            config.image_registration.sitk,
        )

        # TODO: think about assigning found transform to pred_acc or keeping ref transforms as ref for found pose

        ir_metrics["metric_before"].append(m_before_id)
        ir_metrics["metric_before_gt"].append(m_before_gt)
        ir_metrics["metric_before_pred"].append(m_before_pred)
        ir_metrics["metric_after"].append(m_after)
        ir_metrics["ir_execution_time"].append(time.time() - start_time)

        centre_idx = first_registered + i

        ir_transforms[centre_idx] = ir_transform

    return ir_metrics, ir_transforms, counter


def create_ir_scan_plots(sweep_name, ir_ref, ir_gt, ir_transforms, config, figs_individual):

    plot_cfg = cfg_get(config, "plot")

    if plot_cfg is None:

        return

    figs = figs_individual.setdefault(sweep_name, {})

    if "plot_ir_pose_differences" in plot_cfg:

        figs["ir_pose_diffs_ref_gt"] = plot_pose_differences(ir_ref, ir_gt, title="GT vs Pred")
        plt.close()
        figs["ir_pose_diffs_ir_gt"] = plot_pose_differences(ir_transforms, ir_gt, title="GT vs IR")
        plt.close()

    if "plot_ir_trajectories" in plot_cfg:

        figs["ir_trajectories_gt_ref_ir"] = plot_trajectories([extract_positions(inbetween_to_accumulated(t[1:])) for t in (ir_gt, ir_ref, ir_transforms)],
                                                              labels=["GT", "Initial estimated", "IR"],
                                                              colors=["blue", "red", "black"])
        plt.close()


def run_image_registration(scan, pred_acc, config, results, counter=0):

    ir = results["ir"]
    frames, gt_acc = scan["frames"], scan["gt_acc"]

    ir_type = cfg_get(config, "image_registration.ir_type")

    if ir_type == "2d":

        step = cfg_get(config, "image_registration.step") or 1
        (idc1,
        idc2,
        ir_ref,
        ir_gt) = sample_pairs_by_step(frames,
                                    pred_acc,
                                    gt_acc,
                                    step)

        (ir_metrics,
        ir_transforms,
        counter,
        confidences) = register_frame_pairs(idc1,
                                            idc2,
                                            frames[idc1],
                                            frames[idc2],
                                            ir_ref,
                                            ir_gt,
                                            config,
                                            counter=counter)
    elif ir_type == "3d":

        WINDOW_SIZE = cfg_get("image_registration.window_size")
        if WINDOW_SIZE is None:
            WINDOW_SIZE = 10
        pred_inbetween = inbetween_to_accumulated(pred_acc[1:]) # skip first identity)
        windows, start = sample_sliding_windows(pred_inbetween, WINDOW_SIZE) # shape (438, 10, 4, 4)

        register_volumes(windows, start, pred_acc, pred_inbetween, config, counter=counter)

    else:

        raise ValueError("Invalid IR type")

    create_ir_scan_plots(scan["sweep_name"],
                         ir_ref,
                         ir_gt,
                         ir_transforms,
                         config,
                         ir["figs_individual"])

    ir_transforms_acc = inbetween_to_accumulated(ir_transforms[1:])
    ir_gt_acc = inbetween_to_accumulated(ir_gt[1:])

    ir["metrics"] = ir_metrics
    ir["transforms"]["ir_transforms"].append(ir_transforms[1:])
    ir["transforms"]["ir_gt_transforms"].append(ir_gt[1:])
    ir["transforms"]["ir_ref_transforms"].append(ir_ref[1:])
    ir["drift_metrics_after_ir"].append(get_drift_metrics(ir_gt_acc, ir_transforms_acc))
    ir["ddf_metrics_after_ir"].append(get_ddf_metrics(
        ir_transforms_acc, ir_transforms, ir_gt_acc, ir_gt,
        scan["calibration_matrix"], scan["image_shape_hw"], mode="5pt-landmark",
    ))

    return counter, ir["transforms"]["ir_transforms"], idc1, idc2, confidences


def create_ir_general_plots(results: dict, config):

    ir = results["ir"]
    ref = np.concatenate(ir["transforms"]["ir_ref_transforms"], axis=0)
    est = np.concatenate(ir["transforms"]["ir_transforms"], axis=0)
    gt = np.concatenate(ir["transforms"]["ir_gt_transforms"], axis=0)

    ir["figs_general"]["ir_error_mags_ref_gt"] = plot_motion_vs_error(ref, gt, title="Ref vs GT")
    plt.close()
    ir["figs_general"]["ir_error_mags_ir_gt"] = plot_motion_vs_error(est, gt, title="IR vs GT")
    plt.close()


def sample_sliding_windows(
    data: np.ndarray,
    window_size: int = 11,
    stride: int = 1,
) -> tuple[np.ndarray, int]:

    n = len(data)

    if window_size <= 0:

        raise ValueError("window_size must be > 0.")

    if stride <= 0:

        raise ValueError("stride must be > 0.")

    if n < window_size:

        raise ValueError("window_size must not exceed the number of samples.")

    if window_size % 2 == 0:
            
            raise ValueError("Slice-to-volume registration requires an odd window size.")

    windows = []

    # Start index of the last complete window
    last_start = n - window_size

    # Sample from back to front
    starts = list(range(last_start, -1, -stride))
    starts.reverse()

    for start in starts:
        windows.append(data[start:start + window_size])

    return np.stack(windows), starts[0]


# ==========================================================================
# metrics
# ==========================================================================

def compute_pgo_scan_metrics(scan, pred_acc, pred_inbetween, results, execute_pgo, optimized_pred=None):

    pgo = results["pgo"]
    gt_acc = torch.tensor(scan["gt_acc"])
    calib, shape = scan["calibration_matrix"], scan["image_shape_hw"]

    pgo["drift_metrics_original"].append(get_drift_metrics(gt_acc, pred_acc))
    pgo["ddf_metrics_original"].append(get_ddf_metrics(pred_acc,
                                                       pred_inbetween,
                                                       gt_acc,
                                                       scan["gt_inbetween"],
                                                       torch.tensor(calib),
                                                       shape,
                                                       mode="5pt-landmark"))

    if not execute_pgo:

        return

    pgo["drift_metrics_after_pgo"].append(get_drift_metrics(gt_acc, optimized_pred))
    pgo["ddf_metrics_after_pgo"].append(get_ddf_metrics(optimized_pred,
                                                        compute_inbetween_transforms(optimized_pred),
                                                        gt_acc,
                                                        scan["gt_inbetween"],
                                                        calib,
                                                        shape,
                                                        mode="5pt-landmark"))


def save_all_results(results: dict, config):

    pgo, ir = results["pgo"], results["ir"]

    save_results(
        output_dir=f"{config.dirs.output_dir}/results",
        graph=pgo["graph"], initial=pgo["initial"], optimized=pgo["optimized"],
        metrics_original=[pgo["drift_metrics_original"], pgo["ddf_metrics_original"]],
        metrics_after_pgo=[pgo["drift_metrics_after_pgo"], pgo["ddf_metrics_after_pgo"]],
        ir_metrics=[ir["metrics"], ir["drift_metrics_after_ir"], ir["ddf_metrics_after_ir"]],
        figs_individual=ir["figs_individual"], figs_general=ir["figs_general"],
    )


# ==========================================================================
# main
# ==========================================================================

def main():

    # load config
    args = parse_arguments()
    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)

    # load data
    input_pred = cfg_require(config, "dirs.input_pred")
    flags = get_execution_flags(config)
    data, nr_of_scans = get_scan_list(config, input_pred)

    # init
    results = init_results()
    counter = 0  # stride counter

    # iterate over scans
    for i, el in enumerate(tqdm(data, desc="Working", total=nr_of_scans)):

        scan = load_scan_data(input_pred, el, i, config)
        if scan is None:
            continue

        pred_acc, pred_inbetween = scan["pred_acc"], scan["pred_inbetween"]
        pred_graph = None

        # init graph and PGO
        if flags["pgo"]:

            pred_graph, pred_acc, pred_inbetween = init_pose_graph(pred_acc, pred_inbetween, flags)

            # LC
            if cfg_has(config, "loop_closure"):

                loop_closures = detect_loop_closures(feature_vectors=scan["fvs"],
                                                     frames=scan["frames"],
                                                     transforms=pred_inbetween,
                                                     image_registration_cfg=config.image_registration,
                                                     **config.loop_closure)

                for lc in loop_closures:

                    pred_graph.add_constraint(
                        lc["source_idx"],
                        lc["target_idx"],
                        lc["transform"],
                        registration_noise_model(confidence=lc["combined_score"], ref_sigma=config.general.ref_values_sigma))

            # OF
            if cfg_has(config, "loop_closure"):

                pass

            # noise constraints
            if flags["noise_constraints"]:

                pred_inbetween_la = linear_approximation(pred_inbetween, 0.9, 0.8)

                for i, transform in enumerate(pred_inbetween_la[1:-1]):

                    pred_graph.add_constraint(node_i=i, node_j=i + 1, transform=transform)
                # -> bringt trotzdem keine Verbesserung

        # IR
        if cfg_has(config, "image_registration"):

            (counter,
             ir_transforms,
             idc1,
             idc2,
             confidences) = run_image_registration(scan,
                                                   pred_acc,
                                                   config,
                                                   results,
                                                   counter=counter)

            # add IR contraints
            if flags["pgo"]:

                stride = cfg_get(config, "general.counter") or 1
                ref_sigma = cfg_get(config, "general.ref_values_sigma") or 1e-2

                for i, _ in enumerate(ir_transforms):

                    counter += 1
                    if counter % stride == 0:

                        pred_graph.add_constraint(
                            node_i=idc1[i],
                            node_j=idc2[i],
                            transform=ir_transforms[i],
                            ref_sigma=registration_noise_model(confidence=confidences[i],ref_sigma=ref_sigma))

        # optimize graph
        optimized_pred = None

        if flags["pgo"]:

            results["pgo"]["graph"], _, pred_optimized = pred_graph.build_graph()
            optimized_pred = gtsam_to_numpy(pred_optimized)

        # retrieve (pgo) results
        compute_pgo_scan_metrics(scan,
                                 pred_acc,
                                 pred_inbetween,
                                 results,
                                 flags["pgo"],
                                 optimized_pred)
        
        results["pgo"]["initial"] = pred_acc
        results["pgo"]["optimized"] = optimized_pred

    # create ir plots
    if (cfg_has(config, "image_registration") and
        cfg_has(config, "plot") and
        "plot_ir_error_magnitudes" in cfg_get(config, "plot")):

        create_ir_general_plots(results, config)

    # save results
    if cfg_has(config, "dirs.output_dir"):

        save_all_results(results, config)


if __name__ == "__main__":
    main()
