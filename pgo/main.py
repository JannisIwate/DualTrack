import torch
import os
import sys
import h5py
import pandas as pd
sys.path.append(os.getcwd())
from graph.build_graph import *
from graph.error_metrics import *
sys.path.append("/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack")
from src.utils.pose import get_drift_metrics

## setup
DATA_PATH_DT25_25 = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_25_val/validation_run/scans/"
DATA_PATH_DT25_24 = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_24_val/run_1/scans/"

metrics_original = []
metrics_after_pgo = []

for el in os.listdir(DATA_PATH_DT25_24):
    sweep_path = os.path.join(DATA_PATH_DT25_24, el, "export.h5")

    ## load data
    with h5py.File(sweep_path, "r") as f:

        pred_acc = np.array(f["pred_tracking"])
        gt_acc = np.array(f["gt_tracking"])

        # print("pred_acc shape:", pred_acc.shape)
        # print("gt_acc shape:", gt_acc.shape)

        # compute inbetween transforms
        pred_inbetween = compute_inbetween_transforms(pred_acc)
        gt_inbetween = compute_inbetween_transforms(gt_acc)

        # convert to torch
        pred_acc_torch = torch.from_numpy(pred_acc).float()
        gt_acc_torch = torch.from_numpy(gt_acc).float()

        pred_inbetween_torch = torch.from_numpy(pred_inbetween).float()
        gt_inbetween_torch = torch.from_numpy(gt_inbetween).float()

        # sanity check
        reconstructed = pred_acc[9] @ pred_inbetween[10]

        # print("\nReconstruction error (should be near zero):")
        # print(np.abs(reconstructed - pred_acc[10]).max())

    # remove unnecessary first and last transform
    # pred_inbetween_torch = pred_inbetween_torch[1:-1]
    # pred_acc_torch = pred_acc_torch[1:-1]
    # gt_inbetween_torch = gt_inbetween_torch[1:-1]
    # gt_acc_torch = gt_acc_torch[1:-1]

    ## build graphs
    graph_pred, initial_pred, optimized_pred = build_graph(pred_acc_torch, pred_inbetween_torch, True)
    graph_gt, initial_gt, optimized_gt = build_graph(gt_acc_torch, gt_inbetween_torch, True)

    ## plot and evaluate
    # plot marginals
    #plot_marginals(graph_pred, optimized_pred, 1, 10)

    # plot trajectories (no rotation!)
    # plot_trajectories([extract_positions(pred_acc_torch, pose_type="torch_tensor"), extract_positions(gt_acc_torch, pose_type="torch_tensor")],
    #                   labels=["Initial estimated", "GT"],
    #                   colors=["blue", "red"])

    # plot_trajectories([extract_positions(initial_gt, pose_type="gtsam_values"), extract_positions(optimized_gt, pose_type="gtsam_values")],
    #                   labels=["GT initial", "GT acc"],
    #                   colors=["blue", "red"])

    # error metrics
    # error_pred_initial = graph_pred.error(initial_pred)
    # error_pred_optimized = graph_pred.error(optimized_pred)
    # error_gt_initial = graph_gt.error(initial_gt)
    # error_gt_optimized = graph_gt.error(optimized_gt)

    # print("\n\n==== Errors ====\n")

    # print(f"Initial graph error pred: {error_pred_initial}\n")
    # print(f"Optimized graph error pred: {error_pred_optimized}\n")
    # print(f"Initial graph error gt: {error_gt_initial}\n")
    # print(f"Optimized graph error gt: {error_gt_optimized}\n")

    # avg_t_ib_err, avg_r_ib_err = avg_trajectory_error(gt_inbetween_torch, pred_inbetween_torch)
    # avg_t_acc_err, avg_r_acc_err = avg_trajectory_error(gt_acc_torch, pred_acc_torch)
    # print(f"Average translation error inbetween (gt and initial pred)\n: {avg_t_ib_err}\n")
    # print(f"Average rotation error inbetween (gt and initial pred)\n: {avg_r_ib_err}\n")
    # print(f"Average translation error accumulated (gt and initial pred)\n: {avg_t_acc_err}\n")
    # print(f"Average rotation error accumulated (gt and initial pred)\n: {avg_r_acc_err}\n")

    # avg_t_ib_err, avg_r_ib_err = avg_trajectory_error(gt_inbetween_torch, gtsam_values_to_torch(optimized_pred))
    # print(f"Average translation error inbetween (gt and optimized pred)\n: {avg_t_ib_err}\n")
    # print(f"Average rotation error inbetween (gt and optimized pred)\n: {avg_r_ib_err}\n")

    # convert inbetween transforms to accumulated
    def inbetween_to_accumulated(inbetween_transforms):
        """Convert inbetween relative transforms to accumulated absolute transforms."""
        accumulated = [np.eye(4)]
        for i in range(len(inbetween_transforms)):
            accumulated.append(accumulated[-1] @ inbetween_transforms[i])
        return np.stack(accumulated)

    # gt_acc_torch vs pred_acc_torch (should return same values as evaluate.py)
    drift_metrics_pred_vs_gt_acc = get_drift_metrics(gt_acc_torch.numpy(), pred_acc_torch.numpy())
    metrics_original.append(drift_metrics_pred_vs_gt_acc)

    # gt_inbetween_torch vs optimized_pred
    optimized_pred_torch = gtsam_values_to_torch(optimized_pred).numpy()
    drift_metrics_optimized_vs_gt_ib = get_drift_metrics(gt_acc_torch.numpy(), optimized_pred_torch)
    metrics_after_pgo.append(drift_metrics_optimized_vs_gt_ib)
    #break

## drift metrics
avg_metrics_original_df = pd.DataFrame(metrics_original).mean()
avg_metrics_after_pgo_df = pd.DataFrame(metrics_after_pgo).mean()

print(f"Avg drift metrics (gt vs initial pred):")
print(f"  Final drift rate: {avg_metrics_original_df['final_drift_rate']:.4f}%")
print(f"  Avg drift rate: {avg_metrics_original_df['avg_drift_rate']:.4f}%")
print(f"  Max drift: {avg_metrics_original_df['max_drift']:.4f} mm")
print(f"  Sum of drift: {avg_metrics_original_df['sum_of_drift']:.4f} mm\n")

print(f"Avg drift metrics (gt vs optimized pred):")
print(f"  Final drift rate: {avg_metrics_after_pgo_df['final_drift_rate']:.4f}%")
print(f"  Avg drift rate: {avg_metrics_after_pgo_df['avg_drift_rate']:.4f}%")
print(f"  Max drift: {avg_metrics_after_pgo_df['max_drift']:.4f} mm")
print(f"  Sum of drift: {avg_metrics_after_pgo_df['sum_of_drift']:.4f} mm\n")