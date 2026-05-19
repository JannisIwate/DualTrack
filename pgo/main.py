import torch
import os
import sys
import h5py
sys.path.append(os.getcwd())
from graph.build_graph import *
from graph.error_metrics import *

## setup
INPUT_FILE = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/eval/test/scans/sweep_00000/export.h5"

## load data
with h5py.File(INPUT_FILE, "r") as f:

    pred_acc = np.array(f["pred_tracking"])
    gt_acc = np.array(f["gt_tracking"])

    print("pred_acc shape:", pred_acc.shape)
    print("gt_acc shape:", gt_acc.shape)

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

    print("\nReconstruction error (should be near zero):")
    print(np.abs(reconstructed - pred_acc[10]).max())

# remove unnecessary first and last transform
inbetween_transforms_pred = pred_inbetween_torch[1:-1]
acc_transforms_pred = pred_acc_torch[1:-1]
inbetween_transforms_gt = gt_inbetween_torch[1:-1]
acc_transforms_gt = gt_acc_torch[1:-1]

## build graphs
graph_pred, initial_pred, optimized_pred = build_graph(acc_transforms_pred, inbetween_transforms_pred, True)
graph_gt, initial_gt, optimized_gt = build_graph(acc_transforms_gt, inbetween_transforms_gt, True)

## plot and evaluate
# plot marginals
#plot_marginals(graph_pred, optimized_pred, 1, 10)

# plot trajectories (no rotation!)
# plot_trajectories([extract_positions(acc_transforms_pred, pose_type="torch_tensor"), extract_positions(acc_transforms_gt, pose_type="torch_tensor")],
#                   labels=["Initial estimated", "GT"],
#                   colors=["blue", "red"])

plot_trajectories([extract_positions(initial_gt, pose_type="gtsam_values"), extract_positions(optimized_gt, pose_type="gtsam_values")],
                  labels=["GT initial", "GT acc"],
                  colors=["blue", "red"])

# error metrics
error_pred_initial = graph_pred.error(initial_pred)
error_pred_optimized = graph_pred.error(optimized_pred)
error_gt_initial = graph_gt.error(initial_gt)
error_gt_optimized = graph_gt.error(optimized_gt)

print("\n\n==== Errors ====\n")

print(f"Initial error pred: {error_pred_initial}\n")
print(f"Optimized error pred: {error_pred_optimized}\n")
print(f"Initial error gt: {error_gt_initial}\n")
print(f"Optimized error gt: {error_gt_optimized}\n")

avg_t_err, avg_r_err = avg_trajectory_error(acc_transforms_gt, acc_transforms_pred)
print(f"Average translation error acc\n: {avg_t_err}\n")
print(f"Average rotation error acc\n: {avg_r_err}\n")

avg_t_err, avg_r_err = avg_trajectory_error(inbetween_transforms_gt, inbetween_transforms_pred)
print(f"Average translation error inbetween\n: {avg_t_err}\n")
print(f"Average rotation error inbetween\n: {avg_r_err}\n")