import torch
import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
sys.path.append("/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack")

from graph.build_graph import *
from graph.error_metrics import *
from src.utils.pose import get_drift_metrics


def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the data directory containing tracking data (export.h5 files)"
    )
    parser.add_argument(
        "--loop_closure", "--lc",
        action="store_true",
        dest="loop_closure",
        help="Enable loop closure detection during graph optimization"
    )
    parser.add_argument(
        "--image_registration", "--ir",
        action="store_true",
        dest="image_registration",
        help="Enable image registration for additional constraint refinement"
    )
    parser.add_argument(
        "--optical_flow", "--of",
        action="store_true",
        dest="optical_flow",
        help="Enable optical flow-based constraints in the graph"
    )

    return parser.parse_args()


def main():

    args = parse_arguments()
    
    data_path = args.input
    
    metrics_original = []
    metrics_after_pgo = []
    
    for el in os.listdir(data_path):
        sweep_path = os.path.join(data_path, el, "export.h5")

        ## load data
        with h5py.File(sweep_path, "r") as f:

            pred_acc = np.array(f["pred_tracking"])
            gt_acc = np.array(f["gt_tracking"])

            # compute inbetween transforms
            pred_inbetween = compute_inbetween_transforms(pred_acc)
            gt_inbetween = compute_inbetween_transforms(gt_acc)

            # convert to torch
            pred_acc_torch = torch.from_numpy(pred_acc).float()
            gt_acc_torch = torch.from_numpy(gt_acc).float()

            pred_inbetween_torch = torch.from_numpy(pred_inbetween).float()
            gt_inbetween_torch = torch.from_numpy(gt_inbetween).float()

            # sanity check
            # reconstructed = pred_acc[9] @ pred_inbetween[10]
            # print("\nReconstruction error (should be near zero):")
            # print(np.abs(reconstructed - pred_acc[10]).max())

        ## build graphs
        _, _, optimized_pred = build_graph(pred_acc_torch, pred_inbetween_torch, True)
        #_, _, optimized_gt = build_graph(gt_acc_torch, gt_inbetween_torch, True)

        # gt_acc_torch vs pred_acc_torch (should return same values as evaluate.py)
        drift_metrics_pred_vs_gt_acc = get_drift_metrics(gt_acc_torch.numpy(), pred_acc_torch.numpy())
        metrics_original.append(drift_metrics_pred_vs_gt_acc)

        # gt_inbetween_torch vs optimized_pred
        optimized_pred_torch = gtsam_values_to_torch(optimized_pred).numpy()
        drift_metrics_optimized_vs_gt_ib = get_drift_metrics(gt_acc_torch.numpy(), optimized_pred_torch)
        metrics_after_pgo.append(drift_metrics_optimized_vs_gt_ib)
        #break
    
    # drift metrics
    avg_metrics_original_df = pd.DataFrame(metrics_original).mean()
    avg_metrics_after_pgo_df = pd.DataFrame(metrics_after_pgo).mean()

    print(f"\nAvg drift metrics (gt vs initial pred):")
    print(f"  Final drift rate:  {avg_metrics_original_df['final_drift_rate']:.4f}%")
    print(f"  Avg drift rate:    {avg_metrics_original_df['avg_drift_rate']:.4f}%")
    print(f"  Max drift:         {avg_metrics_original_df['max_drift']:.4f} mm")
    print(f"  Sum of drift:      {avg_metrics_original_df['sum_of_drift']:.4f} mm")

    print(f"\nAvg drift metrics (gt vs optimized pred)")
    print(f"  Final drift rate:  {avg_metrics_after_pgo_df['final_drift_rate']:.4f}%")
    print(f"  Avg drift rate:    {avg_metrics_after_pgo_df['avg_drift_rate']:.4f}%")
    print(f"  Max drift:         {avg_metrics_after_pgo_df['max_drift']:.4f} mm")
    print(f"  Sum of drift:      {avg_metrics_after_pgo_df['sum_of_drift']:.4f} mm")

if __name__ == "__main__":
    main()
