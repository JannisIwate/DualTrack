import h5py
import numpy as np
import torch


INPUT_FILE = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/eval/test/scans/sweep_00000/export.h5"


def compute_inbetween_transforms(acc_transforms):

    n = acc_transforms.shape[0]

    inbetween = np.zeros_like(acc_transforms)

    # first frame has no predecessor
    inbetween[0] = np.eye(4)

    for i in range(1, n):
        prev_T = acc_transforms[i - 1]
        curr_T = acc_transforms[i]

        delta_T = np.linalg.inv(prev_T) @ curr_T

        inbetween[i] = delta_T

    return inbetween


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