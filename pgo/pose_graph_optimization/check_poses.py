import os
import torch
import numpy as np


BASE_PATH = os.path.join(os.getcwd(), "../freehand_adapted", "results", "seq_len10__lr0.0001__pred_type_parameter__label_type_point")

abs_pose_preds = torch.load(BASE_PATH + '/pose_data/predictions.pt').cpu().numpy()
# (N, 3, 4), x y z for four corner points
rel_transformation_preds = torch.load(BASE_PATH + '/pose_data/predictions_transforms_locaL.pt').cpu().numpy()
# (N, 4, 4), rot in [0:2, 0:2], translation in [0:3, 3], lower row is [0,0,0,1]
i = 1

def to_homogeneous(points:np.ndarray) -> np.ndarray:

    ones = np.ones((1, points.shape[1]))

    return np.concatenate([points, ones], axis=0)

def apply_transform(T:np.ndarray, points:np.ndarray) -> np.ndarray:

    pts_h = to_homogeneous(points)          # (4,4)
    pts_t = np.matmul(T, pts_h)             # (4,4)

    return pts_t[:3]

def compute_error(p1:np.ndarray, p2:np.ndarray) -> np.ndarray:

    return np.linalg.norm(p1 - p2, axis=0).mean()

def check_internal_consistency(pred_pts:np.ndarray, pred_T:np.ndarray, use_inverse=False) -> np.ndarray:

    # check if point poses and transforms between those match
    N = pred_pts.shape[0]
    errors = []

    for i in range(N - 2):
        pose_i = pred_pts[i+1]
        pose_next = pred_pts[i+2]
        T = pred_T[i+1]

        if use_inverse:
            T = np.linalg.inv(T)

        pose_check = apply_transform(T, pose_i)

        err = compute_error(pose_check, pose_next)
        errors.append(err)
        #break

    errors = np.array(errors)

    print("\nSummary:")
    print(f"Mean: {np.mean(errors):.6f}")
    print(f"Max:  {np.max(errors):.6f}")

    return errors

check_internal_consistency(abs_pose_preds, rel_transformation_preds, use_inverse=False)