import os
import torch
import numpy as np
from gtsam import Pose3, Rot3, Point3


BASE_PATH = os.path.join(os.getcwd(), "../freehand_adapted", "results", "seq_len10__lr0.0001__pred_type_parameter__label_type_point")

abs_pose_preds = torch.load(BASE_PATH + '/pose_data/predictions.pt')
# (N, 3, 4), x y z for four corner points
rel_transformation_preds = torch.load(BASE_PATH + '/pose_data/predictions_transforms_locaL.pt')
# (N, 4, 4), rot in [0:2, 0:2], translation in [0:3, 3], lower row is [0,0,0,1]
i = 1
# print(abs_pose_preds[i])
# print(abs_pose_preds[i+1])
# print(rel_transformation_preds[i+1])

def to_homogeneous(points):
    ones = torch.ones(1, points.shape[1])
    return torch.cat([points, ones], dim=0)

def apply_transform(T, points):
    pts_h = to_homogeneous(points)          # (4,4)
    pts_t = T @ pts_h                       # (4,4)
    return pts_t[:3]

def compute_error(p1, p2):
    return torch.norm(p1 - p2, dim=0).mean()

def check_internal_consistency(pred_pts, pred_T, use_inverse=False):
    N = pred_pts.shape[0]
    errors = []

    for i in range(N - 2):
        pose_i = pred_pts[i+1]
        pose_next = pred_pts[i+2]
        T = pred_T[i+1]   # matches your indexing (you stored at idx_f0+1)

        if use_inverse:
            T = torch.inverse(T)

        pose_check = apply_transform(T, pose_i)
        # print("T:\n", T)
        # print("pose_next:\n", pose_next)
        # print("pose_i:\n", pose_i)
        # print("pose_check:\n", pose_check)
        

        err = compute_error(pose_check, pose_next)
        errors.append(err.item())
        #break


    errors = torch.tensor(errors)

    print("\nSummary:")
    print(f"Mean: {errors.mean():.6f}")
    print(f"Max:  {errors.max():.6f}")

    return errors

check_internal_consistency(abs_pose_preds, rel_transformation_preds, use_inverse=False)