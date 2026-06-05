import gtsam
import numpy as np
import torch
import pandas as pd


def pose3_to_mat4(pose):

    R = pose.rotation().matrix()
    t = pose.translation()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def mat4_to_pose3(T):

    T = T.cpu().numpy()

    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(*T[:3, 3])

    return gtsam.Pose3(R, t)


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


def gtsam_values_to_torch(values: gtsam.Values, dtype=torch.float32):

    data_list = []

    for key in values.keys():

        val = values.atPose3(key)
        matrix = pose3_to_mat4(val)
        data_list.append(matrix)

    stacked = np.stack(data_list, axis=0)
    return torch.tensor(stacked, dtype=dtype)


def inbetween_to_accumulated(inbetween_transforms):

        accumulated = [np.eye(4)]

        for i in range(len(inbetween_transforms)):

            accumulated.append(accumulated[-1] @ inbetween_transforms[i])
        return np.stack(accumulated)