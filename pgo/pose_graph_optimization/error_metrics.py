
# import sys
# import os
# sys.path.append(os.getcwd())
from .utils import mat4_to_pose3
import numpy as np


def pose_error(T_gt, T_pred):

    delta = T_gt.between(T_pred)
    t_err = delta.translation()
    r_err = delta.rotation().matrix()

    return t_err, r_err

def avg_trajectory_error(transforms_1, transforms_2):

    if len(transforms_1) != len(transforms_2):
        raise ValueError("Inputs must have the same length")

    avg_t_err, avg_r_err = np.zeros(3), np.zeros(shape=(3, 3))

    for el in zip(transforms_1, transforms_2):
        T_gt = mat4_to_pose3(el[0])
        T_pred = mat4_to_pose3(el[1])
        t_err, r_err = pose_error(T_gt, T_pred)
        avg_t_err += t_err
        avg_r_err += r_err
    avg_t_err /= len(transforms_1)
    avg_r_err /= len(transforms_1)

    return avg_t_err, avg_r_err