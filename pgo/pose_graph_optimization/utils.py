import gtsam
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def pose3_to_mat4(pose: gtsam.Pose3) -> np.ndarray:

    R = pose.rotation().matrix()
    t = pose.translation()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def mat4_to_pose3(T: np.ndarray) -> gtsam.Pose3:

    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(*T[:3, 3])

    return gtsam.Pose3(R, t)


def compute_inbetween_transforms(acc_transforms: np.ndarray) -> np.ndarray:

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


def gtsam_to_numpy(values: gtsam.Values) -> np.ndarray:

    data_list = []

    for key in values.keys():
        val = values.atPose3(key)
        matrix = pose3_to_mat4(val)
        data_list.append(matrix)

    stacked = np.stack(data_list, axis=0)

    return stacked


def accumulate(inbetween_transforms: np.ndarray) -> np.ndarray:

    accumulated = np.eye(4)

    for i in range(len(inbetween_transforms)):
        accumulated = accumulated @ inbetween_transforms[i]

    return accumulated


def inbetween_to_accumulated(inbetween_transforms: np.ndarray) -> np.ndarray:

    accumulated = [np.eye(4)]

    for i in range(len(inbetween_transforms)):
        accumulated.append(accumulated[-1] @ inbetween_transforms[i])

    return np.stack(accumulated)


def threedof_to_sixdof(
    T_ref: np.ndarray,
    T_2d: np.ndarray,
) -> np.ndarray:
    
    T_new = T_ref.copy()

    # translation
    T_new[0, 3] = T_2d[0, 2]
    T_new[1, 3] = T_2d[1, 2]

    # observed image rotation
    Rz = np.eye(3)
    Rz[:2, :2] = T_2d[:2, :2]

    R_ref = T_ref[:3, :3]

    # keep roll/pitch from ref, replace only image-plane rotation
    R_delta = Rotation.from_matrix(Rz)
    rot_ref = Rotation.from_matrix(R_ref)

    # adjust ref rotation by registration rotation
    rot_new = R_delta * rot_ref

    T_new[:3, :3] = rot_new.as_matrix()

    return T_new


def pose3_to_se2(T: np.ndarray) -> tuple[float, float, float]:

    x = float(T[0, 3])
    y = float(T[1, 3])

    R = T[:3, :3]

    yaw = float(
        np.arctan2(
            R[1, 0],
            R[0, 0],
        )
    )

    return x, y, yaw