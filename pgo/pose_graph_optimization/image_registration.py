import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from pose_graph_optimization.utils import accumulate
from pose_graph_optimization.utils import pose3_to_se2
from scipy.spatial.transform import Rotation
from src.submission.tus_rec_challenge_baseline import transform
from pose_graph_optimization.sitk import sitk_2d_register, sitk_3d_register


def register_2d(
    sitk,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    ref_transform: np.ndarray,
    gt_transform: np.ndarray,
    step:int = 1,
    max_metric_change: float = 20,
    cross_check: bool = False,
    ir_type: str = "2d"
) -> tuple[
    np.ndarray,
    float,
    bool,
    float,
    float,
    float,
    float,
]:
    # --------------------------------------------------------
    # init
    # --------------------------------------------------------
    valid = True

    # --------------------------------------------------------
    # forward registration
    # --------------------------------------------------------
    (
        T_reg_forward,
        metric_before_forward,
        metric_before_gt_forward,
        metric_before_pred_forward,
        metric_after_forward
     ) = sitk_2d_register(
                    frame_i=frame_i,
                    frame_j=frame_j,
                    ref_transform=ref_transform,
                    gt_transform=gt_transform,
                    **sitk
                    )

    # check validity
    eps = 1e-12 # prevent zero divs
    metric_change_forward = (abs(metric_after_forward - metric_before_forward) / (abs(metric_before_forward) + eps)) * 100.0
    valid = (metric_change_forward <= max_metric_change)
    valid = True

    # --------------------------------------------------------
    # cross check, backwards registration
    # --------------------------------------------------------
    if cross_check:

        T_dl_backwards = np.linalg.inv(ref_transform)
        (
            _,
            metric_before_backward,
            _,
            _,
            metric_after_backward,
        ) = sitk_2d_register(
            frame_i=frame_j,
            frame_j=frame_i,
            ref_transform=T_dl_backwards,
            gt_transform=gt_transform,
            **sitk
        )

        # check validity
        metric_change_backward = (abs(metric_after_backward- metric_before_backward) / (abs(metric_before_backward) + eps)) * 100.0
        valid = (metric_change_backward <= max_metric_change)


    # --------------------------------------------------------
    # build registration transform
    # --------------------------------------------------------
    T_fused = fuse_3dof_with_6dof(ref_transform, T_reg_forward)
    confidence = max(0.0, 1.0 - metric_change_forward/100)

    return [
        T_fused,
        confidence,
        valid,
        metric_before_forward,
        metric_before_gt_forward,
        metric_before_pred_forward,
        metric_after_forward
    ]


def register_3d(window: np.ndarray,
                pred_acc: np.ndarray,
                gt_acc: np.ndarray,
                sitk_cfg):

    window_size = len(window)
    center = window_size // 2

    volume_frames = np.delete(window, center, axis=0)
    pred_first_inverse = np.linalg.inv(pred_acc[0])
    gt_first_inverse = np.linalg.inv(gt_acc[0])

    volume_poses = np.delete(pred_acc, center, axis=0) @ pred_first_inverse # normalize by first pose to make relative

    slice_frame = window[center]
    slice_frame_pose = pred_acc[center] @ pred_first_inverse
    slice_frame_pose_gt = gt_acc[center] @ gt_first_inverse
    (
        T_reg,
        metric_before_forward,
        metric_before_gt_forward,
        metric_before_pred_forward,
        metric_after_forward,
    ) = sitk_3d_register(
        volume_frames=volume_frames,
        volume_poses=volume_poses,
        slice_frame=slice_frame,
        slice_frame_pose=slice_frame_pose,
        slice_frame_pose_gt=slice_frame_pose_gt,
        **sitk_cfg,
    )

    # compute global pose
    T_reg_global = pred_acc[0] @ T_reg

    return (
        T_reg_global,
        metric_before_forward,
        metric_before_gt_forward,
        metric_before_pred_forward,
        metric_after_forward,
    )

def fuse_3dof_with_6dof(T_ref: np.ndarray, T_reg_se2: np.ndarray) -> np.ndarray:

    T_fused = T_ref.copy()

    # extract translation
    T_fused[0, 3] = T_reg_se2[0, 2]
    T_fused[1, 3] = T_reg_se2[1, 2]

    # extract yaw from registration
    yaw = np.arctan2(
        T_reg_se2[1, 0],
        T_reg_se2[0, 0],
    )

    # extract roll/pitch from reference
    R_ref = Rotation.from_matrix(T_ref[:3, :3])
    roll, pitch, _ = R_ref.as_euler("xyz")

    # build new rotation
    R_fused = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T_fused[:3, :3] = R_fused

    return T_fused


# TODO: Add random sampling
def sample_pairs_by_step(
    frames: np.ndarray,
    acc_transforms_all: np.ndarray,
    gt_acc_transforms_all: np.ndarray,
    step_size: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n = len(frames)# - 1 # last frame has no inbetween transform to next frame
    if step_size > n or n < 2:
        raise ValueError(
            f"Invalid number of frames: {n}"
        )

    # sample pairs of frames given step size
    idc1 = np.arange(0, n - step_size, step_size)
    idc2 = idc1 + step_size
    
    last_frame = n - 1

    if idc2[-1] != last_frame:
        idc1 = np.concatenate([idc1, [idc2[-1]]])
        idc2 = np.concatenate([idc2, [last_frame]])

    ref_transforms = []
    gt_transforms = []

    # get relative transforms between (potentially non-adjacent) frames
    for i in range(len(idc1)):
        ref_transforms.append(np.linalg.inv(acc_transforms_all[idc1[i]]) @ acc_transforms_all[idc2[i]])
        gt_transforms.append(np.linalg.inv(gt_acc_transforms_all[idc1[i]]) @ gt_acc_transforms_all[idc2[i]])
    
    # put identity matrix as first element for consistency
    ref_transforms = np.concatenate((np.eye(4)[None, :, :], np.asarray(ref_transforms)), axis=0)
    gt_transforms = np.concatenate((np.eye(4)[None, :, :], np.asarray(gt_transforms)), axis=0)

    return (
        idc1,
        idc2,
        np.array(ref_transforms),
        np.array(gt_transforms),
    )


def sample_sliding_windows():

    # TODO
    pass