import numpy as np
import SimpleITK as sitk
from pose_graph_optimization.utils import pose3_to_se2
from scipy.spatial.transform import Rotation


def register(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    transform: np.ndarray,
    metric = "mi",
    max_metric_change_percentage: float = 20.0,
):

    # init frames
    fixed = sitk.GetImageFromArray(frame_i.astype(np.float32))
    moving = sitk.GetImageFromArray(frame_j.astype(np.float32))

    fixed.SetSpacing((0.22938919, 0.22097969)) # values from DualTrack repo
    moving.SetSpacing((0.22938919, 0.22097969))

    fixed.SetOrigin((-73.28984642, -52.92463589))
    moving.SetOrigin((-73.28984642, -52.92463589))

    # init transform from DL pose
    x, y, yaw = pose3_to_se2(transform)

    initial = sitk.Euler2DTransform()

    initial.SetTranslation(
        (float(x) / 0.22938919, float(y) / 0.22097969)
    )
    initial.SetAngle(float(yaw))

    # registration setup
    registration = sitk.ImageRegistrationMethod()

    #metric = metric.upper()

    if metric is "mi":
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    elif metric is "corr":
        registration.SetMetricAsCorrelation()

    elif metric is "mse":
        registration.SetMetricAsMeanSquares()

    else:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            "Use 'mi', 'corr' or 'mse'."
        )

    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=200,
    )

    registration.SetInitialTransform(initial)

    # metric before optimization
    metric_before = registration.MetricEvaluate(fixed, moving)

    print(
        f"[{metric}] score before reg: "
        f"{metric_before:.6f}"
    )

    # optimize
    T_px_reg = registration.Execute(fixed, moving)

    metric_after = registration.GetMetricValue()

    print(
        f"[{metric}] score after reg: "
        f"{metric_after:.6f}"
    )

    # compare before/after metric
    eps = 1e-12 # prevent zero divs

    metric_change_percent = (abs(metric_after - metric_before) / (abs(metric_before) + eps)) * 100.0

    rating = metric_change_percent <= max_metric_change_percentage

    print(
        f"[{metric}] metric change: "
        f"{metric_change_percent:.2f}% "
        f"(threshold={max_metric_change_percentage:.2f}%) "
        f"-> rating={rating}"
    )

    # build registration transform
    angle = T_px_reg.GetAngle()
    tx, ty = T_px_reg.GetTranslation()

    c = np.cos(angle)
    s = np.sin(angle)

    T = np.array(
        [
            [c, -s, tx],
            [s,  c, ty],
            [0,  0,  1],
        ],
        dtype=np.float64,
    )

    T_fused = fuse_registration_with_pose(np.array(transform), T)

    confidence = max(0.0, 1.0 - metric_change_percent / max_metric_change_percentage)

    return T_fused, confidence, rating


def fuse_registration_with_pose(
    T_ref: np.ndarray,
    T_reg_se2: np.ndarray,
):
    T_fused = T_ref.copy()

    # translation, keep z from DL
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

    # build new rotation (replace init yaw with reg yaw)
    R_fused = Rotation.from_euler(
        "xyz",
        [roll, pitch, yaw],
    ).as_matrix()

    T_fused[:3, :3] = R_fused

    return T_fused