import numpy as np
import SimpleITK as sitk
import torch
from pose_graph_optimization.utils import pose3_to_se2
from scipy.spatial.transform import Rotation


def register(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    transform: np.ndarray,
    max_metric_change=20,
    max_cross_change=1,
    cross_checking=True,
    metric="mi"
):
    # init
    SPACING_X = 0.22938919 # values from TUSREC, mm per pixel (same for TUSREC24 and 25)
    SPACING_Y = 0.22097969
    ORIGIN_X = -73.28984642
    ORIGIN_Y = -52.92463589
    rating = True

    ## registration init
    fixed = sitk.GetImageFromArray(frame_i.astype(np.float32))
    moving = sitk.GetImageFromArray(frame_j.astype(np.float32))

    fixed.SetSpacing((SPACING_X, SPACING_Y))
    moving.SetSpacing((SPACING_X, SPACING_Y))

    fixed.SetOrigin((ORIGIN_X, ORIGIN_Y))
    moving.SetOrigin((ORIGIN_X, ORIGIN_Y))

    x, y, yaw = pose3_to_se2(transform)

    initial = sitk.Euler2DTransform()
    initial = sitk.CenteredTransformInitializer( # set center to image center (though this is implicitely achieved by the values of origin and spacing)
        fixed,
        moving,
        initial,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    initial.SetTranslation((float(x), float(y)))
    initial.SetAngle(float(yaw))

    registration = sitk.ImageRegistrationMethod()

    if metric == "mi":
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    elif metric == "corr":
        registration.SetMetricAsCorrelation()

    elif metric == "mse":
        registration.SetMetricAsMeanSquares()

    else:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            "Use 'mi', 'corr' or 'mse'."
        )

    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=200)

    registration.SetInitialTransform(initial)

    ## register
    metric_before = registration.MetricEvaluate(fixed, moving)
    T_reg = registration.Execute(fixed, moving)

    metric_after = registration.GetMetricValue()

    ## compare metrics
    eps = 1e-12 # prevent zero divs
    metric_change = (abs(metric_after - metric_before) / (abs(metric_before) + eps)) * 100.0

    if rating:
        rating = (metric_change <= max_metric_change)

    ## additional checks
    if cross_checking:

        # start from inverse DL transform
        x, y, yaw = pose3_to_se2(np.linalg.inv(transform))

        initial = sitk.Euler2DTransform()
        initial.SetTranslation((float(x), float(y)))
        initial.SetAngle(float(yaw))

        registration.SetInitialTransform(initial)

        # register
        fixed_temp = fixed
        fixed = moving
        moving = fixed_temp

        T_cross = registration.Execute(fixed, moving)

        # compare reg transform and reverse reg transform
        T_error = itk_to_3dof(T_reg) @ itk_to_3dof(T_cross)

        translation_error = np.linalg.norm(T_error[:2, 2])
        rotation_error = np.rad2deg(np.arctan2(T_error[1, 0], T_error[0, 0]))

        if rating:
            if translation_error >= max_cross_change or rotation_error >= max_cross_change:
                rating = False

    ## build registration transform
    T = itk_to_3dof(T_reg)
    T_fused = fuse_registration_with_pose(np.array(transform), T)
    confidence = max(0.0, 1.0 - metric_change / max_metric_change)

    # print("Image registered")

    return T_fused, confidence, rating


def fuse_registration_with_pose(T_ref: np.ndarray, T_reg_se2: np.ndarray):

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
    R_fused = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()

    T_fused[:3, :3] = R_fused

    return T_fused


def itk_to_3dof(T_itk):

    angle = T_itk.GetAngle()
    tx, ty = T_itk.GetTranslation()

    c = np.cos(angle)
    s = np.sin(angle) # equivalent to T_itk.GetMatrix()

    T = np.array(
        [
            [c, -s, tx],
            [s,  c, ty],
            [0,  0,  1],
        ],
        dtype=np.float64,
    )

    return T


def sample_random_pairs(transforms, num_pairs):

    n = len(transforms)

    if n < 2:
        raise ValueError("Need at least two transforms.")
    elif num_pairs > n // 2:
        raise ValueError(
            "num_pairs cannot exceed n//2."
        )
    else:
        perm = torch.randperm(n)
        idc1 = perm[:num_pairs]
        idc2 = perm[num_pairs:2 * num_pairs]

    return (
        idc1,
        idc2,
        transforms[idc1],
        transforms[idc2]
    )


def sample_pairs_by_step(transforms, step_size):

    n = len(transforms)

    idc1 = torch.arange(0, n - step_size, step_size)
    idc2 = idc1 + step_size

    last_frame = n - 1

    if idc2[-1] != last_frame:
        idc1 = torch.cat([idc1, idc2[-1].unsqueeze(0)])
        idc2 = torch.cat([idc2, torch.tensor([last_frame])])

    return (
        idc1,
        idc2,
        transforms[idc1],
        transforms[idc2]
    )