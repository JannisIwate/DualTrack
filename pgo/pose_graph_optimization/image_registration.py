import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from pose_graph_optimization.utils import accumulate
from pose_graph_optimization.utils import pose3_to_se2
from scipy.spatial.transform import Rotation
from src.submission.tus_rec_challenge_baseline import transform


def register(
    sitk,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    ref_transform: np.ndarray,
    gt_transform: np.ndarray,
    step:int = 1,
    max_metric_change: float = 20,
    cross_check: bool = False,
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
     ) = itk_register(
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

    # --------------------------------------------------------
    # cross check, backwards registration
    # --------------------------------------------------------
    if cross_check:

        T_dl_backwards = np.linalg.inv(ref_transform)
        (
            T_reg_forward,
            metric_before_forward,
            _,
            _,
            metric_after_forward,
        ) = itk_register(
            frame_i=frame_j,
            frame_j=frame_i,
            ref_transform=T_dl_backwards,
            gt_transform=gt_transform,
            **sitk
        )

        # check validity
        metric_change_forward = (abs(metric_after_forward - metric_before_forward) / (abs(metric_before_forward) + eps)) * 100.0
        valid = (metric_change_forward <= max_metric_change)


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


def itk_register(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    ref_transform: np.ndarray,
    gt_transform: np.ndarray,
    metric: str = "mi",
    optimizer:str = "gradient",
    multi_resolution = False,
    use_center = False
) -> tuple[
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    # --------------------------------------------------------
    # init
    # --------------------------------------------------------

    ## images
    SPACING_X = 0.22938919 # values from TUSREC, mm per pixel (same for TUSREC24 and 25), needed so that transform is in mm and not pixels
    SPACING_Y = 0.22097969
    ORIGIN_X = -73.28984642 # origin of pixel coord system is upper left corner, so negative values for center of image
    ORIGIN_Y = -52.92463589

    fixed = sitk.GetImageFromArray(frame_i.astype(np.float32)) # 480x640 (x, y), other way round for sitk!
    moving = sitk.GetImageFromArray(frame_j.astype(np.float32))

    # use center part of image which is not as affected as rim by pitch and roll
    if use_center:

        # image_plot(fixed, title="fixed before")
        # image_plot(moving, title="moving before")

        roi_size, roi_index = get_center_roi_params(fixed.GetSize(), 0.5)

        fixed = sitk.RegionOfInterest(
            fixed,
            size=roi_size,
            index=roi_index,
        )
        moving = sitk.RegionOfInterest(
            moving,
            size=roi_size,
            index=roi_index,
        )
        # image_plot(fixed, title="fixed after")
        # image_plot(moving, title="moving after")
        # plt.show()
        # breakpoint()
        
    fixed.SetSpacing((SPACING_X, SPACING_Y))
    moving.SetSpacing((SPACING_X, SPACING_Y))

    fixed.SetOrigin((ORIGIN_X, ORIGIN_Y))
    moving.SetOrigin((ORIGIN_X, ORIGIN_Y))

    ## registration
    registration = sitk.ImageRegistrationMethod()

    # metric
    if metric == "mi": # gives best results so far
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) # as it is negative: lower is better
    elif metric == "corr":
        registration.SetMetricAsCorrelation()
    elif metric == "mse":
        registration.SetMetricAsMeanSquares()
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            "Use 'mi', 'corr' or 'mse'."
        )
    
    # mask to account for non-changing background
    mask = fixed > 0
    registration.SetMetricFixedMask(mask)
    registration.SetMetricMovingMask(mask)

    # interpolator and optimizer
    registration.SetInterpolator(sitk.sitkLinear)

    if optimizer == "gradient":
        registration.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=1e-4, numberOfIterations=200)
        registration.SetOptimizerScalesFromPhysicalShift() # balance translation and rotation
    elif optimizer == "exhaustive": # takes too long
        registration.SetOptimizerAsExhaustive([10, 10, 10])
        registration.SetOptimizerScales([np.deg2rad(0.005), 1/20, 1/20])
    elif optimizer == "amoeba": # takes too long
        registration.SetOptimizerAsAmoeba(
            simplexDelta=1.0,
            numberOfIterations=200,
            parametersConvergenceTolerance=1e-6,
            functionConvergenceTolerance=1e-4,
            withRestarts=True
        )
    else:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. "
            "Use 'gradient' or 'exhaustive'."
        )

    # multi-resolution (perform registration at different resolutions)
    if multi_resolution:
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # iteration callback (for debugging)
    def iteration_callback():
        print(
            f"Iteration: {registration.GetOptimizerIteration():3d}, "
            f"Metric: {registration.GetMetricValue():.6f}"
        )

    # registration.AddCommand(
    #     sitk.sitkIterationEvent,
    #     iteration_callback
    # )

    # initial transform
    initial = sitk.Euler2DTransform()
    initial = sitk.CenteredTransformInitializer( # set center to image center (though this is implicitely achieved by the values of origin and spacing, gt has center at image center)
        fixed,
        moving,
        initial,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # set initial transform to gt transform and evaluate
    x, y, yaw = pose3_to_se2(gt_transform)
    initial.SetTranslation((float(x), float(y)))
    initial.SetAngle(float(yaw))
    registration.SetInitialTransform(initial)
    metric_before_gt = registration.MetricEvaluate(fixed, moving)

    # set initial transform to pred transform ref and evaluate
    x, y, yaw = pose3_to_se2(ref_transform)
    initial.SetTranslation((float(x), float(y)))
    initial.SetAngle(float(yaw))
    registration.SetInitialTransform(initial)
    metric_before_pred = registration.MetricEvaluate(fixed, moving)

    # set initial transform to identity and evaluate
    initial.SetTranslation((0, 0))
    initial.SetAngle(0)
    registration.SetInitialTransform(initial)
    metric_before_identity = registration.MetricEvaluate(fixed, moving)


    # --------------------------------------------------------
    # registration
    # --------------------------------------------------------

    # register
    transform_reg = registration.Execute(fixed, moving)
    transform_reg_inv = np.linalg.inv(sitk_to_3dof(transform_reg)) # inverse as sitk finds Tj->i
    metric_after = registration.GetMetricValue()

    # check images
    # image_plot(fixed, title="fixed")
    # image_plot(fixed, title="fixed")

    # image_plot(moving, title="moving")
    # image_plot(moving, title="moving")

    # registered_image_ir = sitk.Resample(
    #     moving,
    #     fixed,
    #     transform_reg,
    #     sitk.sitkLinear,
    #     0.0
    # )
    # image_plot(registered_image_ir, title="ir transform")
    # print(transform_reg)
    # plt.show()
    # breakpoint()

    return (
        #transform_reg_inv, # Was jetzt?
        sitk_to_3dof(transform_reg),
        float(metric_before_identity),
        float(metric_before_gt),
        float(metric_before_pred),
        float(metric_after),
)


def image_plot(img, title=None, margin=0.05, dpi=80): # img is sitk image
        
        nda = sitk.GetArrayViewFromImage(img)
        spacing = img.GetSpacing()

        ysize = nda.shape[0]
        xsize = nda.shape[1]

        figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

        fig = plt.figure(title, figsize=figsize, dpi=dpi)
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        extent = (0, xsize * spacing[1], 0, ysize * spacing[0])

        t = ax.imshow(
            nda, extent=extent, interpolation="hamming", cmap="gray", origin="upper"
        )

        if title:
            plt.title(title)


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


def sitk_to_3dof(T_itk) -> np.ndarray:

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


def _6dof_to_sitk(T_itk):

    angle = np.arctan2(T_itk[1, 0], T_itk[0, 0])
    tx = T_itk[0, 3]
    ty = T_itk[1, 3]

    transform = sitk.Euler2DTransform()
    transform.SetAngle(angle)
    transform.SetTranslation((tx, ty))

    return transform


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
    n = len(frames) - 1 # last frame has no inbetween transform to next frame
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

def get_center_roi_params(size, fraction):

    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in the range (0, 1].")

    roi_size = [max(1, int(round(s * fraction))) for s in size]
    roi_index = [(s - rs) // 2 for s, rs in zip(size, roi_size)]

    return roi_size, roi_index