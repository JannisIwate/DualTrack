import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from pose_graph_optimization.utils import accumulate
from pose_graph_optimization.utils import pose3_to_se2
from scipy.spatial.transform import Rotation
from src.submission.tus_rec_challenge_baseline import transform


def register(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    ref_transform: np.ndarray,
    gt_transform: np.ndarray,
    max_metric_change: float = 20,
    cross_check: bool = False,
    metric: str = "mi",
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
     ) = itk_register(frame_i=frame_i,
                    frame_j=frame_j,
                    ref_transform=ref_transform,
                    gt_transform=gt_transform,
                    metric=metric)

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
            metric=metric,
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
    ORIGIN_X = -73.28984642 # origin of pixel coord system is upper left corner, so negativ values for center of image
    ORIGIN_Y = -52.92463589

    fixed = sitk.GetImageFromArray(frame_i.astype(np.float32)) # 480x640 (x, y), other way round for sitk!
    moving = sitk.GetImageFromArray(frame_j.astype(np.float32))

    fixed.SetSpacing((SPACING_X, SPACING_Y))
    moving.SetSpacing((SPACING_X, SPACING_Y))

    fixed.SetOrigin((ORIGIN_X, ORIGIN_Y))
    moving.SetOrigin((ORIGIN_X, ORIGIN_Y))

    ## registration
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
    
    # mask to account for non-changing background
    mask = frame_i > 0
    mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask.CopyInformation(fixed) # copy meta data
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=1e-4, numberOfIterations=200)
    registration.SetOptimizerScalesFromPhysicalShift() # balance translation and rotation
    registration.SetMetricFixedMask(mask)
    registration.SetMetricMovingMask(mask)

    # multi-resolution (perform registration at different resolutions)
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

    # set initial transform to identity and evaluate
    initial.SetTranslation((0, 0))
    initial.SetAngle(0)
    registration.SetInitialTransform(initial)
    metric_before_identity = registration.MetricEvaluate(fixed, moving)

    # set initial transform to pred transform ref and evaluate
    x, y, yaw = pose3_to_se2(ref_transform)
    initial.SetTranslation((float(x), float(y)))
    initial.SetAngle(float(yaw))
    registration.SetInitialTransform(initial)
    metric_before_pred = registration.MetricEvaluate(fixed, moving)


    # --------------------------------------------------------
    # registration
    # --------------------------------------------------------

    # register
    transform_reg = registration.Execute(fixed, moving) # SimpleITK finds transform i+1->i!
    transform_reg_inv = np.linalg.inv(sitk_to_3dof(transform_reg)) # inverse as sitk finds Tj->i
    metric_after = registration.GetMetricValue()

    # registering an image on itself with identity transform: error of 0, passt
    # normal pipeline: rather large error, no great improvement even though images look fine
    # sometimes there is even a disimprovement?
    # calling .GetMetricValue() and .MetricEvaluate(fixed, moving) after registration yield different results??
    # .MetricEvaluate(fixed, fixed) (same image) gives way better result
    # more iterations = better results, stagnation reached very quickly
    # start from identity instead of DT transform yields better and faster results
    # mse (ideal value 0.0): gets better, performance is worse, fastest
    # mi (ideal value 0.0): gets worse, performance is worse, takes some time
    # corr (ideal value -1.0): gets slightly better, performance is worse, takes some time
    # smaller value range for less distance between image
    # GT transform does not mean zero error as images are not only moved but whole scene
    # IR is "too good", MSE is better with IR then with GT transform
    # IR takes longer the closer frames are (Why?)
    # errors happen near the transducer and along long edges
    # transforms found by IR differ greatly from GT, even though images look fine

    # check result
    test_transform = sitk.Euler2DTransform()
    test_transform.SetTranslation((0.0, -10.0))  # 10 mm nach oben
    registered_image_gt = sitk.Resample(
        moving,
        fixed,
        _6dof_to_sitk(np.linalg.inv(gt_transform)), # gt transform is Ti->j, sitk is Tj->i
        sitk.sitkLinear,
        0.0
    )
    registered_image_reg = sitk.Resample(
        moving,
        _6dof_to_sitk(transform_reg_inv),
        sitk.sitkLinear,
        0.0
    )

    image_plot(fixed, title="fixed")
    image_plot(moving, title="moving")
    image_plot(registered_image_gt, title="gt transform")
    image_plot(registered_image_reg, title="ir transform")
    #plt.show()

    return (
        transform_reg_inv,
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
    list[np.ndarray],
    list[np.ndarray],
]:
    n = len(frames)
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
    return (
        idc1,
        idc2,
        ref_transforms,
        gt_transforms,
    )