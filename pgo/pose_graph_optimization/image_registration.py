import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from pose_graph_optimization.utils import accumulate
from pose_graph_optimization.utils import pose3_to_se2
from scipy.spatial.transform import Rotation


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
    ## init
    valid = True
    
    ## forward registration
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

    ## cross check, backwards registration
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

    ## build registration transform
    T_fused = fuse_registration_with_pose(ref_transform, T_reg_forward)
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
    ## init
    # images
    SPACING_X = 0.22938919 # values from TUSREC, mm per pixel (same for TUSREC24 and 25)
    SPACING_Y = 0.22097969
    ORIGIN_X = -73.28984642
    ORIGIN_Y = -52.92463589

    fixed = sitk.GetImageFromArray(frame_i.astype(np.float32))
    moving = sitk.GetImageFromArray(frame_j.astype(np.float32))

    fixed.SetSpacing((SPACING_X, SPACING_Y))
    moving.SetSpacing((SPACING_X, SPACING_Y))

    fixed.SetOrigin((ORIGIN_X, ORIGIN_Y))
    moving.SetOrigin((ORIGIN_X, ORIGIN_Y))

    # registration
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
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=1e-4, numberOfIterations=15)

    # def iteration_callback():
    #     print(
    #         f"Iteration: {registration.GetOptimizerIteration():3d}, "
    #         f"Metric: {registration.GetMetricValue():.6f}"
    #     )

    # registration.AddCommand(
    #     sitk.sitkIterationEvent,
    #     iteration_callback
    # )

    # initial
    initial = sitk.Euler2DTransform()
    initial = sitk.CenteredTransformInitializer( # set center to image center (though this is implicitely achieved by the values of origin and spacing)
        fixed,
        moving,
        initial,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # set initial transform to pred transform ref and evaluate
    x, y, yaw = pose3_to_se2(ref_transform)
    initial.SetTranslation((float(x), float(y)))
    initial.SetAngle(float(yaw))
    registration.SetInitialTransform(initial)
    metric_before_pred = registration.MetricEvaluate(fixed, moving)

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

    ## register
    transform_reg = registration.Execute(fixed, moving)
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

    ## check result
    registered_image = sitk.Resample(
        moving,
        fixed,
        transform_reg,
        sitk.sitkLinear,
        0.0
    )
    error = sitk.SquaredDifference(moving, registered_image)
    error_np = sitk.GetArrayFromImage(error)

    x = 170
    y = 90

    moving_val = moving[x, y]
    reg_val = registered_image[x, y]

    # print(moving_val)
    # print(reg_val)
    # print((moving_val - reg_val) ** 2)
    # print(error_np[y, x])

    # plt.imshow(error_np, cmap="hot")
    # plt.colorbar()
    # image_plot(moving, title="moving")
    # image_plot(registered_image, title="image after IR transform")
    # plt.show()
    #breakpoint()

    # image_plot(fixed, title="fixed")
    # image_plot(moving, title="moving")
    # image_plot(registered_image_init, title="init transform")
    # plt.show()
    
    T_reg = sitk_to_3dof(transform_reg)

    return (
        T_reg,
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


def fuse_registration_with_pose(T_ref: np.ndarray, T_reg_se2: np.ndarray) -> np.ndarray:

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

    idc1 = np.arange(0, n - step_size, step_size)
    idc2 = idc1 + step_size

    last_frame = n - 1

    if idc2[-1] != last_frame:
        idc1 = np.concatenate([idc1, [idc2[-1]]])
        idc2 = np.concatenate([idc2, [last_frame]])

    ref_transforms = []
    gt_transforms = []

    for i in range(len(idc1)):
        ref_transforms.append(np.linalg.inv(acc_transforms_all[idc1[i]]) @ acc_transforms_all[idc2[i]])
        gt_transforms.append(np.linalg.inv(gt_acc_transforms_all[idc1[i]]) @ gt_acc_transforms_all[idc2[i]])

    return (
        idc1,
        idc2,
        ref_transforms,
        gt_transforms,
    )