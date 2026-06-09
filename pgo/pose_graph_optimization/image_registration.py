import numpy as np
import SimpleITK as sitk
from pose_graph_optimization.utils import pose3_to_se2
from scipy.spatial.transform import Rotation


def register_old(frame_i, frame_j):

    # dummy implementation for now
    # in practice, this would involve feature matching and pose estimation
    transform = np.eye(4)

    return transform


def register(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    transform: np.ndarray,
    pixel_to_image
):

    # init frames
    fixed = sitk.GetImageFromArray(frame_i.astype(np.float32))
    moving = sitk.GetImageFromArray(frame_j.astype(np.float32))

    fixed.SetSpacing(
    (
        0.22938919,
        0.22097969
    )
    )

    moving.SetSpacing(
        (
            0.22938919,
            0.22097969
        )
    )
    fixed.SetOrigin(
        (-73.28984642, -52.92463589)
    )

    moving.SetOrigin(
        (-73.28984642, -52.92463589)
    )

    # print("fixed size:", fixed.GetSize())
    # print("moving size:", moving.GetSize())

    # print("fixed origin:", fixed.GetOrigin())
    # print("moving origin:", moving.GetOrigin())

    # print("fixed spacing:", fixed.GetSpacing())
    # print("moving spacing:", moving.GetSpacing())

    # init initial transformation (tried to fix ir failing (All samples map outside moving image buffer) by initializing with DL T but didn't work)
    # maybe try elastix IR (should be more robust)
    # look at units of DL Ts, pixel to mm, init of coordinate system
    x, y, yaw = pose3_to_se2(transform)
    initial = sitk.Euler2DTransform()
    # initial.SetTranslation(
    #     ((float(x) - 73.28984642) / 0.22938919, (float(y) - 52.92463589) / 0.22097969)
    # )
    initial.SetTranslation(
        (float(x) / 0.22938919, float(y) / 0.22097969)
    )
    initial.SetAngle(
        float(yaw)
    )

    # set up registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=200,
    )
    registration.SetInitialTransform(initial)
    # registration.AddCommand(
    #     sitk.sitkIterationEvent,
    #     lambda: print(
    #         registration.GetOptimizerPosition()
    #     )
    # )

    # register
    T_px_reg = registration.Execute(fixed, moving)

    # build transformation matrix
    angle = T_px_reg.GetAngle()
    tx, ty = T_px_reg.GetTranslation()

    c = np.cos(angle)
    s = np.sin(angle)

    T = np.array([
            [c, -s, tx],
            [s,  c, ty],
            [0,  0,  1],
            ],
            dtype=np.float64
            )
    
    T_fused = fuse_registration_with_pose(
        np.array(transform),
        T,
    )

    # get confidence
    metric = registration.GetMetricValue()

    confidence = 1.0 / (1.0 + abs(metric))

    return T_fused, confidence


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