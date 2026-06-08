import numpy as np
import SimpleITK as sitk
from pose_graph_optimization.utils import pose3_to_se2


def register_old(frame_i, frame_j):

    # dummy implementation for now
    # in practice, this would involve feature matching and pose estimation
    transform = np.eye(4)

    return transform


def register(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    transform: np.ndarray
):

    # init frames
    fixed = sitk.GetImageFromArray(frame_i.astype(np.float32))
    moving = sitk.GetImageFromArray(frame_j.astype(np.float32))

    # init initial transformation (tried to fix ir failing (All samples map outside moving image buffer) by initializing with DL T but didn't work)
    # maybe try elastix IR (should be more robust)
    # look at units of DL Ts, pixel to mm, init of coordinate system
    x, y, yaw = pose3_to_se2(transform)
    initial = sitk.Euler2DTransform()
    initial.SetTranslation(
        (float(x), float(y))
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

    # register
    transform = registration.Execute(fixed, moving)

    # build transformation matrix
    angle = transform.GetAngle()
    tx, ty = transform.GetTranslation()

    c = np.cos(angle)
    s = np.sin(angle)

    T = np.array([
            [c, -s, tx],
            [s,  c, ty],
            [0,  0,  1],
            ],
            dtype=np.float64
            )

    # get confidence
    metric = registration.GetMetricValue()

    confidence = 1.0 / (1.0 + abs(metric))

    return T, confidence