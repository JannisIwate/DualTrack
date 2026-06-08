import numpy as np
import SimpleITK as sitk


def register_old(frame_i, frame_j):

    # dummy implementation for now
    # in practice, this would involve feature matching and pose estimation
    transform = np.eye(4)

    return transform


def register(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
):

    fixed = sitk.GetImageFromArray(frame_i.astype(np.float32))

    moving = sitk.GetImageFromArray(frame_j.astype(np.float32))

    initial = sitk.Euler2DTransform()

    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=200,
    )

    registration.SetInitialTransform(initial)

    transform = registration.Execute(fixed,moving)

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

    metric = registration.GetMetricValue()

    confidence = 1.0 / (1.0 + abs(metric))

    return T, confidence