import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from pose_graph_optimization.utils import accumulate
from pose_graph_optimization.utils import pose3_to_se2
from pose_graph_optimization.defines import *
import pyvista as pv
import pyvista as pv
import itertools
import time



def sitk_2d_register(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    ref_transform: np.ndarray,
    gt_transform: np.ndarray,
    metric: str = "mi",
    optimizer:str = "gradient",
    options: str = "",
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

    options = options or ""

    ## images
    fixed = sitk.GetImageFromArray(frame_i.astype(np.float32)) # 640x480 (x, y), other way round for sitk!
    moving = sitk.GetImageFromArray(frame_j.astype(np.float32))

    # use center part of image which is not as affected as outer part by pitch and roll
    if "use_center" in options: # -> Verbesserung von 233% fuer FDR, 240% fuer GPE, 260% fuer Ausfuehrungszeit)

        # image_plot(fixed, title="fixed before")
        # plt.show()
        # breakpoint()
        # image_plot(moving, title="moving before")

        roi_size, roi_index = get_center_roi_params(fixed.GetSize(), (0.5, 0.5))

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
        
    fixed.SetSpacing((SPACING_X, SPACING_Y))
    moving.SetSpacing((SPACING_X, SPACING_Y))

    fixed.SetOrigin((ORIGIN_X, ORIGIN_Y)) # origin in SITK is not origin in DualTrack! Also, origin only has minimal effect when it is the same for both
    # (still it makes sense to leave it as is as CenteredTransformInitializer computes R center from this)
    moving.SetOrigin((ORIGIN_X, ORIGIN_Y))
    # image_plot(fixed, title="fixed before")
    # plt.show()
    # breakpoint()

    ## registration
    registration = build_registration_object(metric, optimizer, options)

    # mask to account for non-changing background
    mask = fixed > 0
    
    registration.SetMetricFixedMask(mask)
    registration.SetMetricMovingMask(mask)

    # extend mask to parts of image which differ to much (spawning feature)
    if "patch_mask" in options: # -> Verbesserung im Vergleich zu nur Centering von 5% FDR, keine signifikante Verbesserung von GPE, 1.5% fuer Ausfuehrungszeit

        mask = get_mask_from_patches(mask, fixed, moving, ref_transform, 4, 0.7) # 4 and 0.7 turn out to be ideal

    mask.CopyInformation(fixed)
    # image_plot(mask, title="mask")
    # image_plot(fixed, title="fixed")
    # image_plot(moving, title="moving")
    # plt.show()
    # breakpoint()

    registration.SetMetricFixedMask(mask)
    registration.SetMetricMovingMask(mask)

    # initial transform
    initial = sitk.CenteredTransformInitializer( # set center to image center (though this is implicitely achieved by the values of origin and spacing, gt has center at image center)
        fixed,
        moving,
        sitk.Euler2DTransform(),
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
        transform_reg_inv,
        float(metric_before_identity),
        float(metric_before_gt),
        float(metric_before_pred),
        float(metric_after),
    )

# TODO: viermal Slice ausprobieren
# TODO: Idee: Nimm ein Slice aus Volumen bei Frame Pose, veraendere Pose mehrmals leicht anhand von t und R Varianzen, berechne jedes Mal 2d Metrik zwischen Frame und Slice,
# Pose mit bester Metric wird zurueckgegeben

def sitk_3d_register(
    volume_frames: np.ndarray,
    volume_poses: np.ndarray,
    slice_frame: np.ndarray,
    slice_frame_pose: np.ndarray,
    slice_frame_pose_gt: np.ndarray,
    metric: str = "mi",
    optimizer: str = "gradient",
    options: str = "",
    transform_type: str = "versor",
) -> tuple[
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    all_start = time.time()
    # init
    options = options or ""

    # build volumes
    start = time.time()
    volume, _, _ = build_volume_from_slices(volume_frames, volume_poses)
    volume_building_time = time.time() - start
    start = time.time()
    # slice_volume = slice_to_volume(slice_frame,
    #                                slice_frame_pose,
    #                                volume.GetSpacing()[2],
    #                                thickness=4)
    slice_frame = np.expand_dims(slice_frame, axis=0)
    slice_ = sitk.GetImageFromArray(slice_frame.astype(np.float32))
    slice_.SetSpacing(
        (
            SPACING_X,
            SPACING_Y,
            volume.GetSpacing()[2],
        )
    )
    slice_volume = sitk.Expand(slice_, [1, 1, 4])
    slice_building_time = time.time() - start

    fixed = slice_volume
    moving = volume

    if "use_center" in options:
        
        roi_size, roi_index = get_center_roi_params(fixed.GetSize(), (0.5, 0.5, 1.0))
        fixed = sitk.RegionOfInterest(
            fixed,
            size=roi_size,
            index=roi_index,
        )
        roi_size, roi_index = get_center_roi_params(moving.GetSize(), (0.5, 0.5, 1.0))
        moving = sitk.RegionOfInterest(
            moving,
            size=roi_size,
            index=roi_index,
        )

    # init IR
    registration = build_registration_object(metric, optimizer, options)

    # set initial transform to gt transform and evaluate
    initial = set_transform_from_pose(
        fixed,
        moving,
        slice_frame_pose_gt,
        transform_type,
    )
    registration.SetInitialTransform(initial)
    metric_before_gt = registration.MetricEvaluate(
        fixed,
        moving,
    )

    # set initial transform to identity and evaluate
    initial = set_transform_from_pose(
            fixed,
            moving,
            np.eye(4),
            transform_type,
        )
    registration.SetInitialTransform(initial)
    metric_before_identity = registration.MetricEvaluate(
        fixed,
        moving,
    )

    # set initial transform to predicted transform and evaluate
    initial = set_transform_from_pose(
        fixed,
        moving,
        slice_frame_pose,
        transform_type,
    )
    registration.SetInitialTransform(initial)
    metric_before_pred = registration.MetricEvaluate(
        fixed,
        moving,
    )

    # register
    start = time.time()
    transform_reg = registration.Execute(
        fixed=fixed,
        moving=moving,
    )
    reg_time = time.time() - start
    transform_reg = sitk_to_6dof(transform_reg)
    metric_after = registration.GetMetricValue()

    # slice_volume_after = slice_to_volume(slice_frame,
    #                                 transform_reg,
    #                                 volume.GetSpacing()[2],
    #                                 thickness=4)

    # slice_volume_after = sitk.GetImageFromArray(slice_frame.astype(np.float32))
    # slice_volume_after.SetSpacing(
    #     (
    #         SPACING_X,
    #         SPACING_Y,
    #         volume.GetSpacing()[2],
    #     )
    # )
    # slice_volume = sitk.Expand(slice_, [1, 1, 4])
    
    show_volumes([volume, slice_volume], outline_colors=["red", "black"])
    # breakpoint()
    all_time = time.time() - all_start
    # print(f"Total time: {all_time}")
    # print(f"vol building: {volume_building_time}, {(volume_building_time / all_time) * 100}%") # -> heavily fluctuates
    # print(f"slice building: {slice_building_time}, {(slice_building_time / all_time) * 100}%") # -> takes roughly the same time every try, very little time
    # print(f"reg: {reg_time}, {(reg_time / all_time) * 100}%") # -> heavily fluctuates
    # breakpoint()

    return (
            transform_reg, # global pose
            float(metric_before_identity),
            float(metric_before_gt),
            float(metric_before_pred),
            float(metric_after),
        )


def show_volumes(
    volumes: list[sitk.Image],
    opacities=None,
    outline_colors=None,
):
    if len(volumes) == 0:
        raise ValueError("At least one volume must be provided.")

    if opacities is None:
        opacities = [0.25] * len(volumes)

    if len(opacities) != len(volumes):
        raise ValueError("Need one opacity per volume.")

    def sitk_to_grid(img: sitk.Image):

        arr = sitk.GetArrayFromImage(img)

        grid = pv.ImageData()
        grid.dimensions = np.array(arr.shape[::-1])
        grid.spacing = img.GetSpacing()
        grid.origin = img.GetOrigin()

        grid.point_data["values"] = arr.ravel(order="F")

        return grid

    plotter = pv.Plotter()

    for img, opacity in zip(volumes, opacities):

        grid = sitk_to_grid(img)

        plotter.add_volume(
            grid,
            opacity=opacity,
        )

        plotter.add_mesh(
            grid.outline(),
            color=next(itertools.cycle(outline_colors)),
            line_width=4,
        )

    plotter.add_axes()
    plotter.show()


def set_transform_from_pose(fixed,
                            moving,
                            pose,
                            transform_type):

    R = pose[:3, :3].astype(np.float64)
    t = pose[:3, 3].astype(np.float64)

    transform_type = transform_type.lower()

    if transform_type == "versor":

        initial = sitk.VersorRigid3DTransform()

        initial.SetMatrix(correct_to_orthogonal(R).reshape(-1).tolist())
        initial.SetTranslation(t.tolist())

    elif transform_type == "affine": # affine transform?? Jannis?

        initial = sitk.AffineTransform(3)
        initial.SetMatrix(R.reshape(-1).tolist())
        initial.SetTranslation(t.tolist())

    elif transform_type == "euler":

        initial = sitk.Euler3DTransform()
        initial.SetMatrix(correct_to_orthogonal(R).reshape(-1).tolist())
        initial.SetTranslation(t.tolist())

    else:
        raise ValueError(f"Unknown transform_type '{transform_type}'")

    # center
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        initial,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    return initial


def correct_to_orthogonal(rotation_matrix):

    U, _, Vt = np.linalg.svd(rotation_matrix)
    R_corrected = U @ Vt

    if np.linalg.det(R_corrected) < 0:

        U[:, -1] *= -1
        R_corrected = U @ Vt

    return R_corrected


def slice_to_volume(
    slice: np.ndarray,
    pose: np.ndarray,
    z_spacing: float,
    thickness: int = 4,
) -> sitk.Image:

    # create four identical slices
    slice_volume_array = np.repeat(
        slice[np.newaxis].astype(np.float32),
        repeats=thickness,
        axis=0,
    )

    img = sitk.GetImageFromArray(slice_volume_array)

    # init
    img.SetSpacing(
        (
            SPACING_X,
            SPACING_Y,
            z_spacing,
        )
    )

    R = pose[:3, :3]
    t = pose[:3, 3]

    # local image coordinate system -> world
    img.SetDirection(R.reshape(-1).tolist())

    # first slice lies exactly at the given pose
    origin = [ORIGIN_X + t[0], ORIGIN_Y + t[1], t[2]] # account for pixel coords!
    origin = [float(x) for x in origin] # SetOrigin needs float instead of np.float :/
    img.SetOrigin(origin)

    return img


def build_volume_from_slices(
    frames: np.ndarray,
    poses: np.ndarray,
    volume_spacing: tuple[float, float, float] = (SPACING_X, SPACING_Y, SPACING_X),
):
    # init
    sx, sy = (SPACING_X, SPACING_Y)
    # ox, oy = (ORIGIN_X, ORIGIN_Y)
    ox, oy = (0, 0) # origin doesn't matter for volume

    n, h, w = frames.shape

    image_corners = np.array([
        [0,     0,     0, 1],
        [w - 1, 0,     0, 1],
        [0,     h - 1, 0, 1],
        [w - 1, h - 1, 0, 1],
    ])

    # convert pixel to image coords
    image_corners[:, 0] = image_corners[:, 0] * sx + ox # in image coords
    image_corners[:, 1] = image_corners[:, 1] * sy + oy

    # get corner positions in world
    world_points = []

    for pose in poses: # first is identity

        pts = (pose @ image_corners.T).T[:, :3] # points are in stored in rows, need columns
        world_points.append(pts)

    world_points = np.concatenate(world_points) # all corner points in world
    world_min = world_points.min(axis=0) # min x, y and z
    world_max = world_points.max(axis=0)

    # create empty volume
    spacing = np.asarray(volume_spacing)

    volume_size = np.ceil( # bounding box around volume
        (world_max - world_min) / spacing
    ).astype(int) + 1

    volume_np = np.zeros(
        (
            volume_size[2],
            volume_size[1],
            volume_size[0],
        ),
        dtype=np.float32,
    )

    weight_np = np.zeros_like(volume_np)

    # create image coord system based on frames meta data
    xx, yy = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    ximg = (xx * sx + ox).ravel()
    yimg = (yy * sy + oy).ravel()
    xy = np.stack([ximg, yimg], axis=-1)          # (P, 2)

    R_xy = poses[:, :3, :2]                        # (n, 3, 2)
    t    = poses[:, :3, 3]                         # (n, 3)

    # apply transforms
    world = np.matmul(xy, R_xy.transpose(0, 2, 1)) + t[:, None, :]

    # compute voxel indices of volume
    voxel = np.round((world - world_min) / spacing).astype(np.int64)
    valid = (
        (voxel[..., 0] >= 0) & (voxel[..., 0] < volume_size[0]) &
        (voxel[..., 1] >= 0) & (voxel[..., 1] < volume_size[1]) &
        (voxel[..., 2] >= 0) & (voxel[..., 2] < volume_size[2])
    )
    flat_idx = (voxel[..., 2] * volume_size[1] + voxel[..., 1]) * volume_size[0] + voxel[..., 0]
    flat_idx = flat_idx[valid]
    values = frames.reshape(n, -1)[valid]

    # assign voxel values
    total = volume_size[0] * volume_size[1] * volume_size[2]
    sums   = np.bincount(flat_idx, weights=values, minlength=total)
    counts = np.bincount(flat_idx, minlength=total)

    volume_np = sums.reshape(volume_size[2], volume_size[1], volume_size[0]).astype(np.float32)
    weight_np = counts.reshape(volume_size[2], volume_size[1], volume_size[0]).astype(np.float32)

    # average overlapping voxels
    mask = weight_np > 0
    volume_np[mask] /= weight_np[mask] # shape (z, y, x)

    # convert to SITK format 
    volume = sitk.GetImageFromArray(volume_np)
    volume.SetSpacing(volume_spacing)
    volume.SetOrigin(tuple(world_min))
    volume.SetDirection(np.eye(3).flatten())

    # fill holes
    mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_image.CopyInformation(volume)

    start = time.time()
    # volume = _fill_interior_holes(volume, mask_image) # -> braucht nicht viel Zeit gemessen an Gesamtausfuehrungszeit
    # print(f"hole filling: {time.time() - start}")

    return (
        volume,
        tuple(world_min),
        volume_spacing,
    )


def show_orthogonal_slices(volume: sitk.Image, title: str = "") -> None:
 
    arr = sitk.GetArrayFromImage(volume)  # Form: (Z, Y, X)
    z, y, x = (s // 2 for s in arr.shape)  # jeweils mittlerer Schnitt
 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(arr[z, :, :], cmap="gray")
    axes[0].set_title("Axial (z-Mitte)")
    axes[1].imshow(arr[:, y, :], cmap="gray")
    axes[1].set_title("Coronal (y-Mitte)")
    axes[2].imshow(arr[:, :, x], cmap="gray")
    axes[2].set_title("Sagittal (x-Mitte)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def build_registration_object(metric, optimizer, options):

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

    # interpolator and optimizer
    registration.SetInterpolator(sitk.sitkLinear)

    if optimizer == "gradient":

        registration.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=1e-4, numberOfIterations=50)
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

    # multi-resolution (perform registration at different resolutions) -> keine signifikante Verbesserung von FDR oder GPE, -25% fuer Ausfuehrungszeit
    # registration.SetSmoothingSigmasPerLevel([SPACING_X/2])
    # registration.SetShrinkFactorsPerLevel([1])

    if "multi_resolution" in options:

        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # iteration callback (for debugging)
    if "callback" in options:

        def iteration_callback():
            print(
                f"Iteration: {registration.GetOptimizerIteration():3d}, "
                f"Metric: {registration.GetMetricValue():.6f}"
            )
    
        registration.AddCommand(
            sitk.sitkIterationEvent,
            iteration_callback
        )

    return registration


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


def sitk_to_6dof(T_itk) -> np.ndarray:

    R = np.array(
        T_itk.GetMatrix(),
        dtype=np.float64,
    ).reshape(3, 3)

    t = np.array(
        T_itk.GetTranslation(),
        dtype=np.float64,
    )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def _6dof_to_sitk(T_itk):

    angle = np.arctan2(T_itk[1, 0], T_itk[0, 0])
    tx = T_itk[0, 3]
    ty = T_itk[1, 3]

    transform = sitk.Euler2DTransform()
    transform.SetAngle(angle)
    transform.SetTranslation((tx, ty))

    return transform


def get_center_roi_params(size, fractions):

    if len(size) != len(fractions):
        raise ValueError("size and fractions must have the same length.")

    roi_size = []
    roi_index = []

    for s, f in zip(size, fractions):

        if not (0 < f <= 1):
            raise ValueError("All fractions must be in the range (0, 1].")

        rs = max(1, int(round(s * f)))
        idx = (s - rs) // 2

        roi_size.append(rs)
        roi_index.append(idx)

    return roi_size, roi_index


def get_mask_from_patches(mask, fixed, moving, ref_transform, grid_size, threshold):

    # init
    fixed_np = sitk.GetArrayViewFromImage(fixed).astype(np.float32)
    moving_np = sitk.GetArrayViewFromImage(moving).astype(np.float32)

    mask_np = sitk.GetArrayFromImage(mask).astype(np.uint8)

    rows, cols = fixed_np.shape

    grid_y = grid_size
    grid_x = grid_size

    patch_h = rows // grid_y
    patch_w = cols // grid_x

    T = _6dof_to_sitk(ref_transform)

    for gy in range(grid_y):
        for gx in range(grid_x):
            
            # determine grid coords
            y0 = gy * patch_h
            y1 = rows if gy == grid_y - 1 else (gy + 1) * patch_h

            x0 = gx * patch_w
            x1 = cols if gx == grid_x - 1 else (gx + 1) * patch_w
            
            # get fixed patch pixel values
            patch_fixed = fixed_np[y0:y1, x0:x1]

            corners_fixed = [
                (x0, y0),
                (x1 - 1, y0),
                (x0, y1 - 1),
                (x1 - 1, y1 - 1),
            ]

            corners_moving = []

            # transform corner points of fixed
            for corner in corners_fixed:

                p_phys = fixed.TransformIndexToPhysicalPoint(corner)
                p_phys_moving = T.TransformPoint(p_phys)

                try:
                    idx = moving.TransformPhysicalPointToIndex(p_phys_moving)
                    corners_moving.append(idx)
                except RuntimeError:
                    continue
            
            # skip patch if it reaches out of image
            if len(corners_moving) != 4:

                mask_np[y0:y1, x0:x1] = 0
                continue
            
            # get integer outline/bounding box
            xs = [p[0] for p in corners_moving]
            ys = [p[1] for p in corners_moving]

            mx0 = max(0, int(np.floor(min(xs))))
            mx1 = min(cols, int(np.ceil(max(xs))) + 1)

            my0 = max(0, int(np.floor(min(ys))))
            my1 = min(rows, int(np.ceil(max(ys))) + 1)

            # skip invalid patch
            if mx1 <= mx0 or my1 <= my0:

                mask_np[y0:y1, x0:x1] = 0
                continue
            
            # get moving patch pixel values
            patch_moving = moving_np[my0:my1, mx0:mx1]
            
            # skip patch if it is not in image
            if patch_moving.size == 0:

                mask_np[y0:y1, x0:x1] = 0
                continue
            
            # adapt moving shape with fixed shape in case its dims are insufficient
            if patch_moving.shape != patch_fixed.shape:

                patch_moving = patch_moving[
                    :patch_fixed.shape[0],
                    :patch_fixed.shape[1],
                ]
                # skip if not possible
                if patch_moving.shape != patch_fixed.shape:

                    mask_np[y0:y1, x0:x1] = 0
                    continue
                    
            
            # get similarity value (here: ncc)
            pf = patch_fixed - patch_fixed.mean()
            pm = patch_moving - patch_moving.mean()

            denom = np.linalg.norm(pf) * np.linalg.norm(pm)

            if denom < 1e-6:
                similarity = 1.0
            else:
                similarity = np.sum(pf * pm) / denom

            if similarity < threshold:
                mask_np[y0:y1, x0:x1] = 0

    return sitk.GetImageFromArray(mask_np)


# def build_volume_from_tracked_slices_claude(
#     frames: np.ndarray,
#     poses: np.ndarray,
#     slice_thickness: float = 1,
#     interpolator: int = sitk.sitkLinear,
#     fill_holes: bool = True,
#     max_hole_fill_iterations: int = 50,
# ) -> sitk.Image:
#     """
#     Baut aus 2D-Frames mit bekannter Pose ein 3D-Volumen (SimpleITK Image).
 
#     Konvention (an eigene Daten anpassen, falls abweichend!):
#     - frames[i]: 2D-Graustufenbild der Form (H, W).
#     - poses[i]: 4x4 homogene Rigid-Transformationsmatrix, die einen Punkt im
#       lokalen Bildkoordinatensystem (mm, Ursprung = Pixel (0,0), x-Achse
#       entlang der Spalten, y-Achse entlang der Zeilen, z=0-Ebene) auf den
#       Punkt im Welt-/Tracker-Koordinatensystem (mm) abbildet.
#       -> poses[i] = "ImageToWorld"-Transform zum Aufnahmezeitpunkt von frames[i].
#     - pixel_spacing = (sx, sy): physikalische Pixelgroesse in mm, fuer alle
#       Frames gleich angenommen.
#     - fill_holes: wenn True (Default), werden vollstaendig umschlossene
#       Luecken zwischen Slices mit dem Wert des naechstgelegenen befuellten
#       Nachbar-Voxels aufgefuellt. Bereiche ausserhalb des tatsaechlich
#       gescannten Volumens (die den Rand des Zielrasters beruehren) bleiben
#       immer 0. Achtung: das einfache Auffuellverfahren geht von nicht-
#       negativen Intensitaeten aus (z.B. 0-255); bei Bilddaten mit negativen
#       Werten (z.B. rohe CT-Hounsfield-Units) bitte fill_holes=False setzen
#       und selbst behandeln.
#     - max_hole_fill_iterations: Sicherheitsobergrenze fuer die Anzahl der
#       Auffuell-Iterationen (jede Iteration wächst die Werte um 1 Voxel).
 
#     Rueckgabe:
#         sitk.Image (Float32, 3D, achsparallel), das rekonstruierte Volumen.
#         Wo kein Frame Daten beigesteuert hat, ist der Voxelwert 0.
 
#     Hinweis: einfache Pixel-/Voxel-Compounding-Rekonstruktion (Mittelung
#     ueberlappender Beitraege). Fuer produktionsreife Freehand-3D-Rekon-
#     struktion mit besserer Loch-/Kantenbehandlung siehe z.B. das PLUS Toolkit.
#     """
#     frames = np.asarray(frames)
#     poses = np.asarray(poses)
 
#     if frames.ndim != 3:
#         raise ValueError("frames muss die Form (N, H, W) haben (Graustufen).")
#     if poses.shape[-2:] != (4, 4):
#         raise ValueError("poses muss die Form (N, 4, 4) haben (homogene 4x4-Matrizen).")
#     if frames.shape[0] != poses.shape[0]:
#         raise ValueError("frames und poses muessen gleich viele Eintraege (N) haben.")
 
#     n_frames, h, w = frames.shape
#     sx, sy = (SPACING_X, SPACING_Y)
 
#     # Ziel-Voxelgroesse: falls nicht vorgegeben, kleinste Pixelkante nutzen
#     output_spacing = min(sx, sy)
 
#     # Dicke, mit der jeder Frame als duenne 3D-Scheibe eingebettet wird.
#     # In Groessenordnung des Ausgabe-Spacings waehlen, damit beim Resampling
#     # ins Zielraster garantiert mindestens eine Voxelschicht getroffen wird
#     # (sonst kann die unendlich duenne Ebene "durchs Raster fallen").
#     if slice_thickness is None:
#         slice_thickness = output_spacing
 
#     # --- Schritt 1: jeden Frame als duennes 3D-Bild im Weltraum platzieren --
#     slice_images = []
#     for i in range(n_frames):
#         # numpy (H, W) -> sitk erwartet fuer 3D-Arrays Achsreihenfolge (z,y,x)
#         arr = frames[i].astype(np.float32)[np.newaxis, :, :]  # (1, H, W)
#         img = sitk.GetImageFromArray(arr)  # resultierende sitk-Groesse: (W, H, 1)
#         img.SetSpacing((sx, sy, slice_thickness))
 
#         rotation = poses[i][:3, :3]
#         translation = poses[i][:3, 3]
 
#         # Rotationsteil orthonormalisieren (robust gegen kleines numerisches
#         # Rauschen in den Pose-Daten) -> gueltige Direction-Matrix fuer sitk.
#         u, _, vt = np.linalg.svd(rotation)
#         rotation_orthonormal = u @ vt
 
#         img.SetDirection(rotation_orthonormal.flatten().tolist())
#         img.SetOrigin(translation.tolist())
#         slice_images.append(img)
 
#     # --- Schritt 2: Bounding Box aller Frames im Weltkoordinatensystem -----
#     # Eckpunkte jedes Frames (lokale mm-Koordinaten) mit der jeweiligen Pose
#     # ins Weltsystem transformieren und min/max je Achse sammeln.
#     corners_local = np.array([
#         [0, 0, 0],
#         [w * sx, 0, 0],
#         [0, h * sy, 0],
#         [w * sx, h * sy, 0],
#     ])
#     world_min = np.full(3, np.inf)
#     world_max = np.full(3, -np.inf)
#     for i in range(n_frames):
#         R = poses[i][:3, :3]
#         t = poses[i][:3, 3]
#         world_corners = corners_local @ R.T + t
#         world_min = np.minimum(world_min, world_corners.min(axis=0))
#         world_max = np.maximum(world_max, world_corners.max(axis=0))
 
#     # --- Schritt 3: leeres, achsparalleles Ziel-Volumen anlegen ------------
#     size = np.ceil((world_max - world_min) / output_spacing).astype(int) + 1
#     size = tuple(int(s) for s in size)
 
#     reference = sitk.Image(size, sitk.sitkFloat32)
#     reference.SetSpacing((output_spacing,) * 3)
#     reference.SetOrigin(world_min.tolist())
#     # Direction bleibt Identitaet -> Volumen liegt achsparallel im Weltsystem
 
#     identity_transform = sitk.Transform(3, sitk.sitkIdentity)
 
#     # --- Schritt 4: Compounding - jeden Frame ins Zielraster resamplen -----
#     # sum_image:  Summe der Intensitaeten je Zielvoxel
#     # sum_weight: Anzahl/Gewicht der Beitraege je Zielvoxel (fuer Mittelung
#     #             und um zu wissen, wo ueberhaupt Daten vorhanden sind)
#     sum_image = sitk.Image(size, sitk.sitkFloat32)
#     sum_image.CopyInformation(reference)
#     sum_weight = sitk.Image(size, sitk.sitkFloat32)
#     sum_weight.CopyInformation(reference)
 
#     for slice_img in slice_images:
#         resampled = sitk.Resample(
#             slice_img, reference, identity_transform, interpolator, 0.0, sitk.sitkFloat32
#         )
#         # Gewichtsbild: gleiche Geometrie wie slice_img, aber ueberall 1.
#         # Nach dem Resampling zeigt es, wo dieser Frame im Zielraster
#         # tatsaechlich Daten beigetragen hat.
#         weight_img = sitk.Image(slice_img.GetSize(), sitk.sitkFloat32)
#         weight_img.CopyInformation(slice_img)
#         weight_img += 1.0
#         resampled_weight = sitk.Resample(
#             weight_img, reference, identity_transform, interpolator, 0.0, sitk.sitkFloat32
#         )
 
#         sum_image += resampled
#         sum_weight += resampled_weight
 
#     # --- Schritt 5: Mittelung, Division durch 0 vermeiden -------------------
#     safe_weight = sitk.Maximum(sum_weight, 1e-6)
#     volume = sitk.Divide(sum_image, safe_weight)
#     # Sicherheitshalber Bereiche ohne jeglichen Beitrag explizit auf 0 setzen
#     has_data_mask = sitk.Cast(sum_weight > 0, sitk.sitkUInt8)
#     volume = sitk.Mask(volume, has_data_mask)
 
#     # --- Schritt 6 (optional): umschlossene Luecken auffuellen -------------
#     if fill_holes:
#         volume = _fill_interior_holes(volume, has_data_mask, max_hole_fill_iterations)
 
#     return volume
 
 
def _fill_interior_holes(
    volume: sitk.Image, has_data_mask: sitk.Image, max_iterations: int = 5
) -> sitk.Image:

    # find holes
    should_have_data_mask = sitk.BinaryFillhole(
        has_data_mask, fullyConnected=False, foregroundValue=1
    )
    hole_mask = should_have_data_mask - has_data_mask
 
    if sitk.GetArrayViewFromImage(hole_mask).sum() == 0:

        return volume  # no holes

    # init
    dilate = sitk.GrayscaleDilateImageFilter()
    dilate.SetKernelRadius(1)
 
    filled_volume = volume
    filled_mask = has_data_mask
 
    # fill holes
    for _ in range(max_iterations): # multiple times as dilatation only assigns neighboring values for empty pixels

        already_filled_in_hole = hole_mask * filled_mask
        remaining = hole_mask - already_filled_in_hole

        if sitk.GetArrayViewFromImage(remaining).sum() == 0:

            break
 
        dilated = dilate.Execute(filled_volume)
        filled_volume = filled_volume + sitk.Mask(dilated, remaining)
        filled_mask = filled_mask + remaining

    return filled_volume