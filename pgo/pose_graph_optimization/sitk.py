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
    replace_pred_factor: float = 0.1,
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

    crop_offset_y = crop_offset_x = 0
    if "crop" in options:
        frame_i, (crop_offset_y, crop_offset_x) = crop_center_frames(
            frame_i, (0.5, 0.5)
        )
        frame_j, _ = crop_center_frames(frame_j, (0.5, 0.5))

    ## images
    fixed = sitk.GetImageFromArray(frame_i.astype(np.float32)) # 640x480 (x, y), other way round for sitk!
    moving = sitk.GetImageFromArray(frame_j.astype(np.float32))

    if "roi" in options:
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

    image_origin = (
        ORIGIN_X + crop_offset_x * SPACING_X,
        ORIGIN_Y + crop_offset_y * SPACING_Y,
    )
    fixed.SetOrigin(image_origin) # origin in SITK is not origin in DualTrack! Also, origin only has minimal effect when it is the same for both
    # (still it makes sense to leave it as is as CenteredTransformInitializer computes R center from this)
    moving.SetOrigin(image_origin)
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
    transform_reg_inv = np.linalg.inv(sitk_to_3dof(transform_reg)) # inverse as sitk finds moving to fixed, so Tj->i
    metric_after = registration.GetMetricValue()

    if "show_ir" in options:

        image_plot(fixed, title="fixed")
        image_plot(fixed, title="fixed")

        image_plot(moving, title="moving")
        image_plot(moving, title="moving")

        registered_image_ir = sitk.Resample(
            moving,
            fixed,
            transform_reg,
            sitk.sitkLinear,
            0.0
        )
        image_plot(registered_image_ir, title="ir transform")
        plt.show()

    print("rotation determinants:", np.linalg.det(ref_transform))
    print(f"image size: {fixed.GetSize()}")

    return (
        transform_reg_inv,
        float(metric_before_identity),
        float(metric_before_gt),
        float(metric_before_pred),
        float(metric_after),
    )


def sitk_3d_register( # benutzt alle 20 Kerne
    volume_frames: np.ndarray,
    volume_poses: np.ndarray,
    slice_frame: np.ndarray,
    slice_frame_pose: np.ndarray,
    slice_frame_pose_gt: np.ndarray,
    metric: str = "mi",
    optimizer: str = "gradient",
    options: str = "",
    transform_type: str = "versor",
    replace_pred_factor: float = 0.1,
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

    if "crop" in options:
        volume_frames, (volume_offset_y, volume_offset_x) = crop_center_frames(
            volume_frames, (0.5, 0.5)
        )
        slice_frame, (slice_offset_y, slice_offset_x) = crop_center_frames(
            slice_frame, (0.5, 0.5)
        )
    else:
        volume_offset_y = volume_offset_x = 0
        slice_offset_y = slice_offset_x = 0

    volume_origin = (
        ORIGIN_X + volume_offset_x * SPACING_X,
        ORIGIN_Y + volume_offset_y * SPACING_Y,
    )
    slice_origin = (
        ORIGIN_X + slice_offset_x * SPACING_X,
        ORIGIN_Y + slice_offset_y * SPACING_Y,
    )

    # build volumes
    start = time.time()
    volume, _, _ = build_volume_from_slices(
        volume_frames,
        volume_poses,
        options=options,
        image_origin=volume_origin,
    )
    volume_building_time = time.time() - start
    
    start = time.time()
    slice_volume = slice_to_volume(slice_frame,
                                    slice_frame_pose,
                                   (SPACING_X, SPACING_Y, SPACING_X),
                                   (*slice_origin, 0.0),
                                   thickness=4)
    slice_building_time = time.time() - start

    # init IR
    reg_init_time_start = time.time()
    fixed = slice_volume
    moving = volume

    if "roi" in options:
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

    registration = build_registration_object(metric, optimizer, options)

    # # set initial transform to gt transform and evaluate
    # initial = set_transform_from_pose(
    #     fixed,
    #     moving,
    #     slice_frame_pose_gt,
    #     transform_type,
    # )
    # registration.SetInitialTransform(initial)
    # metric_before_gt = registration.MetricEvaluate( # takes up to 60% of total runtime!!
    #     fixed,
    #     moving,
    # )

    # # set initial transform to predicted transform and evaluate
    # initial = set_transform_from_pose(
    #     fixed,
    #     moving,
    #     slice_frame_pose,
    #     transform_type,
    # )
    # registration.SetInitialTransform(initial)
    # metric_before_pred = registration.MetricEvaluate(
    #     fixed,
    #     moving,
    # )

    # set initial transform to identity and evaluate
    initial = set_transform_from_pose(
            fixed,
            moving,
            np.eye(4),
            transform_type,
        )
    registration.SetInitialTransform(initial)
    # metric_before_identity = registration.MetricEvaluate(
    #     fixed,
    #     moving,
    # )
    reg_init_time = time.time() - reg_init_time_start

    # register
    start = time.time()
    transform_reg = registration.Execute(
        fixed=fixed,
        moving=moving,
    )
    reg_time = time.time() - start
    transform_reg = sitk_to_6dof(transform_reg)
    transform_reg_inv = np.linalg.inv(transform_reg)
    metric_after = registration.GetMetricValue()

    if "show_ir" in options:
        show_volumes(
            volumes=[volume, slice_volume, slice_volume, slice_volume],
            volume_poses=[np.eye(4), transform_reg, slice_frame_pose_gt, slice_frame_pose],
            volume_outline_colors=["black", "red", "green", "yellow"],
            frames=volume_frames,
            frame_poses=volume_poses,
            volume_labels=["volume", "ir slice", "gt slice", "pred slice"]
        )
    all_time = time.time() - all_start

    metric_before_identity = 0
    metric_before_gt = 0
    metric_before_pred = 0
    metric_after = 0
    
    if "debug" in options:
        print(f"\n=== 3D Registration Timing ===")
        print(f"Total time:      {all_time:.3f}s")
        print(f"Vol building:    {volume_building_time:.3f}s ({(volume_building_time / all_time) * 100:.1f}%)")
        print(f"Slice building:  {slice_building_time:.3f}s ({(slice_building_time / all_time) * 100:.1f}%)")
        print(f"Registration:    {reg_time:.3f}s ({(reg_time / all_time) * 100:.1f}%)")
        print(f"IR Init:         {reg_init_time:.3f}s ({(reg_init_time / all_time) * 100:.1f}%)")
        print(f"Volume shape: {moving.GetSize()}")
        print(f"Fixed shape: {fixed.GetSize()}")
        print(f"Metric before (id/gt/pred): {metric_before_identity:.4f} / {metric_before_gt:.4f} / {metric_before_pred:.4f}")
        print(f"Metric after: {metric_after:.4f}")

    return (
            transform_reg_inv, # pose relative to first window frame pose
            float(metric_before_identity),
            float(metric_before_gt),
            float(metric_before_pred),
            float(metric_after),
        )


def show_volumes(
    volumes: list[sitk.Image],
    volume_poses: list[np.ndarray] | None = None,
    frames: list[np.ndarray] | None = None,
    frame_poses: list[np.ndarray] | None = None,
    volume_outline_colors=None,
    frame_color="blue",
    volume_labels: list[str] | None = None,
    frame_label: str | None = "frames",
):
    if len(volumes) == 0:
        raise ValueError("At least one volume must be provided.")

    if volume_poses is None:
        volume_poses = [np.eye(4) for _ in volumes]

    if len(volume_poses) != len(volumes):
        raise ValueError("volumes and volume_poses must have the same length.")

    if volume_outline_colors is None:
        volume_outline_colors = ["black"] * len(volumes)

    if len(volume_outline_colors) != len(volumes):
        raise ValueError("Need one outline color per volume.")

    if volume_labels is not None and len(volume_labels) != len(volumes):
        raise ValueError("volume_labels must have the same length as volumes.")

    plotter = pv.Plotter()

    # volumes
    for idx, (img, pose, color) in enumerate(zip(
        volumes,
        volume_poses,
        volume_outline_colors,
    )):
        # build grid and add to plot
        grid = _sitk_to_grid(img)
        outline = grid.outline()
        outline.transform(pose, inplace=True)

        plotter.add_mesh(
            outline,
            color=color,
            line_width=4,
            label=volume_labels[idx] if volume_labels is not None else None,
        )

    # frames
    if frames is not None:
        if frame_poses is None:
            raise ValueError("frame_poses must be provided when frames are shown.")

        if len(frames) != len(frame_poses):
            raise ValueError("frames and frame_poses must have the same length.")

        for frame, pose in zip(frames, frame_poses):

            # build sitk image
            frame_volume = slice_to_volume(
                slice_frame=frame,
                pose=pose,
                spacing=[SPACING_X, SPACING_Y, SPACING_X],
                origin=volumes[0].GetOrigin(),
                thickness=1
            )

            # build grid and add to plot
            grid = _sitk_to_grid(frame_volume)
            outline = grid.outline()
            outline.transform(pose, inplace=True)

            plotter.add_mesh(
                outline,
                color=frame_color,
                line_width=4,
                label=frame_label,
            )

    # plot
    labels = [*volume_labels, "frames"]
    colors = [*volume_outline_colors, frame_color]
    plotter.add_legend(
        labels=[[l, c] for l, c in zip(labels, colors)],
        bcolor="grey",
        loc="upper right",
    )
    plotter.add_axes()
    plotter.show()


def _sitk_to_grid(img: sitk.Image):

    arr = sitk.GetArrayFromImage(img)

    grid = pv.ImageData()
    grid.dimensions = np.array(arr.shape[::-1])
    grid.spacing = img.GetSpacing()
    grid.origin = img.GetOrigin()

    grid.point_data["values"] = arr.ravel(order="F")

    return grid


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
    slice_frame: np.ndarray,
    pose: np.ndarray,
    spacing: list[float],
    origin: list[float],
    thickness: int = 4,
    set_direction = False
) -> sitk.Image:

    # create sitk image
    slice_frame = np.expand_dims(slice_frame, axis=0)
    slice_volume = sitk.GetImageFromArray(slice_frame.astype(np.float32))
    slice_volume.SetSpacing((spacing[0], spacing[1], spacing[2]))
    slice_volume.SetOrigin((origin[0], origin[1], origin[2]))
    slice_volume = sitk.Expand(slice_volume, [1, 1, thickness]) # expand to fulfill min four voxel criteria

    R = pose[:3, :3]
    t = pose[:3, 3]

    # local image coordinate system -> world
    if set_direction:
        slice_volume.SetDirection(R.reshape(-1).tolist())

    return slice_volume


def _sparse_compound_volume(
    flat_idx: np.ndarray,
    values: np.ndarray,
    volume_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate voxel sums/counts without allocating a full-length bincount array.

    The expensive part in the original code was:
        np.bincount(flat_idx, minlength=volume_total)
    which creates an array of length equal to the entire volume bounding box. Here we
    compress to only the occupied voxels first, then scatter back into the dense array
    only for the actually used indices.
    """
    if flat_idx.size == 0:
        dense_shape = (volume_size[2], volume_size[1], volume_size[0])
        return (
            np.zeros(dense_shape, dtype=np.float32),
            np.zeros(dense_shape, dtype=np.float32),
        )

    unique_idx, inverse = np.unique(flat_idx, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=unique_idx.size)
    counts = np.bincount(inverse, minlength=unique_idx.size)

    total_voxels = int(volume_size[0] * volume_size[1] * volume_size[2])
    dense_values = np.zeros(total_voxels, dtype=np.float32)
    dense_counts = np.zeros(total_voxels, dtype=np.float32)
    dense_values[unique_idx] = sums.astype(np.float32)
    dense_counts[unique_idx] = counts.astype(np.float32)

    dense_shape = (volume_size[2], volume_size[1], volume_size[0])
    volume_np = dense_values.reshape(dense_shape)
    weight_np = dense_counts.reshape(dense_shape)
    return volume_np, weight_np


# def build_volume_from_slices(
#     frames: np.ndarray,
#     poses: np.ndarray,
#     volume_spacing: tuple[float, float, float] = (SPACING_X, SPACING_Y, SPACING_X),
#     options: str = "",
# ):
#     """
#     Build 3D volume from 2D frames with known poses.

#     This keeps the same API as the original implementation, but avoids the expensive
#     full-volume bincount allocation by first reducing only to occupied voxels.

#     Options:
#         "fill_holes": Enable hole filling (disabled by default for speed)
#         Any other options are ignored here
#     """
#     # init
#     sx, sy = (SPACING_X, SPACING_Y)
#     ox, oy = (ORIGIN_X, ORIGIN_Y)
#     # ox, oy = (0, 0)

#     n, h, w = frames.shape

#     image_corners = np.array([
#         [0,     0,     0, 1],
#         [w - 1, 0,     0, 1],
#         [0,     h - 1, 0, 1],
#         [w - 1, h - 1, 0, 1],
#     ])

#     # convert pixel to image coords
#     image_corners[:, 0] = image_corners[:, 0] * sx + ox # in image coords
#     image_corners[:, 1] = image_corners[:, 1] * sy + oy

#     # get corner positions in world
#     world_points = []

#     for pose in poses: # first is identity

#         pts = (pose @ image_corners.T).T[:, :3] # points are in stored in rows, need columns
#         world_points.append(pts)

#     world_points = np.concatenate(world_points) # all corner points in world
#     world_min = world_points.min(axis=0) # min x, y and z
#     world_max = world_points.max(axis=0)

#     # create empty volume
#     spacing = np.asarray(volume_spacing)

#     volume_size = np.ceil( # bounding box around volume
#         (world_max - world_min) / spacing
#     ).astype(int) + 1

#     # create image coord system based on frames meta data
#     xx, yy = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
#     ximg = (xx * sx + ox).ravel()
#     yimg = (yy * sy + oy).ravel()
#     xy = np.stack([ximg, yimg], axis=-1)          # (P, 2)

#     R_xy = poses[:, :3, :2]                        # (n, 3, 2)
#     t    = poses[:, :3, 3]                         # (n, 3)

#     # apply transforms
#     world = np.matmul(xy, R_xy.transpose(0, 2, 1)) + t[:, None, :]

#     # compute voxel indices of volume
#     voxel = np.round((world - world_min) / spacing).astype(np.int64)
#     valid = (
#         (voxel[..., 0] >= 0) & (voxel[..., 0] < volume_size[0]) &
#         (voxel[..., 1] >= 0) & (voxel[..., 1] < volume_size[1]) &
#         (voxel[..., 2] >= 0) & (voxel[..., 2] < volume_size[2])
#     )
#     flat_idx = (voxel[..., 2] * volume_size[1] + voxel[..., 1]) * volume_size[0] + voxel[..., 0]
#     flat_idx = flat_idx[valid]
#     values = frames.reshape(n, -1)[valid]

#     # Sparse-style accumulation: avoid full-volume bincount length.
#     # This keeps memory proportional to occupied voxels instead of the whole bounding box.
#     volume_np, weight_np = _sparse_compound_volume(flat_idx, values, volume_size)

#     # average overlapping voxels
#     mask = weight_np > 0
#     volume_np[mask] /= weight_np[mask] # shape (z, y, x)

#     # convert to SITK format 
#     volume = sitk.GetImageFromArray(volume_np)
#     volume.SetSpacing(volume_spacing)
#     # volume.SetOrigin(tuple(world_min))
#     volume.SetDirection(np.eye(3).flatten())

#     # fill holes (only if explicitly requested)
#     if "fill_holes" in options:
#         mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
#         mask_image.CopyInformation(volume)
#         volume = _fill_interior_holes(volume, mask_image)

#     return (
#         sitk.Expand(volume, [1, 1, 4]),
#         tuple(world_min),
#         volume_spacing,
#     )

# claude
def build_volume_from_slices(
    frames: np.ndarray,
    poses: np.ndarray,
    volume_spacing: tuple[float, float, float] = (SPACING_X, SPACING_Y, SPACING_X),
    options: str = "",
    image_origin: tuple[float, float] = (ORIGIN_X, ORIGIN_Y),
):
    """Build a 3D volume from 2D frames with known poses.

    frames: (n, h, w) in (z, y, x) order — n frames stacked along z.
    poses:  (n, 4, 4) world_from_local transforms, R | t, normalized so poses[0] == identity.
    """
    n, h, w = frames.shape
    poses = np.asarray(poses, dtype=np.float64)

    # Pixel indices (upper-left origin) -> local image-space coords (origin at image center).
    xx, yy = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    x_local = xx.ravel() * SPACING_X + image_origin[0]
    y_local = yy.ravel() * SPACING_Y + image_origin[1]
    local_pts = np.stack(
        [x_local, y_local, np.zeros_like(x_local), np.ones_like(x_local)], axis=-1
    )  # (P, 4)

    # Map every pixel of every frame into world space in one shot:
    # world[i, p] = poses[i] @ local_pts[p]
    world_points = np.einsum("nij,pj->npi", poses, local_pts)[..., :3]  # (n, P, 3)
    world_flat = world_points.reshape(-1, 3)
    values_flat = frames.reshape(-1)  # matches world_flat ordering (n slowest, then y, then x)

    world_min = world_flat.min(axis=0)
    world_max = world_flat.max(axis=0)

    spacing = np.asarray(volume_spacing, dtype=np.float64)
    volume_size = np.ceil((world_max - world_min) / spacing).astype(np.int64) + 1  # (nx, ny, nz)

    voxel_idx = np.round((world_flat - world_min) / spacing).astype(np.int64)
    valid = np.all((voxel_idx >= 0) & (voxel_idx < volume_size), axis=1)
    voxel_idx = voxel_idx[valid]
    values_flat = values_flat[valid]

    flat_idx = (voxel_idx[:, 2] * volume_size[1] + voxel_idx[:, 1]) * volume_size[0] + voxel_idx[:, 0]

    volume_np, weight_np = _sparse_compound_volume(flat_idx, values_flat, volume_size)

    # Average overlapping voxels.
    mask = weight_np > 0
    volume_np[mask] /= weight_np[mask]

    # World coordinates are already axis-aligned physical coordinates, so the volume's
    # direction is identity — poses only ever affect *where* points land, not the grid's
    # own orientation. origin = world_min places voxel (0,0,0) at the bounding-box corner.
    volume = sitk.GetImageFromArray(volume_np)
    volume.SetSpacing(volume_spacing)
    volume.SetOrigin(tuple(world_min))
    volume.SetDirection(np.eye(3).flatten())

    if "fill_holes" in options:
        mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_image.CopyInformation(volume)
        volume = _fill_interior_holes(volume, mask_image)

    volume = sitk.Expand(volume, [1, 1, 4])

    print("rotation determinants:", [
        np.linalg.det(pose[:3, :3]) for pose in poses
    ])
    print(f"volume size: {volume.GetSize()}")

    return volume, tuple(world_min), volume_spacing


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
    """
    Build SITK registration object with optimized parameters.
    
    KEY OPTIMIZATION: Reduced iterations from 200 to 50 for faster registration.
    This prevents long registration times (+10 seconds) on larger volumes.
    """

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
        # OPTIMIZED: Reduced numberOfIterations from 200 to 50
        # This provides ~4x speedup while maintaining registration quality
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=0.1, 
            minStep=1e-4, 
            numberOfIterations=50  # REDUCED from 200
        )
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
    # Highly recommended: improves convergence and reduces iteration count
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

        ysize = nda.shape[0] # DT has (y, x) instead of (x, y)
        xsize = nda.shape[1]

        figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

        fig = plt.figure(title, figsize=figsize, dpi=dpi)
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        extent = (0, xsize * spacing[1], 0, ysize * spacing[0])

        t = ax.imshow(
            nda, extent=extent, interpolation="hamming", cmap="gray", origin="upper" # origin is upper left instead of lower left for DT
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


def crop_center_frames(
    frames: np.ndarray,
    fractions: tuple[float, float] = (0.5, 0.5),
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop the last two dimensions of frames around their common image center."""
    if frames.ndim < 2:
        raise ValueError("frames must have at least two dimensions.")
    if len(fractions) != 2:
        raise ValueError("fractions must contain (height, width).")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("All fractions must be in the range (0, 1].")

    height, width = frames.shape[-2:]
    crop_height = max(1, int(round(height * fractions[0])))
    crop_width = max(1, int(round(width * fractions[1])))
    offset_y = (height - crop_height) // 2
    offset_x = (width - crop_width) // 2

    slices = (..., slice(offset_y, offset_y + crop_height), slice(offset_x, offset_x + crop_width))
    return frames[slices], (offset_y, offset_x)


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
