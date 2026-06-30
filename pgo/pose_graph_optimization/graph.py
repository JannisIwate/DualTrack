import numpy as np
import gtsam
from gtsam import NonlinearFactorGraph, Values, noiseModel
import torch
from pose_graph_optimization.utils import mat4_to_pose3


class PoseGraph:

    def __init__(
        self,
        poses,
        constraints,
        initial_pose=None,
        noise_type="gaussian",
        prior_noise_sigma=1e-5,
        odom_noise_sigma=1e-2,
        optimizer="gauss-newton",
        optimizer_params=None,
    ):
        self.abs_poses = poses
        self.constraints = constraints
        self.initial_pose = initial_pose

        self.noise_type = noise_type

        self.prior_noise_sigma = prior_noise_sigma
        self.odom_noise_sigma = odom_noise_sigma

        self.optimizer_name = optimizer
        self.optimizer_params = optimizer_params or {}

        self.N = poses.shape[0]

        # if len(constraints) != self.N - 1:
        #     raise ValueError(
        #         "Expected len(rel_poses) == N - 1."
        #     )

        self.additional_constraints = []

        self.graph = None
        self.initial = None
        self.optimized = None

    def _create_noise_model(self, sigma):

        sigma = np.asarray(sigma)

        if sigma.ndim == 0:
            sigma = np.full(6, sigma)

        elif sigma.size == 1:
            sigma = np.full(6, sigma.item())

        elif sigma.size != 6:
            raise ValueError(
                "Noise must be scalar or length-6 vector."
            )

        return noiseModel.Diagonal.Sigmas(sigma)

    def add_constraint(
        self,
        node_i,
        node_j,
        transform,
        noise_sigma=1e-2,
    ):
        self.additional_constraints.append(
            {
                "i": node_i,
                "j": node_j,
                "transform": transform,
                "noise": noise_sigma,
            }
        )

    def build_graph(self):

        self.graph = NonlinearFactorGraph()
        self.initial = Values()

        ## insert nodes
        for idx in range(self.N):
            self.initial.insert(
                idx,
                mat4_to_pose3(self.abs_poses[idx]),
            )

        # prior
        prior_pose = (
            self.initial_pose
            if self.initial_pose is not None
            else self.abs_poses[0]
        )

        self.graph.add(
            gtsam.PriorFactorPose3(
                0,
                mat4_to_pose3(prior_pose),
                self._create_noise_model(self.prior_noise_sigma),
            )
        )

        ## add constraints
        if np.isscalar(self.odom_noise_sigma):

            edge_noises = [self.odom_noise_sigma] * (self.N - 1)

        else:

            if len(self.odom_noise_sigma) != self.N - 1:
                raise ValueError(
                    f"Expected {self.N - 1} odom noises "
                    f"but got {len(self.odom_noise_sigma)}"
                )

            edge_noises = self.odom_noise_sigma

        for i in range(self.N - 1):

            self.graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    mat4_to_pose3(self.constraints[i]),
                    self._create_noise_model(edge_noises[i])
                )
            )

        # additional constraints
        for c in self.additional_constraints:

            self.graph.add(
                gtsam.BetweenFactorPose3(
                    c["i"],
                    c["j"],
                    mat4_to_pose3(c["transform"]),
                    c["noise"],
                )
            )

        self._optimize()

        return self.graph, self.initial, self.optimized

    def _optimize(self):

        # if self.graph is None or self.initial is None:
        #     raise RuntimeError(
        #         "Call build_graph() first."
        #     )

        optimizer_name = self.optimizer_name.lower()

        if optimizer_name == "gauss-newton":

            params = gtsam.GaussNewtonParams()

            optimizer = gtsam.GaussNewtonOptimizer(
                self.graph,
                self.initial,
                params,
            )

        elif optimizer_name == "levenberg-marquardt":

            params = gtsam.LevenbergMarquardtParams()

            optimizer = (
                gtsam.LevenbergMarquardtOptimizer(
                    self.graph,
                    self.initial,
                    params,
                )
            )

        elif optimizer_name == "dogleg":

            params = gtsam.DoglegParams()

            optimizer = gtsam.DoglegOptimizer(
                self.graph,
                self.initial,
                params,
            )

        else:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer_name}"
            )

        # optional parameter injection
        for key, value in self.optimizer_params.items():

            setter = f"set{key[0].upper()}{key[1:]}"

            if hasattr(params, setter):
                getattr(params, setter)(value)

        self.optimized = optimizer.optimize()

        return self.optimized


def registration_noise_model(confidence: float, ref_sigma=1e-2):

    sigma_xy = min(2*1e-5, (2*1e-5) * confidence)
    #sigma_yaw = np.deg2rad(max(2.0,20.0 * (1.0 - confidence)))
    #print(f"sigma: {sigma_xy}")

    sigmas = np.array([ref_sigma,       # roll
                       ref_sigma,       # pitch
                       sigma_xy,  # yaw
                       sigma_xy,  # x
                       sigma_xy,  # y
                       ref_sigma,       # z
                       ]
                    )

    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def rotation_matrix_to_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Berechnet den Rotationswinkel einer Rotationsmatrix.

    Args:
        R: Tensor [..., 3, 3]

    Returns:
        Tensor [...] mit Winkel in Radiant.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    return torch.acos(cos_theta)


def slerp_rotations(
    R1: torch.Tensor,
    R2: torch.Tensor,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Interpoliert zwischen zwei Rotationen über Log/Exp.

    Args:
        R1: [3, 3]
        R2: [3, 3]
        alpha: Interpolationsfaktor

    Returns:
        Interpolierte Rotation [3, 3]
    """
    R_rel = R1.T @ R2

    theta = rotation_matrix_to_angle(R_rel)

    if theta < 1e-8:
        return R1.clone()

    skew = (R_rel - R_rel.T) / (2.0 * torch.sin(theta))

    omega = torch.tensor(
        [
            skew[2, 1],
            skew[0, 2],
            skew[1, 0]
        ],
        device=R1.device,
        dtype=R1.dtype
    )

    omega = omega * alpha * theta

    angle = torch.linalg.norm(omega)

    if angle < 1e-8:
        return R1.clone()

    axis = omega / angle

    K = torch.tensor(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ],
        device=R1.device,
        dtype=R1.dtype
    )

    R_inc = (
        torch.eye(3, device=R1.device, dtype=R1.dtype)
        + torch.sin(angle) * K
        + (1.0 - torch.cos(angle)) * (K @ K)
    )

    return R1 @ R_inc


def compute_pose_weights(
    poses: torch.Tensor,
    base_weight: float = 1e-2,
    translation_scale: float = 0.01,
    rotation_scale_deg: float = 2.0
):
    """
    Bewertet mittlere Pose in Dreiergruppen.

    Args:
        poses:
            [N, 4, 4]

        base_weight:
            Referenzgewicht

        translation_scale:
            Fehler in Metern, bei dem Gewicht deutlich sinkt.

        rotation_scale_deg:
            Fehler in Grad, bei dem Gewicht deutlich sinkt.

    Returns:
        dict mit:
            weights
            translation_errors
            rotation_errors_deg
    """
    weights = len(poses)*[base_weight]
    translation_errors = []
    rotation_errors_deg = []

    rot_scale = torch.deg2rad(
        torch.tensor(rotation_scale_deg)
    )

    for i in range(0, len(poses) - 2, 3):

        T0 = poses[i]
        T1 = poses[i + 1]
        T2 = poses[i + 2]

        R0 = T0[:3, :3]
        R1 = T1[:3, :3]
        R2 = T2[:3, :3]

        t0 = T0[:3, 3]
        t1 = T1[:3, 3]
        t2 = T2[:3, 3]

        t_mid = 0.5 * (t0 + t2)

        R_mid = slerp_rotations(
            R0,
            R2,
            alpha=0.5
        )

        translation_error = torch.linalg.norm(
            t1 - t_mid
        )

        R_error = R_mid.T @ R1

        rotation_error = rotation_matrix_to_angle(
            R_error
        )

        normalized_error = (
            translation_error.item() / translation_scale
            + rotation_error.item() / rot_scale
        )

        weight = base_weight * (
            1.0 + normalized_error.item()/100
        )

        weights[i] = weight
        translation_errors.append(translation_error)
        rotation_errors_deg.append(
            torch.rad2deg(rotation_error)
        )

    return {
        "weights": weights,
        "translation_errors": torch.stack(
            translation_errors
        ),
        "rotation_errors_deg": torch.stack(
            rotation_errors_deg
        )
    }