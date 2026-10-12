import warnings

import numpy as np
import gtsam
from gtsam import NonlinearFactorGraph, Values, noiseModel
from pose_graph_optimization.utils import mat4_to_pose3

# Alles von Claude!!!
class PoseGraph:

    def __init__(
        self,
        poses: np.ndarray,
        constraints: np.ndarray,
        initial_pose: np.ndarray | None = None,
        noise_type: str = "gaussian",
        prior_noise_sigma=1e-5,
        odom_noise_sigma=1e-2,
        optimizer: str = "gauss-newton",
        optimizer_params: dict[str, object] | None = None,
        constraint_offset: int | None = None,
    ):
        """
        constraint_offset:
            0    -> constraints[i] ist die Transformation  i -> i+1   (len == N-1)
            1    -> constraints[k] ist die Transformation  k-1 -> k   (len == N,
                    constraints[0] wird ignoriert)
            None -> automatisch: waehlt den Offset, der am besten zu abs_poses passt
        """
        self.abs_poses = poses
        self.constraints = constraints
        self.initial_pose = initial_pose

        self.noise_type = noise_type

        self.prior_noise_sigma = prior_noise_sigma
        self.odom_noise_sigma = odom_noise_sigma

        self.optimizer_name = optimizer
        self.optimizer_params = optimizer_params or {}

        self.N = poses.shape[0]
        self.constraint_offset = self._resolve_offset(constraint_offset)

        self.additional_constraints = []

        self.graph = None
        self.initial = None
        self.optimized = None

    # ------------------------------------------------------------------
    # Index-Handling
    # ------------------------------------------------------------------
    def _relative_from_abs(self, i: int) -> np.ndarray:
        """Erwartete Relativtransformation i -> i+1 aus den absoluten Posen."""
        return np.linalg.inv(self.abs_poses[i]) @ self.abs_poses[i + 1]

    def _deviation(self, offset: int) -> float:
        """Maximale Abweichung zwischen constraints und aus abs_poses abgeleiteten Relativposen."""
        return max(
            np.abs(self._relative_from_abs(i) - self.constraints[i + offset]).max()
            for i in range(self.N - 1)
        )

    def _resolve_offset(self, offset: int | None) -> int:
        n_c = len(self.constraints)

        valid = []
        if n_c >= self.N - 1:
            valid.append(0)
        if n_c >= self.N:
            valid.append(1)

        if offset is not None:
            if offset not in valid:
                raise ValueError(
                    f"constraint_offset={offset} passt nicht zu "
                    f"N={self.N} Posen und {n_c} Constraints."
                )
            return offset

        if not valid:
            raise ValueError(
                f"Zu wenige Constraints: N={self.N}, len(constraints)={n_c}."
            )

        devs = {o: self._deviation(o) for o in valid}
        best = min(devs, key=devs.get)

        if devs[best] > 1e-6:
            warnings.warn(
                f"Constraints sind nicht konsistent zu abs_poses "
                f"(max. Abweichung je Offset: {devs}). "
                f"Verwende Offset {best}."
            )
        return best

    def _edges(self):
        """Liefert (i, j, T_ij) fuer alle Odometrie-Kanten mit korrektem Index."""
        for i in range(self.N - 1):
            yield i, i + 1, self.constraints[i + self.constraint_offset]

    # ------------------------------------------------------------------
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
        transform: np.ndarray,
        noise_sigma: gtsam.gtsam.noiseModel.Base = None,
    ):
        # default values
        if noise_sigma is None:
            noise_sigma = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2] * 6))

        self.additional_constraints.append(
            {
                "i": node_i,
                "j": node_j,
                "transform": transform,
                "noise": noise_sigma,
            }
        )

    def build_graph(self) -> tuple[NonlinearFactorGraph, Values, Values]:

        self.graph = NonlinearFactorGraph()
        self.initial = Values()

        # insert nodes
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

        # odometry noise per edge
        if np.isscalar(self.odom_noise_sigma):
            edge_noises = [self.odom_noise_sigma] * (self.N - 1)
        else:
            if len(self.odom_noise_sigma) != self.N - 1:
                raise ValueError(
                    f"Expected {self.N - 1} odom noises "
                    f"but got {len(self.odom_noise_sigma)}"
                )
            edge_noises = self.odom_noise_sigma

        # odometry constraints (korrekt indiziert)
        for k, (i, j, T_ij) in enumerate(self._edges()):
            self.graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    j,
                    mat4_to_pose3(T_ij),
                    self._create_noise_model(edge_noises[k]),
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

    def _optimize(self) -> Values:

        optimizer_name = self.optimizer_name.lower()

        if optimizer_name == "gauss-newton":
            params = gtsam.GaussNewtonParams()
            optimizer_cls = gtsam.GaussNewtonOptimizer

        elif optimizer_name == "levenberg-marquardt":
            params = gtsam.LevenbergMarquardtParams()
            optimizer_cls = gtsam.LevenbergMarquardtOptimizer

        elif optimizer_name == "dogleg":
            params = gtsam.DoglegParams()
            optimizer_cls = gtsam.DoglegOptimizer

        else:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer_name}"
            )

        # Parameter MUESSEN vor dem Erzeugen des Optimizers gesetzt werden,
        # da dieser die Params kopiert.
        for key, value in self.optimizer_params.items():
            setter = f"set{key[0].upper()}{key[1:]}"
            if hasattr(params, setter):
                getattr(params, setter)(value)
            else:
                warnings.warn(f"Optimizer-Parameter '{key}' unbekannt, ignoriert.")

        optimizer = optimizer_cls(self.graph, self.initial, params)
        self.optimized = optimizer.optimize()

        return self.optimized


def registration_noise_model(confidence, ref_sigma=1e-2):

    sigma_xy = min(2 * 1e-5, (2 * 1e-5) * confidence)

    # GTSAM Pose3 Tangent-Reihenfolge: [rot(x,y,z), trans(x,y,z)]
    sigmas = np.array([
        ref_sigma,  # roll
        ref_sigma,  # pitch
        sigma_xy,   # yaw
        sigma_xy,   # x
        sigma_xy,   # y
        ref_sigma,  # z
    ])

    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)
