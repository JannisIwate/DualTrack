import numpy as np
import gtsam
from gtsam import NonlinearFactorGraph, Values, noiseModel
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
        self.rel_poses = constraints
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
                    mat4_to_pose3(self.rel_poses[i]),
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


def registration_noise_model(confidence: float):

    sigma_xy = max(0.5,10.0 * (1.0 - confidence))

    sigma_yaw = np.deg2rad(max(2.0,20.0 * (1.0 - confidence)))

    sigmas = np.array([1e4,       # roll
                       1e4,       # pitch
                       sigma_yaw, # yaw
                       sigma_xy,  # x
                       sigma_xy,  # y
                       1e4,       # z
                       ]
                    )

    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)