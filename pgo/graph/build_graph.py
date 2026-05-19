import numpy as np
import gtsam
import matplotlib.pyplot as plt
from gtsam import NonlinearFactorGraph, Values, noiseModel
import gtsam.utils.plot as gtsam_plot
from graph.utils import *


def extract_positions(values, pose_type="torch_tensor"):

    xs, ys, zs = [], [], []

    if pose_type == "torch_tensor":

        for el in values:

            xs.append(el[0, 3])
            ys.append(el[1, 3])
            zs.append(el[2, 3])
    elif pose_type == "gtsam_values":

        for el in sorted(values.keys()):

            p = values.atPose3(el).translation()
            x, y, z = p[0], p[1], p[2]
            
            xs.append(x)
            ys.append(y)
            zs.append(z)
    else:
        raise ValueError(f"Invalid pose_type: {pose_type}")
    return np.array(xs), np.array(ys), np.array(zs)


def build_graph(abs_poses, rel_poses, optimize = True):
    N = abs_poses.shape[0]

    # build graph
    graph = NonlinearFactorGraph()
    initial = Values()

    # gaussian noise models (tune these!)
    prior_noise = noiseModel.Diagonal.Sigmas( # small sigma equals big weight -> high impact on cost function
        np.array([1e-5]*6)
    )

    odom_noise = noiseModel.Diagonal.Sigmas(
        np.array([1e-2]*6)
    )

    # Insert nodes (absolute poses)
    for i in range(N):
        pose = mat4_to_pose3(abs_poses[i])
        initial.insert(i, pose)

    # anchor first pose, prior
    graph.add(
        gtsam.PriorFactorPose3(
            0,
            initial.atPose3(0),
            prior_noise
        )
    )

    # add odometry edges
    for i in range(N-1): # one less edge than nodes (without prior)
        rel_pose = mat4_to_pose3(rel_poses[i])

        graph.add(
            gtsam.BetweenFactorPose3(
                i,
                i + 1,
                rel_pose,
                odom_noise
            )
        )

    print(f"Graph: {graph.size()} factors, {N} poses")

    # optimize
    optimized = None
    if optimize:
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        optimized = optimizer.optimize()

    return graph, initial, optimized

def plot_marginals(graph, optimized, step_size=1, n_values=None):

    if n_values is None:
        n_values = optimized.size()

    if n_values > optimized.size():
        raise ValueError(f"n_values must be less than or equal to the number of poses!")

    marginals = gtsam.Marginals(graph, optimized)

    for i in range(0, n_values, step_size):
        print(f"Pose {i} covariance:\n{marginals.marginalCovariance(i)}\n")
    for i in range(0, n_values, step_size):
        gtsam_plot.plot_pose3(0, optimized.atPose3(i), 0.5, marginals.marginalCovariance(i))
    plt.show()

def plot_trajectories(trajectories, labels=None, colors=None):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = len(trajectories)

    # defaults
    if labels is None:
        labels = [f"traj_{i}" for i in range(n)]
    if colors is None:
        colors = [None] * n

    for i, (xs, ys, zs) in enumerate(trajectories):
        ax.plot(xs, ys, zs,
                label=labels[i],
                color=colors[i])

    ax.scatter(xs[0], ys[0], zs[0], color=colors[i])

    ax.set_title("Pose Graph Trajectories")
    ax.legend()
    plt.show()