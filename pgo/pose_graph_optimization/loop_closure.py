import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from .image_registration import register_2d
from .utils import inbetween_to_accumulated
from src.utils.pose import matrix_to_pose_vector


def detect_loop_closures(
    pred_poses,
    frames,
    transforms,
    gt_transforms=None,
    method="nearest_neighbor",
    stepsize=10,
    temporal_offset=50,
    threshold=0.9,
    n_neighbors=1,
    loop_consistency_check=False,
    registration_options="roi",
    plot_callback=None,
    registration_cache=None,
    verbose=True,
):
    pose_vectors = np.asarray(matrix_to_pose_vector(pred_poses), dtype=np.float32)
    transforms = np.asarray(transforms)
    accumulated_transforms = inbetween_to_accumulated(transforms)

    if gt_transforms is not None:
        gt_transforms = np.asarray(gt_transforms)
        accumulated_gt_transforms = inbetween_to_accumulated(gt_transforms)
    else:
        accumulated_gt_transforms = accumulated_transforms

    n_frames = len(pose_vectors)

    loop_closures = []
    if method == "cosine_similarity":

        for i in range(0, n_frames, stepsize):

            query = pose_vectors[i].reshape(1, -1)

            for j in range(i + temporal_offset, n_frames, stepsize):

                if j <= i:
                    continue

                candidate = pose_vectors[j].reshape(1, -1)

                score = cosine_similarity(query, candidate)[0, 0]

                # [-1,1] -> [0,1]
                score = (score + 1.0) / 2.0

                if score < threshold:
                    continue
                print(f"lc score: {score}")

                candidate = evaluate_candidate(
                    i, int(j), score, frames, accumulated_transforms,
                    accumulated_gt_transforms, registration_options,
                    plot_callback, registration_cache, verbose,
                )
                if candidate is not None:
                    loop_closures.append(candidate)

    elif method == "nearest_neighbor":

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
        nn.fit(pose_vectors)

        for i in range(0, n_frames, stepsize):

            distances, indices = nn.kneighbors(
                pose_vectors[i].reshape(1, -1)
            )

            for dist, j in zip(distances[0], indices[0]):

                if j <= i:
                    continue

                if j - i < temporal_offset:
                    continue

                score = 1.0 - dist

                if score < threshold:
                    continue

                candidate = evaluate_candidate(
                    i, int(j), score, frames, accumulated_transforms,
                    accumulated_gt_transforms, registration_options,
                    plot_callback, registration_cache, verbose,
                )
                if candidate is not None:
                    loop_closures.append(candidate)

    else:
        raise NotImplementedError(
            f"Method '{method}' not implemented."
        )
    
    if loop_consistency_check:

        cycles = find_cycles(loop_closures)

        # print(cycles)

        #TODO finish
    return loop_closures


def evaluate_candidate(
    source_idx,
    target_idx,
    descriptor_score,
    frames,
    accumulated_transforms,
    accumulated_gt_transforms,
    registration_options,
    plot_callback=None,
    registration_cache=None,
    verbose=True,
):
    pair = (source_idx, target_idx)

    if registration_cache is not None and pair in registration_cache:
        registration = registration_cache[pair]
    else:
        registration = register_2d(
            frame_i=frames[source_idx],
            frame_j=frames[target_idx],
            ref_transform=(
                np.linalg.inv(accumulated_transforms[source_idx])
                @ accumulated_transforms[target_idx]
            ),
            gt_transform=(
                np.linalg.inv(accumulated_gt_transforms[source_idx])
                @ accumulated_gt_transforms[target_idx]
            ),
            sitk={
                "metric": "mi",
                "optimizer": "gradient",
                "options": registration_options,
            },
        )
        if registration_cache is not None:
            registration_cache[pair] = registration

    (
        transform,
        confidence,
        valid,
        metric_before_identity,
        metric_before_gt,
        metric_before_pred,
        metric_after,
    ) = registration

    if verbose:
        print(
            f"LC frames {source_idx}->{target_idx}: "
            f"descriptor={descriptor_score:.4f}, "
            f"before_identity={metric_before_identity:.4f}, "
            f"before_gt={metric_before_gt:.4f}, "
            f"before_pred={metric_before_pred:.4f}, "
            f"after={metric_after:.4f}, valid={valid}"
        )

    if plot_callback is not None:
        plot_callback(source_idx, target_idx)

    if not valid:
        return None

    return {
        "source_idx": source_idx,
        "target_idx": target_idx,
        "combined_score": (descriptor_score + confidence) / 2,
        "transform": transform,
    }


def find_cycles(loop_closures):

    G = nx.DiGraph()

    for lc in loop_closures:

        G.add_edge(
            lc["source_idx"],
            lc["target_idx"],
            transform=lc["transform"],
            score=lc["combined_score"],
        )

    return list(nx.simple_cycles(G))