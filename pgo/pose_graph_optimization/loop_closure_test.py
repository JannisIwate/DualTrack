# graph/loop_closure.py

import numpy as np
from sklearn.neighbors import NearestNeighbors


def find_loop_closures(
    features,
    temporal_window=50,
    k_neighbors=10,
    max_distance=0.5,
):
    """
    Returns:
        [(i,j,dist), ...]
    """

    features = features.astype(np.float32)

    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors + 1,
        metric="cosine",
        algorithm="auto",
    )

    nbrs.fit(features)

    distances, indices = nbrs.kneighbors(features)

    loop_closures = []

    for i in range(len(features)):

        for dist, j in zip(distances[i], indices[i]):

            if i == j:
                continue

            if abs(i - j) < temporal_window:
                continue

            if dist > max_distance:
                continue

            loop_closures.append(
                (i, int(j), float(dist))
            )

    return loop_closures