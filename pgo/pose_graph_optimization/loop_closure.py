import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from .image_registration import register


def detect_loop_closures(
    feature_vectors,
    frames,
    transforms,
    pixel_to_image,
    method="nearest_neighbor",
    stepsize=10,
    temporal_offset=50,
    threshold=0.7,
    n_neighbors=1
):
    feature_vectors = np.asarray(feature_vectors, dtype=np.float32)

    n_frames = len(feature_vectors)

    loop_closures = []
    seen_pairs = set()

    # currently yields potential LCs for almost every frame
    if method == "cosine_similarity":

        for i in range(0, n_frames, stepsize):

            query = feature_vectors[i].reshape(1, -1)

            for j in range(i + temporal_offset, n_frames, stepsize):

                candidate = feature_vectors[j].reshape(1, -1)

                score = cosine_similarity(query, candidate)[0, 0]

                # [-1,1] -> [0,1]
                score = (score + 1.0) / 2.0

                if score < threshold:
                    continue
                #print(f"lc score: {score}")

                transform, reg_score = register(
                    frames[i],
                    frames[j],
                    transforms[i],
                    pixel_to_image,
                )

                loop_closures.append(
                    {
                        "source_idx": i,
                        "target_idx": j,
                        "combined_score": (score + reg_score) / 2,
                        "transform": transform,
                    }
                )

    elif method == "nearest_neighbor":

        nn = NearestNeighbors(n_neighbors = n_neighbors + 1, metric="cosine")
        nn.fit(feature_vectors)

        for i in range(0, n_frames, stepsize):

            distances, indices = nn.kneighbors(
                feature_vectors[i].reshape(1, -1)
            )

            for dist, j in zip(distances[0], indices[0]):

                if i == j:
                    continue

                if abs(i - j) < temporal_offset:
                    continue

                pair = tuple(sorted((i, int(j))))

                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)

                score = 1.0 - dist

                if score < threshold:
                    continue
                print(f"lc score: {score}")

                transform, reg_score = register(
                    frames[i],
                    frames[j],
                    transforms[i],
                    pixel_to_image,
                )

                loop_closures.append(
                    {
                        "source_idx": i,
                        "target_idx": int(j),
                        "combined_score": (score + reg_score) / 2,
                        "transform": transform,
                    }
                )

    else:
        raise NotImplementedError(
            f"Method '{method}' not implemented."
        )

    return loop_closures