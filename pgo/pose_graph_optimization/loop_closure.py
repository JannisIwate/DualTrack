import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .image_registration import register


def detect_loop_closures(
    feature_vectors,
    frames,
    method="k_nearest_neighbor",
    stepsize=10,
    temporal_offset=50,
    threshold=0.7,
):
    if method != "k_nearest_neighbor":
        raise NotImplementedError(f"Method '{method}' not implemented.")

    feature_vectors = np.asarray(feature_vectors, dtype=np.float32,)

    n_frames = len(feature_vectors)

    loop_closures = []

    for i in range(0, n_frames, stepsize):

        query_feature = feature_vectors[i].reshape(1, -1)

        # search future frames
        for j in range(i + temporal_offset, n_frames, stepsize,):

            candidate_feature = (feature_vectors[j].reshape(1, -1))

            score = cosine_similarity(query_feature,candidate_feature)[0, 0]

            # convert cosine similarity to score (-1 to 1 -> 0 to 1)
            score = (score + 1.0) / 2.0
            
            # register potential LC frames
            if score >= threshold:

                transform = register(frames[i],frames[j])
                loop_closures.append(
                    {
                        "source_idx": i,
                        "target_idx": j,
                        "score": float(score),
                        "transform": transform,
                    }
                )

    return loop_closures