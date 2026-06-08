import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .image_registration import register


def detect_loop_closures(
    feature_vectors,
    frames,
    transforms,
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

            lc_score = cosine_similarity(query_feature, candidate_feature)[0, 0]

            # convert cosine similarity to score (-1 to 1 -> 0 to 1)
            lc_score = (lc_score + 1.0) / 2.0
            print(f"lc score {lc_score}") # currently very high for every pair, look at how it is created
            
            # register potential LC frames
            if lc_score >= threshold:
                #print("lc detected")

                transform, reg_score = register(frames[i],frames[j], transforms[i])

                combined_score = (lc_score + reg_score) / 2

                loop_closures.append(
                    {
                        "source_idx": i,
                        "target_idx": j,
                        "score": float(combined_score),
                        "transform": transform,
                    }
                )

    return loop_closures