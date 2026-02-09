import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from categories import ALL_CATEGORIZED_WORDS

DIMENSIONS = 24
ITERATIONS = 300
VOCAB_SIZE = len(ALL_CATEGORIZED_WORDS)


max_silhouettes = []

for i in range(ITERATIONS):
    X = np.random.normal(0, 1, (VOCAB_SIZE, DIMENSIONS))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms

    max_sil = -1.0

    for k in range(5, 80):
        tr = k / 100.0

        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=tr,
            metric="cosine",
            linkage="complete",
        )

        labels = model.fit_predict(X)

        n_labels = len(set(labels))
        if 1 < n_labels < VOCAB_SIZE:
            sil = silhouette_score(X, labels, metric="cosine")
            if sil > max_sil:
                max_sil = sil

    max_silhouettes.append(max_sil)
    print(f"run {i+1:02d}/{ITERATIONS}: {max_sil:.4f}")

avg_max = np.mean(max_silhouettes)
std_max = np.std(max_silhouettes)

print(f"random baseline mean:  {avg_max:.4f} (std = {std_max:.4f})")
