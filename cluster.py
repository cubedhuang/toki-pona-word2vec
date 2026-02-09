import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from categories import ALL_CATEGORIZED_WORDS

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from gensim.models import Word2Vec
from models import load_models


def evaluate_model(model: Word2Vec):
    results = []

    vectors = np.array([model.wv[word] for word in ALL_CATEGORIZED_WORDS])
    X = normalize(vectors, norm="l2")

    for k in range(5, 80):
        tr = k / 100
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=tr,
            metric="cosine",
            linkage="complete",
        )
        labels = clustering.fit_predict(X)
        sil = silhouette_score(X, labels, metric="cosine")
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        results.append((tr, sil, ch, db))

        clusters = {}
        for label, word in zip(labels, ALL_CATEGORIZED_WORDS):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(word)

    print("\n".join(map(lambda r: f"{r[0]}, {r[1]}, {r[2]}, {r[3]}", results)))


def main():
    full, pure = load_models()
    print("full model:")
    evaluate_model(full)
    print("pure model:")
    evaluate_model(pure)


if __name__ == "__main__":
    main()
