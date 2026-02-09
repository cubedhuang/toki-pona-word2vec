import numpy as np
from categories import CATEGORIES
from gensim.models import KeyedVectors


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_category_centroids(
    wv: KeyedVectors, categories=CATEGORIES
) -> dict[str, np.float64]:
    centroids = {}
    for cat_name, words in categories.items():
        vectors = []
        for w in words:
            if w in wv:
                vectors.append(normalize(wv[w]))
        if vectors:
            centroids[cat_name] = np.mean(vectors, axis=0)
    return centroids
