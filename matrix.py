import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import gensim.downloader as api

from categories import CATEGORIES, CATEGORIES_EN
from models import load_models
from util import get_category_centroids

PLOT_SUBSET = [
    "animals",
    "plants_and_food",
    "body",
    # "kinship",
    # "colors",
    "spatial",
    "movement",
    "conflict",
    # "knowledge",
    "evaluative.positive",
    "evaluative.negative",
    # "quantity",
    # "time.period",
]
PLOT_SUBSET = None


def compute_similarity_matrix(model, categories):
    """
    Returns a DataFrame where index and columns are category names,
    and values are cosine similarity between centroids.
    """
    cat_names = sorted(list(categories.keys()))
    if PLOT_SUBSET:
        cat_names = [c for c in cat_names if c in PLOT_SUBSET]

    n = len(cat_names)
    matrix = np.zeros((n, n))

    centroids = get_category_centroids(model, categories)

    for i in range(n):
        for j in range(n):
            name_i = cat_names[i]
            name_j = cat_names[j]

            if centroids[name_i] is None or centroids[name_j] is None:
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = np.dot(centroids[name_i], centroids[name_j])

    return pd.DataFrame(matrix, index=cat_names, columns=cat_names)


def save_long_format(df, filename):
    df_long = df.stack().reset_index()
    df_long.columns = ["RowLabel", "ColLabel", "Value"]

    labels = df.index.tolist()
    mapper = {label: i for i, label in enumerate(labels)}

    df_long["x"] = df_long["ColLabel"].map(mapper)
    df_long["y"] = df_long["RowLabel"].map(mapper)

    df_long.to_csv(filename, index=False)
    print(f"saved {filename}")


def save_labels(category_dict, filename: str, subset=None):
    cat_names = sorted(list(category_dict.keys()))
    if subset:
        cat_names = [c for c in cat_names if c in subset]

    clean_names = [name.replace("_", " ") for name in cat_names]

    df_labels = pd.DataFrame(
        {"index": range(1, 1 + len(cat_names)), "label": clean_names}
    )

    df_labels.to_csv(filename, index=False)
    print(f"saved {filename}")


def main():
    (full_model, _) = load_models()

    print("loading google news word2vec")
    en_model = api.load("word2vec-google-news-300")

    df_tp = compute_similarity_matrix(full_model.wv, CATEGORIES)
    df_en = compute_similarity_matrix(en_model, CATEGORIES_EN)

    save_labels(
        CATEGORIES, filename="output/matrix_labels_full.csv", subset=PLOT_SUBSET
    )
    save_long_format(df_tp, "output/matrix_tp_full.csv")
    save_long_format(df_en, "output/matrix_en_full.csv")


if __name__ == "__main__":
    main()
