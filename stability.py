import os
import numpy as np
from gensim.models import Word2Vec
from categories import CATEGORIES, ALL_CATEGORIZED_WORDS
from models import load_models
from util import get_category_centroids, normalize


def get_top_category(word, model, centroids):
    if word not in model.wv:
        return None, 0.0

    word_vec = normalize(model.wv[word])

    best_cat = None
    best_score = -1.0

    for cat_name, centroid_vec in centroids.items():
        centroid_norm = np.linalg.norm(centroid_vec)
        if centroid_norm == 0:
            continue

        score = np.dot(word_vec, centroid_vec) / centroid_norm

        if score > best_score:
            best_score = score
            best_cat = cat_name

    return best_cat


def main():
    full_model, pure_model = load_models()

    if not full_model or not pure_model:
        return

    noisy_centroids = get_category_centroids(full_model.wv)
    filtered_centroids = get_category_centroids(pure_model.wv)

    matches = 0
    total_analyzed = 0
    mismatches = []

    for word in sorted(ALL_CATEGORIZED_WORDS):
        cat_full = get_top_category(word, full_model, noisy_centroids)
        cat_filtered = get_top_category(word, pure_model, filtered_centroids)

        total_analyzed += 1

        if cat_full == cat_filtered:
            matches += 1
        else:
            mismatches.append(
                {"word": word, "full_cat": cat_full, "pure_cat": cat_filtered}
            )

    print(f"\nanalyzed words: {total_analyzed}")
    print(f"matches:        {matches}")
    print(f"mismatches:     {len(mismatches)}")
    print(f"stability:      {matches/total_analyzed:.1%}")

    if mismatches:
        print()
        print("-" * 60)
        print(f"{'word':<15} {'full':<25} {'pure':<25}")
        print("-" * 60)
        for m in mismatches:
            print(f"{m['word']:<15} {m['full_cat']:<25} -> {m['pure_cat']:<25}")
        print("-" * 60)


if __name__ == "__main__":
    main()
