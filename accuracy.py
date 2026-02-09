import numpy as np
from gensim.models import Word2Vec
from categories import CATEGORIES
from models import load_models
from util import get_category_centroids, normalize


def get_word_accuracy(model_obj, centroids, word_to_cats_map):
    hits = 0
    total_assignments = 0

    for word, true_cats in word_to_cats_map.items():
        if word not in model_obj.wv:
            continue

        word_vec = normalize(model_obj.wv[word])

        all_sims = []
        cat_sim_map = {}

        for cat_name, centroid in centroids.items():
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm == 0:
                sim = 0.0
            else:
                sim = np.dot(word_vec, centroid) / centroid_norm

            all_sims.append(sim)
            cat_sim_map[cat_name] = sim

        mu = np.mean(all_sims)
        sigma = np.std(all_sims)
        threshold = mu + 1 * sigma

        for cat in true_cats:
            if cat in cat_sim_map:
                total_assignments += 1
                score = cat_sim_map[cat]

                if score > threshold:
                    hits += 1

    return hits, total_assignments


def main():
    word_to_cats = {}
    for cat, words in CATEGORIES.items():
        for w in words:
            if w not in word_to_cats:
                word_to_cats[w] = []
            word_to_cats[w].append(cat)

    full_model, pure_model = load_models()

    full_centroids = get_category_centroids(full_model.wv)
    f_hits, f_total = get_word_accuracy(full_model, full_centroids, word_to_cats)
    pure_centroids = get_category_centroids(pure_model.wv)
    p_hits, p_total = get_word_accuracy(pure_model, pure_centroids, word_to_cats)

    print(f"{'Metric':<25} | {'Full Model':<12} | {'Pure Model':<12}")
    print("-" * 60)
    print(f"{'Total Assignments':<25} | {f_total:<12} | {p_total:<12}")
    print(f"{'Successful Retrievals':<25} | {f_hits:<12} | {p_hits:<12}")
    print(
        f"{'Accuracy (%)':<25} | {(f_hits/f_total*100):.1f}%        | {(p_hits/p_total*100):.1f}%"
    )
    print("-" * 60)
    print(
        "Diff (Pure - Full):         {:.1f}%".format(
            (p_hits / p_total * 100) - (f_hits / f_total * 100)
        )
    )


if __name__ == "__main__":
    main()
