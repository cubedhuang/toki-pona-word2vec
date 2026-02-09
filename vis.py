import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import umap
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from categories import CATEGORIES, ALL_CATEGORIZED_WORDS
from models import load_models

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["TeX Gyre Heros", "Helvetica"],
        "font.size": 10,
    }
)

OUTPUTS = {
    "main": "./output/vis_main.png",
    "legend": "./output/vis_legend.png",
}
USE_UMAP = True


def process_model(model, words):
    vecs = []
    valid_words = []
    embeddings_dict = {}

    for w in words:
        if w in model.wv:
            v = model.wv[w]
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vecs.append(v)
            valid_words.append(w)
            embeddings_dict[w] = v

    return np.array(vecs), valid_words, embeddings_dict


def compute_primary_categories(embeddings_dict, valid_words):
    centroids = {}
    for cat_name, cat_words in CATEGORIES.items():
        cat_vecs = [embeddings_dict[w] for w in cat_words if w in embeddings_dict]
        if cat_vecs:
            centroid = np.mean(cat_vecs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[cat_name] = centroid

    new_labels = []
    for w in valid_words:
        vec = embeddings_dict[w]
        best_cat = "Uncategorized"
        max_sim = -1.0
        for cat_name, centroid in centroids.items():
            sim = np.dot(vec, centroid)
            if sim > max_sim:
                max_sim = sim
                best_cat = cat_name
        new_labels.append(best_cat)
    return new_labels


def plot_separated(models_data):
    model_labels = {}
    used_categories_set = set()

    for name, (vecs, words, emb_dict) in models_data.items():
        labels = compute_primary_categories(emb_dict, words)
        model_labels[name] = labels
        used_categories_set.update(labels)

    active_cats = sorted(list(used_categories_set))
    print(f"{len(active_cats)} active categories (of {len(CATEGORIES)})")

    palette = sns.color_palette("hls", len(active_cats))
    color_map = dict(zip(active_cats, palette))

    # plots without legend
    fig, axes = plt.subplots(2, 1, figsize=(10, 18))

    for ax, (name, (vecs, words, emb_dict)) in zip(axes, models_data.items()):
        print(f"computing umap for {name}")

        reassigned_labels = model_labels[name]

        if USE_UMAP:
            reducer = umap.UMAP(
                n_components=2,
                metric="cosine",
                n_neighbors=15,
                min_dist=0.1,
                random_state=15,
            )
        else:
            reducer = TSNE(
                n_components=2,
                metric="cosine",
                perplexity=15,
                init="pca",
                random_state=17,
            )
        projections = reducer.fit_transform(vecs)

        df = pd.DataFrame(
            projections, columns=["x", "y"]  # pyright: ignore[reportArgumentType]
        )
        df["word"] = words
        df["category"] = reassigned_labels

        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="category",
            palette=palette,
            hue_order=active_cats,
            ax=ax,
            s=100,
            alpha=0.8,
            legend=False,
        )
        for _, row in df.iterrows():
            ax.text(
                row["x"],
                row["y"] + 0.06,
                row["word"],
                fontsize=11,
                ha="center",
                va="bottom",
                alpha=0.8,
                fontweight="medium",
            )

        ax.set_title(f"{name}", fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(f"{OUTPUTS['main']}", dpi=300, bbox_inches="tight")
    print(f"saved {OUTPUTS['main']}")

    # generate legend

    legend_handles = []
    for cat in active_cats:
        # create a fake marker for the legend
        handle = mlines.Line2D(
            [],
            [],
            color="white",
            marker="o",
            markerfacecolor=color_map[cat],
            markeredgecolor=color_map[cat],
            markersize=10,
            label=cat,
            linestyle="None",
        )
        legend_handles.append(handle)

    if legend_handles:
        fig_leg = plt.figure(figsize=(3, 8))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis("off")

        ax_leg.legend(
            handles=legend_handles,
            loc="center",
            title="Nearest Category",
            fontsize=10,
            title_fontsize=12,
            frameon=False,
            labelspacing=1.2,
        )

        fig_leg.savefig(f"{OUTPUTS['legend']}", dpi=300, bbox_inches="tight")
        print(f"Saved {OUTPUTS['legend']}")


def main():
    data = {}
    chosen_words = ALL_CATEGORIZED_WORDS
    sorted_words = sorted(list(chosen_words))

    full_model, pure_model = load_models()

    data["Full Corpus Model"] = process_model(full_model, sorted_words)
    data["Pure Toki Pona Model"] = process_model(pure_model, sorted_words)

    plot_separated(data)


if __name__ == "__main__":
    main()
