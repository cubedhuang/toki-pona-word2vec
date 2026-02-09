# Toki Pona Word2Vec

- Training
  - `1_train_segment.py` segments the downloaded corpus into sentences and scores them with [Danielson's (2024)](https://github.com/gregdan3/sona-toki) sentence scoring system.
  - `2_train_sanitize.py` removes non-Toki Pona sentences, saving those into `input/2.cleaned.full.txt`, and removes sentences with any non-Toki Pona tokens, saving those into `input/2.cleaned.pure.txt`.
  - `3_train.py` trains a Word2Vec model on the cleaned sentences, saving the model to `models/toki_pona_w2v_{full/pure}.model`.
- Analysis
  - `accuracy.py` evaluates category assignment retrieval accuracy on the trained models.
  - `cluster.py` performs automated agglomerative clustering on the trained models, performing a hyperparameter sweep to find the optimal distance threshold for clustering.
    - `random_baseline.py` creates a random baseline for clustering performance, measuring silhouette scores.
  - `matrix.py` creates RSA matrices for the full model's categories and the Google News model's categories.
  - `stability.py` finds top category assignment changes between the full and pure models.
  - `stats.py` determines the number of sentences and tokens in the full and pure cleaned corpora.
  - `type_count.py` counts the number of non-Toki Pona types and tokens present in the full corpus.
  - `vis.py` creates UMAP visualizations of the full and pure models, coloring points by category.
