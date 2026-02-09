import os
from gensim.models import Word2Vec


BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
FULL_PATH = os.path.join(MODELS_DIR, "toki_pona_w2v.full.model")
PURE_PATH = os.path.join(MODELS_DIR, "toki_pona_w2v.pure.model")


def load_models() -> tuple[Word2Vec, Word2Vec]:
    """
    Loads both models

    :return: A tuple containing the FULL model first, PURE model second
    :rtype: tuple[Word2Vec, Word2Vec]
    """
    return Word2Vec.load(FULL_PATH), Word2Vec.load(PURE_PATH)
