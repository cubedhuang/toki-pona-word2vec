import requests


def load_linku_words(categories=["core", "common"]):
    response = requests.get("https://api.linku.la/v1/words")
    response.raise_for_status()
    words = response.json()
    return [w["word"] for w in words.values() if w["usage_category"] in categories]
