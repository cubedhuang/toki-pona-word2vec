import collections
import re

from linku_words import load_linku_words


INPUT = "./input/2.cleaned.full.txt"

CATEGORY_BASE = "./type_count/"
CATEGORIES = [
    "entities_tp",
    "unadapted",
    "nimisin_uncommon",
    "nimisin_obscure",
    "nimisin_etc",
    "abbv",
    "misc",
]


def count_tokens(file: str, categories: dict[str, list[str]]):
    linku_words = load_linku_words()
    counts = [
        (word, count)
        for word, count in collections.Counter(re.findall(r"\w+", file)).items()
        if count >= 5 and word not in linku_words
    ]
    counts.sort(key=lambda c: c[1])
    token_counts = {cat: 0 for cat in CATEGORIES}
    token_counts["UNCLASSIFIED"] = 0
    type_counts = {cat: 0 for cat in CATEGORIES}
    type_counts["UNCLASSIFIED"] = 0
    for word, count in counts:
        for cat, category in categories.items():
            if word in category:
                token_counts[cat] += count
                type_counts[cat] += 1
                break
        else:
            token_counts["UNCLASSIFIED"] += count
            type_counts["UNCLASSIFIED"] += 1
            print(f"UNCLASSIFIED {word: <12} {count}")
    print("token counts:\n", token_counts)
    print(
        "total nimisin tokens:\n",
        sum(
            (
                token_counts[c]
                for c in ["nimisin_uncommon", "nimisin_obscure", "nimisin_etc"]
            )
        ),
    )
    print("type counts:\n", type_counts)
    print(
        "total nimisin types:\n",
        sum(
            (
                type_counts[c]
                for c in ["nimisin_uncommon", "nimisin_obscure", "nimisin_etc"]
            )
        ),
    )


if __name__ == "__main__":
    categories = {}
    for cat in CATEGORIES:
        categories[cat] = []
        with open(f"{CATEGORY_BASE}{cat}.txt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                w = line.strip()
                if not w:
                    continue
                for existing in categories.values():
                    if existing != categories[cat]:
                        if w in existing:
                            print(f"duplicate {w}")
                            raise Exception(f"duplicate {w}")
                categories[cat].append(w)

    with open(INPUT) as file:
        count_tokens(file.read(), categories)
