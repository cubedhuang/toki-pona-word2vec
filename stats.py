import os


FULL_PATH = "./input/2.cleaned.full.txt"
PURE_PATH = "./input/2.cleaned.pure.txt"


def analyze(path: str):
    sentences = 0
    tokens = 0

    with open(path) as file:
        for line in file:
            if len(line) == 0:
                continue
            sentences += 1
            tokens += len(line.split(" "))

    print(f"{sentences} sentences and {tokens} tokens")


if __name__ == "__main__":
    print("full:")
    analyze(FULL_PATH)
    print("pure:")
    analyze(PURE_PATH)
