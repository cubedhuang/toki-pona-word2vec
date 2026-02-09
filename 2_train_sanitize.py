import json

from linku_words import load_linku_words


def main():
    with open("./input/1.sentences.jsonl") as input, open(
        "./input/2.cleaned.pure.txt", "w"
    ) as pure_output, open("./input/2.cleaned.full.txt", "w") as full_output:
        linku_words = load_linku_words(["core", "common"])

        for line in input:
            message = json.loads(line)
            for sentence in message["sentences"]:
                words = [word.lower() for word in sentence["words"] if word.isalpha()]
                if len(words) > 1:
                    full_output.write(" ".join(words) + "\n")
                if any(word not in linku_words for word in words):
                    continue
                pure_output.write(" ".join(words) + "\n")


if __name__ == "__main__":
    main()
