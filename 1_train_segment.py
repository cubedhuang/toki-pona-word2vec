import json
from sonatoki.ilo import Ilo
from sonatoki.Configs import CorpusConfig

ILO = Ilo(**CorpusConfig)


def main():
    with open("./input/0.aggregated.jsonl") as input, open(
        "./input/1.sentences.jsonl", "w"
    ) as output:
        for line in input:
            message = json.loads(line)
            message_scorecard = ILO.make_scorecard(message["content"])
            message_score = message_scorecard["score"]
            if message_score <= 0.1:
                continue

            sentences = []
            for scorecard in ILO.make_scorecards(message["content"]):
                words = scorecard["cleaned"]
                if not words:
                    continue  # omit empty sentences
                if scorecard["score"] >= 0.8 and (
                    len(words) >= 3 or message_score >= 0.3
                ):
                    sentence = {"words": words, "score": scorecard["score"]}
                    sentences.append(sentence)

            if len(sentences):
                segmented = {
                    "id": message["id"],
                    "sentences": sentences,
                }
                output.write(json.dumps(segmented) + "\n")


if __name__ == "__main__":
    main()
