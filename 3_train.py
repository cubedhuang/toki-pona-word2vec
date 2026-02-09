from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence


class Logger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_train_begin(self, model):
        print(f"started training!")

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"epoch {self.epoch} completed")


def main():
    sentences = LineSentence("./input/2.cleaned.pure.txt")

    model = Word2Vec(
        sentences=sentences,
        vector_size=24,
        workers=4,
        sg=1,
        epochs=10,
        callbacks=[Logger()],
    )

    model.save("./models/toki_pona_w2v.pure.model")


if __name__ == "__main__":
    main()
