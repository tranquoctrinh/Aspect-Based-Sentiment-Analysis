from typing import Dict


class TensorboardAggregator:
    """
        Log average value periodically instead of logging on every batch
    """

    def __init__(self, writer, every=5):
        self.writer = writer
        self.every = every

        self.step = 0
        self.scalars = None

    def log(self, scalars: Dict[str, float]):
        self.step += 1

        if self.scalars is None:
            self.scalars = scalars.copy()
        else:
            for k, v in scalars.items():
                self.scalars[k] += v

        if self.step % self.every == 0:
            for k, v in self.scalars.items():
                self.writer.add_scalar(k, v / self.every, self.step)
            self.scalars = None


SPECIAL_TOKENS = {"additional_special_tokens": ["<e>", "</e>"]}
MAX_LENGTH = 110
CLASS_DICT = {"positive": 0, "neutral": 1, "negative": 2}
TAG2IDX = {
    True: {
        "O": 0,
        "B-positive": 1,
        "B-neutral": 2,
        "B-negative": 3,
        "I-positive": 4,
        "I-neutral": 5,
        "I-negative": 6,
    },
    False: {
        "O": 0,
        "B": 1,
        "I": 2,
    }
}
IDX2TAG = {
    True: {
        0: "O",
        1: "B-positive",
        2: "B-neutral",
        3: "B-negative",
        4: "I-positive",
        5: "I-neutral",
        6: "I-negative",
    },
    False: {
        0: "O",
        1: "B",
        2: "I",
    }
}

config = {
    "model_base": "xlm-roberta-base",  # "google/mt5-base",  #
    "lr": 5e-5,
    "epochs": 20,
    "batch_size": 16,
    "accum_steps": 2,
    "sentiment": False,
}
if config["sentiment"]:
    config["model_path"] = "model_seq_labeling_{}_with_sentiment.pt".format(
        config["model_base"].replace("/", "_"))
else:
    config["model_path"] = "model_seq_labeling_{}_no_sentiment.pt".format(
        config["model_base"].replace("/", "_"))
