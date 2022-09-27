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
