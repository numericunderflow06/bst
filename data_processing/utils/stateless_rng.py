import numpy as np


class StatelessRandomGenerator:
    def __init__(self, seed=42):
        self.seed = seed

    def set_seed(self, new_seed):
        self.seed = new_seed

    def random(self, size=None):
        rng = np.random.default_rng(self.seed)
        return rng.random(size)

    def integers(self, low, high=None, size=None):
        rng = np.random.default_rng(self.seed)
        return rng.integers(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        rng = np.random.default_rng(self.seed)
        return rng.choice(a, size, replace, p)


global_rng = StatelessRandomGenerator(42)


def set_global_seed(new_seed):
    global_rng.set_seed(new_seed)
