"""
gcm_reward.py — Frozen additive Group-Contribution-Method (GCM) model.

Standalone reward/scoring model for downstream use (e.g. RL reward shaping).
Zero PyTorch dependency: only numpy required at inference time (pickle load).
"""

import numpy as np


class FrozenGCM:
    def __init__(self, coefficients, bias, fragment_vocab,
                 experiment, alpha, metrics):
        self.c = np.array(coefficients, dtype=np.float32)
        self.b = float(bias)
        self.fragment_vocab = fragment_vocab
        self.f2i = {f: i for i, f in enumerate(fragment_vocab)}
        self.experiment = experiment
        self.alpha = alpha
        self.metrics = metrics

    def predict(self, fragment_set: list) -> float:
        n = np.zeros(len(self.c), dtype=np.float32)
        for frag in fragment_set:
            if frag in self.f2i:
                n[self.f2i[frag]] += 1
        return float(n @ self.c + self.b)

    def error(self, fragment_set: list, y_true: float) -> float:
        return abs(y_true - self.predict(fragment_set))

    def reward(self, fragment_set: list, y_true: float) -> float:
        return -self.error(fragment_set, y_true)

    def reward_gap(self, y: list, y_prime: list, y_true: float) -> float:
        return self.reward(y, y_true) - self.reward(y_prime, y_true)

    def coverage(self, fragment_set: list) -> float:
        if not fragment_set:
            return 0.0
        return sum(1 for f in fragment_set if f in self.f2i) / len(fragment_set)

    def top_fragments(self, n=20):
        idx = np.argsort(np.abs(self.c))[::-1][:n]
        return [(self.fragment_vocab[i], float(self.c[i])) for i in idx]

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str):
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj
