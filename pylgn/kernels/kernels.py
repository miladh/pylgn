import numpy as np


def create_non_separable_kernel():
    def evaluate(t, x, y):
        raise NotImplementedError
    return evaluate
