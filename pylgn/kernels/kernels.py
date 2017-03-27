import numpy as np


def create_non_separable_kernel():
    def evaluate(w, kx, ky):
        return 0.0
    return evaluate
