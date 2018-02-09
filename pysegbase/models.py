import logging
import os.path as op

logger = logging.getLogger(__name__)
import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def softplus(x, max_error=1, keep_dtype=True):
    x = np.asarray(x)
    dtype = x.dtype
    result = max_error * np.log(1 + np.exp(x / max_error))

    if keep_dtype:
        result = result.astype(dtype)
    return result
