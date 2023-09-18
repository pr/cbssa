import numpy as np

from enum import Enum


class MissingValueAction(Enum):
    DELETE = "delete"
    NEAREST = "nearest"
    MEAN = "mean"
    MEDIAN = "median"
    FILL_VALUE = "fill_value"


def average(
        number_array: list,
        row_weight: list,
) -> float:

    return np.average(number_array, weights=row_weight)


def stddev(
        number_array: list,
        row_weight: list,
) -> float:

    return np.sqrt(np.cov(number_array, aweights=row_weight))
