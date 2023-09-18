import numpy as np


def normal(
        mean: float,
        standard_deviation: float,
        number_of_observations: int,
        sample_mean: float,
        sample_stdev: float,
) -> (float, float):

    post_m = (mean / standard_deviation ** 2 + number_of_observations * sample_mean / sample_stdev ** 2) / \
             (1 / standard_deviation ** 2 + number_of_observations / sample_stdev ** 2)
    post_sd = np.sqrt(1 / (1 / standard_deviation ** 2 + number_of_observations / sample_stdev ** 2))

    return post_m, post_sd
