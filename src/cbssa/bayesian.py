import numpy as np


def normal(
        prior_mean: float,
        prior_standard_deviation: float,
        sample_number_of_observations: int,
        sample_mean: float,
        sample_standard_deviation: float,
) -> (float, float):
    post_mean = (prior_mean / prior_standard_deviation ** 2 + sample_number_of_observations * sample_mean /
                 sample_standard_deviation ** 2) / (1 / prior_standard_deviation ** 2 + sample_number_of_observations /
                                                    sample_standard_deviation ** 2)

    post_standard_deviation = np.sqrt(1 / (1 / prior_standard_deviation ** 2 +
                                           sample_number_of_observations / sample_standard_deviation ** 2))

    return post_mean, post_standard_deviation
