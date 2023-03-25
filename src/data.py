import numpy as np
from typing import Tuple


def sample_multinormals(n_dists: int = 5, n_points: int = 20, dims: int = 2, loc_bounds: Tuple[float, float] = (-5, 5),
                        scale_bounds: Tuple[float, float] = (0.5, 1)) -> Tuple[np.ndarray, np.ndarray]:

    """
    Samples 'n_points' points from 'n_dists' gaussian distributions of dimensions 'dims', each with bounded and
    uniformly sampled location and scale.

    :param n_dists: Number of gaussian distributions.
    :param n_points: Number of points generated for each distribution.
    :param dims: Dimensionality of the data.
    :param loc_bounds: Bounds of the location parameter of the gaussian distributions.
    :param scale_bounds: Bounds of the scale parameter of the gaussian distributions.
    :return:
    """

    # Location of the barycenter for our synthetic data distributions
    dist_locs = np.random.uniform(loc_bounds[0], loc_bounds[1], size=(n_dists, dims))
    synth_data = np.empty((n_dists, n_points, dims))

    for idx, loc in enumerate(dist_locs):
        synth_data[idx] = np.random.normal(loc=loc, scale=np.random.uniform(scale_bounds[0], scale_bounds[1], size=dims),
                                           size=(n_points, dims))

    random_data = np.random.permutation(synth_data.reshape((n_dists*n_points, dims)))

    return synth_data, random_data
