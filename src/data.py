import numpy as np
from typing import Tuple
from enum import Enum

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


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


class SupportedDatasets(Enum):

    # Vision
    MNIST = datasets.MNIST
    CIFAR10 = datasets.CIFAR10
    CIFAR100 = datasets.CIFAR100


def get_dataset(dataset: SupportedDatasets, save_path: str = 'data/', download: bool = True) -> (data.DataLoader, data.DataLoader):

    """
    Fetches, caches and formats a torch dataset.

    :param dataset: Dataset to be downloaded and formatted. Must be from the SupportedDatasets enumerator class.
    :param save_path: Path to desired dataset save directory.
    :param download: If true, the specified dataset will be cached in the save_path directory.
    :return:
    """

    if not isinstance(dataset, SupportedDatasets):
        raise ValueError(f'Please use the {SupportedDatasets} enumerator to specify the dataset.')

    else:

        # Normalize RGB images
        class NormalizeRGB:
            def __call__(self, sample):
                return sample/255

        tfs = transforms.ToTensor()
        if dataset.value == 'CIFAR10' or dataset.value == 'CIFAR100':
            tfs = transforms.Compose([tfs, NormalizeRGB])

        train_ds = dataset.value(save_path, train=True, download=download, transform=tfs)
        test_ds = dataset.value(save_path, train=False, download=download, transform=tfs)
        return train_ds, test_ds


def get_loaders(dataset: SupportedDatasets, batch_size: int, save_path: str = 'data/', download: bool = True,
                shuffle: bool = True) -> (data.DataLoader, data.DataLoader):

    """
    Builds training and testing torch loaders from a torch dataset from the SupportedDatasets enumerator.

    :param dataset: Dataset to be downloaded and formatted. Must be from the SupportedDatasets enumerator class.
    :param batch_size: Loader batch size.
    :param save_path: Path to desired dataset save directory.
    :param download: If true, the specified dataset will be cached in the save_path directory.
    :param shuffle: If true, the first dimension of the loaders will be shuffled.
    :return:
    """

    train_ds, test_ds = get_dataset(dataset, save_path, download)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader

