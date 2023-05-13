from typing import Tuple, Union
from enum import Enum
from functools import partial

import numpy as np
import torch.utils.data as data
import torchvision.datasets as torch_datasets
import torchvision.transforms as transforms


def sample_multinormals(n_dists: int = 5, n_points: int = 50, dims: int = 2, loc_bounds: Tuple[float, float] = (-5, 5),
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

    synth_x = np.empty((0, dims))
    synth_y = np.empty(0)

    for dist in range(n_dists):
        loc = np.random.uniform(loc_bounds[0], loc_bounds[1], size=dims)
        scale = np.random.uniform(scale_bounds[0], scale_bounds[1], size=(dims, dims))
        points = np.random.multivariate_normal(loc, scale, size=n_points)
        synth_x = np.concatenate((synth_x, points))
        synth_y = np.concatenate((synth_y, [dist+1 for _ in range(n_points)]))

    # Randomize the data
    random_idx = np.random.permutation(np.arange(len(synth_x)))
    synth_x = synth_x[random_idx]
    synth_y = synth_y[random_idx]

    return synth_x, synth_y


class CustomDatasets(Enum):
    # Synthetic
    MULTINORMAL = partial(sample_multinormals)


class TorchDatasets(Enum):
    # Vision
    MNIST = torch_datasets.MNIST
    CIFAR10 = torch_datasets.CIFAR10
    CIFAR100 = torch_datasets.CIFAR100


def get_custom_dataset(dataset: CustomDatasets, **kwargs):
    if not isinstance(dataset, CustomDatasets):
        raise ValueError(f'Please use the {CustomDatasets} enumerator to specify the dataset.')
    return dataset.value(**kwargs)


def get_torch_dataset(dataset: TorchDatasets, save_path: str = 'data/', download: bool = True) \
        -> (data.DataLoader, data.DataLoader):

    """
    Fetches, caches and formats a torch dataset.

    :param dataset: Dataset to be downloaded and formatted. Must be from the SupportedDatasets enumerator class.
    :param save_path: Path to desired dataset save directory.
    :param download: If true, the specified dataset will be cached in the save_path directory.
    :return:
    """

    if not isinstance(dataset, TorchDatasets):
        raise ValueError(f'Please use the {TorchDatasets} enumerator to specify the dataset.')

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


def get_loaders(dataset: Union[TorchDatasets, CustomDatasets], batch_size: int, save_path: str = 'data/',
                download: bool = True, shuffle: bool = True, **kwargs) -> (data.DataLoader, data.DataLoader):

    """
    Builds training and testing torch loaders from a torch dataset from the SupportedDatasets enumerator.

    :param dataset: Dataset to be downloaded and formatted. Must be from the SupportedDatasets enumerator class.
    :param batch_size: Loader batch size.
    :param save_path: Path to desired dataset save directory.
    :param download: If true, the specified dataset will be cached in the save_path directory.
    :param shuffle: If true, the first dimension of the loaders will be shuffled.
    :return:
    """

    if isinstance(dataset, TorchDatasets):
        train_ds, test_ds = get_torch_dataset(dataset, save_path, download)
    elif isinstance(dataset, CustomDatasets):
        train_ds, test_ds = get_custom_dataset(dataset, **kwargs)
    else:
        raise ValueError(f'Please use the {TorchDatasets} and {CustomDatasets} enums to specify the dataset.')

    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader

