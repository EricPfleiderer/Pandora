import numpy as np

from pandora.learning.supervised.classification.classifiers import SupervisedClassifier


class NearestNeighbors(SupervisedClassifier):

    def __init__(self, window: float):
        self._window = window
        self._x = None
        self._y = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """

        :param x: N-dimensional array. The first dimension is treated as samples, while the rest are considered spatial dimensions.
        :param y: 1 dimension array. Contains classification labels for points in x.
        :return:
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y must have the same length in the first dimension.')

        if len(y.shape) > 1:
            raise ValueError('y must have a single dimension.')

        self._x = x
        self._y = y

        return self

    def predict(self, x: np.ndarray):

        """
        Predicts a batch of samples.
        :param x: N-dimensional array.The first dimension is treated as samples, while the rest are considered spatial dimensions.
        :return: 1-dimensional array containing the predicted labels of x.
        """

        if x.shape[1:] != self._x.shape[1:]:
            raise ValueError('Shape mismatch. Samples to classify do not share the same trailing dimensions as the '
                             'training data.')

        labels = np.zeros(shape=x.shape[0])

        for idx, sample in enumerate(x):
            # Find neighbors in the window
            neighbor_idx = np.where(np.linalg.norm(sample-self._x) < self._window)

            # Find and count the labels of the neighbors
            unique, counts = np.unique(neighbor_idx, return_counts=True)

            # Majority vote (mode)
            labels[idx] = unique[np.where(counts == np.max(counts))]

        return labels

    def find_neighbors(self, x: np.ndarray):

        """

        :param x: 1-dimensional array (single sample).
        :return: 1-dimensional array containing the nearest
        """