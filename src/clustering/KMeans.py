import numpy as np
import matplotlib.pyplot as plt


class KMeans:

    def __init__(self, data: np.ndarray, n_means: int, init: str = 'forgy', max_iters=100):
        self.data: np.ndarray = data
        self.n_means: int = n_means
        self.init: str = init

        self.means: np.ndarray = np.zeros(shape=(n_means,)+data[0].shape)
        self.history2d: np.ndarray = np.expand_dims(np.zeros(data.shape[0]), axis=0)

        # Forgy initialization
        if init == 'forgy':
            indexes = np.arange(0, data.shape[0])
            self.means = data[np.random.choice(indexes, size=n_means)]

        # Assume random partition initialization
        else:
            cluster_idx = np.random.random_integers(0, n_means, size=data.shape[0])
            for n in range(n_means):
                self.means[n] = np.mean(data[np.where(cluster_idx == n)], axis=0)

        counter = 0
        converged: bool = False
        while not converged:
            counter += 1

            if counter >= max_iters:
                print('Failed to converge within maximum allowed iterations.')
                break
            print(counter)
            min_dist_idx = self.assign()
            converged = self.check_convergence(min_dist_idx)
            if converged:
                self.history2d = self.history2d[1:]
                break
            self.history2d = np.concatenate((self.history2d, np.expand_dims(min_dist_idx, axis=0)))
            self.update(min_dist_idx)

    def assign(self):
        distances = np.transpose(np.array([np.linalg.norm(self.data - self.means[n], axis=1) for n in range(self.n_means)]))
        return np.argmin(distances, axis=1)

    def update(self, min_dist_idx):
        for n in range(self.n_means):
            self.means[n] = np.mean(self.data[np.where(min_dist_idx == n)], axis=0)

    def check_convergence(self, min_dist_idx):
        return np.all(np.equal(min_dist_idx, self.history2d[-1]))

    def save_experiment(self, path):
        for idx, snapshot in enumerate(self.history2d):
            plt.figure()
            for n in range(self.n_means):
                plt.scatter(self.data[np.where(snapshot == n), 0], self.data[np.where(snapshot == n), 1])
                plt.title(f'Iteration #{idx}')
                plt.savefig(f'{path}iter{idx}.png')
