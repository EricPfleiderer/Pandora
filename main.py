"""
data extraction (scraping, online APIS, regex) & formatting (pandas, numpy)

data analysis:
    -descriptive
    -hypothesis testing, p-value
    -clustering
        -Mean-shift
        -DBSCAN
        -EM Clustering with Gaussian Mixtures
        -Agglomerative Hierarchical Algorithm
    -chi-square

machine learning
    -supervised
    -unsupervised
    -reinforcement learning
    -hypertuning
    -statistical ML (KNN, Decision trees, bagging, boosting)
    -ensemble methods (Random forests, adaboosting)
"""

import matplotlib.pyplot as plt
from src.learning.unsupervised.clustering.KMeans import KMeans
from src.data import sample_multinormals
from src.utils import stitch_images, create_experiment_dir
from src.data import SupportedDatasets


def run_clustering_experiment(params: dict):

    # TODO: load from params

    n_means = 4  # Hyperparameter
    n_dists = 4
    experiment_path = create_experiment_dir(save_path='Experiments/', params=params)

    synth_data, randomised_data = sample_multinormals(n_dists=n_dists, n_points=50, loc_bounds=(-20, 20), scale_bounds=(1, 3))

    kmeans = KMeans(randomised_data, n_means=n_means, init='random')
    kmeans.save_experiment(path=experiment_path)
    stitch_images(path_to_imgs=experiment_path, video_name='kmeans.avi')

    plt.figure()
    for idx, data in enumerate(synth_data):
        plt.scatter(data[:, 0], data[:, 1])
    plt.title('Solution')
    plt.savefig(experiment_path + 'sol.png')


def run_classification_experiment(params: dict):
    pass


experiment_params = {'dataset': SupportedDatasets.MNIST,

}

run_classification_experiment(experiment_params)