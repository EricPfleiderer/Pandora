

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


# Experiment
from src.clustering.KMeans import KMeans
import matplotlib.pyplot as plt
from src.data import sample_multinormals
from src.utils import stitch_images

n_means = 7  # Hyperparameter
n_dists = 7
save_path = 'temp/'

synth_data, randomised_data = sample_multinormals(n_dists=n_dists, n_points=50, loc_bounds=(-20, 20), scale_bounds=(1, 5))

kmeans = KMeans(randomised_data, n_means=n_means)
kmeans.save_experiment(path=save_path)
stitch_images(path_to_imgs=save_path, video_name='kmeans.avi')

plt.figure()
for idx, data in enumerate(synth_data):
    plt.scatter(data[:, 0], data[:, 1])
plt.title('Solution')
plt.savefig(save_path + 'sol.png')