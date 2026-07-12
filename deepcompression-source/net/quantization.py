import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def apply_weight_sharing(model, bits=5):
    """
    Applies weight sharing to the given model
    """
    for module in model.children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        if len(mat.data) == 0:
            continue
        min_ = min(mat.data)
        max_ = max(mat.data)
        n_clusters = min(2**bits, len(mat.data))
        space = np.linspace(min_, max_, num=n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, init=space.reshape(-1,1), n_init=1, algorithm="lloyd")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = new_weight
        module.weight.data = torch.from_numpy(mat.toarray()).to(dev)


