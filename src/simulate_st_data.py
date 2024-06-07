from sklearn.decomposition import PCA
from copy import deepcopy
from pathlib import Path
import squidpy as sq
import pandas as pd
import scanpy as sc
import plotnine as p9
from PIL import Image
import numpy as np
import random
import json

from .utils import multi_modal_simulator

def generate_samples(main_dir,
                     sample_names,
                     n_clusters=5,
                     n_points=2500,
                     k=20,
                     sigma_1=0.5,
                     sigma_2=0.1,
                     n_components=15,
                     decay_coef_1=0.6, decay_coef_2=0.3,
                     merge_prob=0.7):
    """
    Parameters:
          main_dir          folder where to save outputs.
          sample_names      list with string names of adatas.
          n_clusters:       number of ground truth clusters.
          n_points:         number of cells to simulate.
          n_components:     Number of components to keep.
          d_1:              dimension of features for transcript modality.
          d_2:              dimension of features for morphological modality.
          k:                dimension of latent code to generate simulate data (for both modality)
          sigma_1:          variance of gaussian noise for transcript modality.
          sigma_2:          variance of gaussian noise for morphological modality.
          decay_coef_1:     decay coefficient of dropout rate for transcript modality.
          decay_coef_2:     decay coefficient of dropout rate for morphological modality.
          merge_prob:       probability to merge neighbor clusters for the generation of modality-specific
                            clusters (same for both modalities)
    """

    Path(f"{main_dir}/data/image").mkdir(parents=True, exist_ok=True)
    Path(f"{main_dir}/data/meta").mkdir(parents=True, exist_ok=True)
    Path(f"{main_dir}/data/h5ad").mkdir(parents=True, exist_ok=True)
    Path(f"{main_dir}/data/image_features").mkdir(parents=True, exist_ok=True)

    for name in sample_names:
        adata = simulate_adata(main_dir, 
                               name,
                               n_clusters=n_clusters,
                               n=n_points,
                               k=k,
                               sigma_1=sigma_1, sigma_2=sigma_2,
                               n_components=n_components,
                               decay_coef_1=decay_coef_1, decay_coef_2=decay_coef_2,
                               merge_prob=merge_prob)

        sq.pl.spatial_scatter(adata, color="ground_truth", size=0.5)


def find_closest_factors(area):

    a, b = 1, 1

    while True:

        if a * b == area:
            return a, b
        elif a * b < area and (a + 1) * (b + 1) > area and (a + 1) * b > area:
            return a + 1, b
        elif a * b > area and (a - 1) * b < area:
            return a, b

        a = a + 1
        b = b + 1

    return a, b


def generate_grid_coord(a, b):
    coords = []
    for x in range(a):
        for y in range(b):
            coords.append([y, x])
    return np.array(coords)


def simulate_adata(main_dir,
                   adata_name,
                   n_clusters=10,
                   n=1000,
                   d_1=500,
                   d_2=500,
                   k=100,
                   sigma_1=0.5,
                   sigma_2=0.5,
                   decay_coef_1=0.5,
                   decay_coef_2=0.1,
                   n_components=15,
                   merge_prob=0.7):

    data = multi_modal_simulator(n_clusters, n,
                                 d_1, d_2,
                                 k,
                                 sigma_1, sigma_2,
                                 decay_coef_1, decay_coef_2,
                                 merge_prob)

    data_a = data['data_a_dropout']
    data_b = data['data_b_dropout']
    label_a = data['data_a_label'].astype(str)
    label_b = data['data_b_label'].astype(str)
    label_true = data['true_cluster'].astype(str)

    data_a = data_a[np.argsort(label_true), :]
    data_b = data_b[np.argsort(label_true), :]
    label_true = label_true[np.argsort(label_true)]

    adata = sc.AnnData(data_a)
    adata.obs["ground_truth"] = label_true

    adata.obsm["X_pca"] = PCA(n_components=n_components).fit_transform(data_a)
    image_features = PCA(n_components=n_components).fit_transform(data_b)
    adata.obsm["image"] = image_features
    np.save(f"{main_dir}/data/image_features/{adata_name}_inception.npy", image_features)

    a, b = find_closest_factors(n)
    print(a, b)
    coord = generate_grid_coord(a, b)

    adata.obs["x_array"] = coord[:, 0][:len(adata)] + 5
    adata.obs["y_array"] = coord[:, 1][:len(adata)] + 5

    adata.obs["x_pixel"] = adata.obs["x_array"]
    adata.obs["y_pixel"] = adata.obs["y_array"]
    adata.obs["Barcode"] = np.arange(0, len(adata))
    adata.obs["Barcode"] = adata.obs["Barcode"].apply(lambda x: f"cell_{x}")
    adata.obs["index"] = adata.obs["Barcode"]
    adata.obs.set_index("index", inplace=True)
    image = 255 * np.ones((b + 10, a + 10, 3), np.uint8)
    Image.fromarray(image).save(f"{main_dir}/data/image/{adata_name}.png")

    adata.obsm['spatial'] = adata.obs[["y_pixel", "x_pixel"]].values
    # create 'spatial' entries
    adata.uns['spatial'] = dict()
    adata.uns['spatial']['library_id'] = dict()
    adata.uns['spatial']['library_id']['images'] = dict()
    adata.uns['spatial']['library_id']['images']['hires'] = image

    adata.write_h5ad(f"{main_dir}/data/h5ad/{adata_name}.h5ad")

    # general meta info about sample
    json_info = {"SAMPLE": adata_name, "spot_diameter_fullres": 1, "dot_size": 0.5}
    json_info

    with open(f"{main_dir}/data/meta/{adata_name}.json", 'w') as f:
        f.write(json.dumps(json_info))

    print(f"{adata_name} created...")
    return adata
