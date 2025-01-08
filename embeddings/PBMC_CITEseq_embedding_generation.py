import scvi
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
import gc
import sys
import os
import pickle
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from data_utils import *

from nanoTxformer.model import nanoTxformer
from nanoTxformer.train import train_model
from nanoTxformer.utils import get_mean_pooled_embeddings

# set torch precision
torch.set_float32_matmul_precision('medium')

#########################
### control variables ###
#########################

run_nanotx_every = 2  # run nanoTx every x iterations
dataset_fractions = np.logspace(-3, 0, 6)

# input arguments
qual, r, N_dims = sys.argv[1:]
qual = float(qual)
r = int(r)
representation_dims = int(N_dims)

# load data
ad = sc.read_h5ad(f'../../scaling_playground/data/PBMC_CITEseq_Q{qual:.3f}_rep{r}.h5ad')
UMI_per_cell = ad.raw.X.sum() / len(ad)

# split data into seen and held-out
seen, held_out = split_ad(ad, frac=0.75)

# create subfolder for this size/quality
output_dir = f'PBMC_CITEseq/Q{qual:.3f}_rep{r}/'
os.makedirs(output_dir, exist_ok=True)

# save held-out cells
held_out.write(f'{output_dir}held_out_cells.h5ad')

# clean up memory
del ad
gc.collect()

# function to save embeddings
def save_embeddings(embeddings, method, frac):
    filename = f'{output_dir}{method}_frac{frac:.3f}.npy'
    np.save(filename, embeddings)

# function to save logs
def save_log(log_object, method, frac):
    filename = f'{output_dir}{method}_frac{frac:.3f}_log.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(log_object, f)

# iterate over fractions
for i, frac in enumerate(tqdm(dataset_fractions, desc='fractions', position=1, leave=False)):
    ad1, _ = split_ad(seen, frac=frac)
    gc.collect()

    ###########
    ### VAE ###
    ###########
    scvi.model.SCVI.setup_anndata(ad1, layer="counts")
    vae = scvi.model.SCVI(ad1, n_hidden=512, n_latent=representation_dims)
    vae.train(batch_size=512, logger=None, enable_progress_bar=False, early_stopping=True)
    emb = vae.get_latent_representation(held_out)
    save_embeddings(emb, "VAE", frac)
    log_object = {"elbo_train": vae.history["elbo_train"], "elbo_validation": vae.history["elbo_validation"]}
    save_log(log_object, "VAE", frac)

    ###########
    ### PCA ###
    ###########
    pca = PCA(n_components=representation_dims, svd_solver='randomized', random_state=0)
    pca.fit(np.asarray(ad1.X.todense()))
    emb = pca.transform(np.asarray(held_out.X.todense()))
    save_embeddings(emb, "PCA", frac)
    variance_explained = np.sum(pca.explained_variance_ratio_)
    log_object = {"variance_explained": variance_explained}
    save_log(log_object, "PCA", frac)

    #########################
    ### Random projection ###
    #########################
    rand_proj = GaussianRandomProjection(n_components=representation_dims)
    emb = rand_proj.fit_transform(np.asarray(held_out.X.todense()))
    save_embeddings(emb, "RandomProjection", frac)

    ##############
    ### nanoTx ###
    ##############
    if i % run_nanotx_every == 0:
        model = nanoTxformer(ad1, embed_size=128, num_heads=4, num_encoder_layers=2).cuda()
        train_losses, val_losses = train_model(model, ad1, epochs=10**4)
        emb = get_mean_pooled_embeddings(model, held_out).cpu().numpy()
        save_embeddings(emb, "nanoTxformer", frac)
        log_object = {"train_losses": train_losses, "val_losses": val_losses}
        save_log(log_object, "nanoTxformer", frac)

print("embedding generation completed! :) ")
