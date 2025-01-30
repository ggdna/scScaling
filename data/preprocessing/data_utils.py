import scanpy as sc
import numpy as np
import numba

def split_ad(adata, frac=0.8):
    """
    split anndata into ad1 ad2 with ad1 getting frac random cells
    """
    N_cells = len(adata)
    split_ind = int((N_cells*frac)//1)
    # print(split_ind)
    inds = np.arange(N_cells)
    np.random.shuffle(inds)
    return adata[inds[:split_ind]].copy(), adata[inds[split_ind:]].copy()

@numba.njit(cache=True)
def downsample_array(col, quality):

    cumcounts = col.cumsum()
    total = cumcounts[-1]
    target = int(total*quality)

    sample = np.random.choice(np.int_(total), target, replace=False)
    sample.sort()

    col[:] = 0
    geneptr = 0
    for count in sample:
        while count >= cumcounts[geneptr]:
            geneptr += 1
        col[geneptr] += 1
    return col

def get_ad_with_quality(adata, quality):
    """
    quality is fraction of total counts to retain
    """
    ad = adata.copy()

    if quality < 1:
        col = downsample_array(ad.X.data.astype(int), quality)
        ad.X.data = col
    
    ad.layers["counts"] = ad.X.copy()  # preserve counts
    ad.raw = ad
    
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)    
    sc.pp.filter_genes(adata, min_counts=3)

    sc.pp.highly_variable_genes(
        ad,
        n_top_genes=750,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        span=1
    )

    return ad