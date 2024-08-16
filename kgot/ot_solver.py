import ot
import numpy as np
from scipy.spatial.distance import cdist
from ot_gpu import partial_gromov_wasserstein_gpu


def solver(partial_a, partial_b, pathway_info, 
        scale=1,
        use_gpu=False, 
        num_iter_max=1000, 
        tolerance=1e-7, 
        nb_dummies=1,
        num_threads=1):
    gene_list_a = partial_a.columns.tolist()
    gene_list_b = partial_b.columns.tolist()
    c1 = cdist(partial_a.T, partial_a.T, metric='correlation')
    c2 = cdist(partial_b.T, partial_b.T, metric='correlation')

    c1 = np.nan_to_num(c1, nan=2) / 2  # range = [0, 2]
    c2 = np.nan_to_num(c2, nan=2) / 2

    p = partial_a.sum(axis=0)  # ignore p /= p.sum()
    q = partial_b.sum(axis=0)  # ignore q /= q.sum()

    assert (p==0).sum() == 0, 'Marginal distribution assertion error. There are genes in partial_a with zero total expression, which may lead to computational errors or invalid results.'
    assert (q==0).sum() == 0, 'Marginal distribution assertion error. There are genes in partial_b with zero total expression, which may lead to computational errors or invalid results.'
    
    mask = (pathway_info['source'].isin(gene_list_a)) & (pathway_info['target'].isin(gene_list_b))
    G0_df = pathway_info.loc[mask, ['source', 'target']].reset_index(drop=True)
    G0_mask = np.zeros((len(gene_list_a), len(gene_list_b)))
    for i in G0_df.itertuples():
        src_idx = gene_list_a.index(i.source)
        tgt_idx = gene_list_b.index(i.target)
        G0_mask[src_idx, tgt_idx] = 1
    G0 = np.outer(p, q) * G0_mask
    
    if not use_gpu:
        _, log = ot.partial.partial_gromov_wasserstein2(
                c1, 
                c2, 
                p,
                q,
                m=min(p.sum(), q.sum()) / scale, 
                nb_dummies=nb_dummies,
                G0=G0,
                numItermax=num_iter_max,
                tol=tolerance,
                log=True,
                numThreads=num_threads)
    else:
        _, log = partial_gromov_wasserstein_gpu(
                c1, 
                c2, 
                p, 
                q,
                m=min(p.sum(), q.sum()) / scale, 
                numItermax=num_iter_max, 
                nb_dummies=nb_dummies,
                tol=tolerance,
                numThreads=num_threads)

    err = log['err']
    gw_dist = log['partial_gw_dist']
    plan = log['T']
    return err, gw_dist, plan, gene_list_a, gene_list_b

