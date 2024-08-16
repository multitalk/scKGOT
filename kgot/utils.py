import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import logging
import json
import pickle
from tqdm import tqdm
import glob
logger = logging.getLogger()


def snapshot(obj, path):
    # save predictions
    if isinstance(obj, pd.DataFrame):
        if all([pd.api.types.is_sparse(i) for i in obj.dtypes]):
            obj.to_pickle(path)
        else:
            obj.to_csv(path, index=False, header=True, encoding='utf-8')
    elif isinstance(obj, dict):
        json.dump(obj, path)
    else:
        raise NotImplementedError(f'Not supported type, got {type(obj)}')


def get_pathway_mark(cell_data_file):
    if 'human' in cell_data_file:
        mark = 'human'
    elif 'mouse' in cell_data_file:
        mark = 'mouse'
    else:
        raise ValueError(f"We expected cell data file contains keyword 'human' or 'mouse' to decide the background knowledge, but got {cell_data_file}.")
    return mark


def post_processing(args):
    folder = args.exp_folder
    p_threshold = args.p_threshold
    for p in glob.glob(f'{folder}/*/*'):
        # ==========================
        # Gene Importance Comparison
        # ==========================
        lr_pred = pd.read_csv(f'{p}/lr_predictions.csv')
        lr_perm = pickle.load(open(f'{p}/lr_permutation.pkl', 'rb'))
        logger.info(f'Shape of prediction: {lr_pred.shape}')
        logger.info(f'Shape of permutation: {lr_perm.shape}')
        if not lr_pred.empty:
            lr_pred = lr_pred[lr_pred['is_ligand'] & lr_pred['is_receptor']]
            lr_pred['rank'] = lr_pred['score'].rank(ascending=False).astype(int)
            res = pd.merge(lr_pred, lr_perm, on=['src_gene', 'tgt_gene'], how='left')
            logger.info(f'Shape of merged: {res.shape}')
            columns = [i for i in res.columns if i.startswith('score_')]
            pred_score = res['score'].values[:, np.newaxis]
            perm_score = res[columns].values
            total_perm = perm_score.shape[1]
            res['p_value'] = (pred_score < perm_score).sum(axis=1)[:, np.newaxis] / total_perm  # larger the better
            res = res.loc[res['p_value'] <= p_threshold, ['src_gene', 'tgt_gene', 'score', 'rank', 'p_value']]
            res['rank'] = res['score'].rank(ascending=False).astype(int)
        else:
            res = pd.DataFrame(data=None, columns=['src_gene', 'tgt_gene', 'score', 'rank', 'p_value'])
        # result to respective folder
        snapshot(res, f'{p}/lr_final_results.csv')
        # ================================
        # Knowledge Discrepancy Comparison
        # ================================
        pw_pred = pd.read_csv(f'{p}/pw_predictions.csv')
        pw_perm = pd.read_csv(f'{p}/pw_permutation.csv')
        if not pw_pred.empty:
            res = pd.merge(pw_pred, pw_perm, on='pathway_name', how='left')
            columns = [i for i in res.columns if i.startswith('discrepancy_')]
            pred_score = res['discrepancy'].values[:, np.newaxis]
            perm_score = res[columns].values
            total_perm = perm_score.shape[1]
            res['p_value'] = (pred_score > perm_score).sum(axis=1)[:, np.newaxis] / total_perm  # smaller the better
            res = res.loc[res['p_value'] <= p_threshold, ['pathway_name', 'discrepancy', 'rank', 'p_value']]
            res['rank'] = res['discrepancy'].rank(ascending=True).astype(int)
        else:
            res = pd.DataFrame(data=None, columns=['pathway_name', 'discrepancy', 'rank', 'p_value'])
        # result to respective folder
        snapshot(res, f'{p}/pw_final_results.csv')


def create_subfolders(exp_folder, cname_a, cname_b, cnum_a, cnum_b):
    # create subfolder for sub tasks
    sub_folder = f'{exp_folder}/{cname_a}_{cnum_a}/{cname_b}_{cnum_b}'.replace(' ', '_')
    os.makedirs(sub_folder, exist_ok=True)
    return sub_folder

