import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dataholder import TaskData
from ot_solver import solver
from utils import (
    snapshot,
    create_subfolders,
)
import time
from datetime import timedelta
import logging
logger = logging.getLogger()


def meta_prediction(args, task_holder, pathway_info):
    '''scKGOT functionality: predicting the results
    
    We predict the ligand-receptor pairs per active pathway,
    given a subtask of scRNA-seq dataset.
    '''
    # initial outputs
    empty_df = pd.DataFrame(columns=['ligand', 'receptor', 'ratio', 'name', 'gw_dist', 'num_ga', 'num_gb'])
    # handle pathway information
    pathway_name = pathway_info['pathway'].unique()[0]
    partial_a, partial_b = task_holder.filter_by_pathway(pathway_info)
    num_threads = 'max' if args.use_openmp else 1
    try:
        err, gw_dist, plan, gene_list_a, gene_list_b = solver(partial_a, partial_b, pathway_info,
                scale=args.scale,
                use_gpu=args.use_gpu,
                num_iter_max=args.num_iter_max,
                tolerance=args.tolerance,
                nb_dummies=args.nb_dummies,
                num_threads=num_threads)
        # TODO this may be problematic?
        if len(err) == 1:
            return empty_df, (None, None)
    except Exception as e:
        print(f'Error from func:meta_prediction when running the solver. Complete error message: \n{e}')
        return empty_df, (None, None)
    # decode the plan
    rowrow, colcol = plan.nonzero()
    summary_info = []
    total = plan.sum()
    for x, y in zip(rowrow, colcol):
        value = plan[x, y]
        src = gene_list_a[x]
        tgt = gene_list_b[y]
        summary_info.append([src, tgt, value / total])
    summary_info = pd.DataFrame(summary_info, columns=['src_gene', 'tgt_gene', 'ratio'])
    summary_info['name'] = pathway_name
    summary_info['gw_dist'] = gw_dist
    summary_info['num_ga'] = len(gene_list_a)
    summary_info['num_gb'] = len(gene_list_b)
    
    if not summary_info.empty:
        return summary_info, (pathway_name, gw_dist)
    else:
        return empty_df, (pathway_name, gw_dist)
        

def solve_kgot_per_pair(args, subtask_data, index_a, index_b, LRPKG):
    '''scKGOT functionality: predicting the results.
    We first apply filters for genes and pathways to reduce the computational complexity.
    Then we predict the ligand-receptor pairs per active pathway, given a subtask of scRNA-seq dataset.
    Finally, we combine the results from all subtasks to generate the final results.

    Args:
        args: arguments from command line
        subtask_data: scRNA-seq dataset with cells as rows and genes as columns
        index_a: index of cell type A for selecting the corresponding cells in subtask_data
        index_b: index of cell type B for selecting the corresponding cells in subtask_data
        LRPKGManager: pathway information 
    Returns:
    '''
    job_data_a = subtask_data.iloc[index_a]
    job_data_b = subtask_data.iloc[index_b]
    task_holder = TaskData(job_data_a, job_data_b, args.expr_drop_ratio, verbose=True)
    
    genes_in_cdata = set(task_holder.src_data.columns.tolist() + task_holder.tgt_data.columns.tolist())
    high_coverages = LRPKG.get_pathway_name_with_high_coverage(genes_in_cdata)
    name2pathway = LRPKG.name2pathway
    ligands = LRPKG.ligands
    receptors = LRPKG.receptors
    if not len(high_coverages):
        logger.info('[Core] Not enough pathways, skipping current task. Please check your inputs.')
        lr_pairs_out = pd.DataFrame(columns=['src_gene', 'tgt_gene', 'score', 'rank', 'is_ligand', 'is_receptor'])
        pathway_out = pd.DataFrame(columns=['pathway_name', 'discrepancy', 'rank'])
        all_results_df = pd.DataFrame(columns=['src_gene', 'tgt_gene', 'ratio', 'name', 'gw_dist', 'num_ga', 'num_gb'])
        return lr_pairs_out, pathway_out, all_results_df
    else:
        dist_results = Parallel(n_jobs=args.ncores, verbose=0)(
                delayed(meta_prediction)(args, task_holder, name2pathway[name]) 
                for name in high_coverages
        )
        res_df, discrepancy = zip(*dist_results)
        all_results_df = pd.concat(res_df, axis=0, ignore_index=True)
        if all_results_df.empty:
            # check if any result is returned
            lr_pairs_out = pd.DataFrame(columns=['src_gene', 'tgt_gene', 'score', 'rank'])
        else:
            # summarize gene importance for ligand-receptor pairs
            all_results_df['weight'] = all_results_df['gw_dist'].rank(pct=True, ascending=False)
            lr_pairs_out = all_results_df.groupby(['src_gene', 'tgt_gene'])[['weight', 'ratio']].apply(
                lambda x: (x['ratio'] * x['weight']).sum()).to_frame(name='score')
            lr_pairs_out = lr_pairs_out.reset_index(level=['src_gene', 'tgt_gene'])
            lr_pairs_out = lr_pairs_out.dropna(axis=0, ignore_index=True)
            lr_pairs_out['rank'] = lr_pairs_out['score'].rank(ascending=False)  # bigger is better
            lr_pairs_out['rank'] = lr_pairs_out['rank'].astype(int)
            lr_pairs_out['is_ligand'] = lr_pairs_out['src_gene'].isin(ligands)
            lr_pairs_out['is_receptor'] = lr_pairs_out['tgt_gene'].isin(receptors)
        # summarize knowledge discrepancy information for pathways
        pathway_out = pd.DataFrame(discrepancy, columns=['pathway_name', 'discrepancy'])
        pathway_out = pathway_out.dropna(axis=0, ignore_index=True)
        pathway_out['rank'] = pathway_out['discrepancy'].rank(ascending=True)  # smaller is better
        pathway_out['rank'] = pathway_out['rank'].astype(int)
        return lr_pairs_out, pathway_out, all_results_df


def kgot_per_cell_pair(args, subtasks_list, cell_holder, LRPKG):
    '''scKGOT functionality: running KGOT algorithm on each cell pair.
    
    The result is automatically ensemble by the scKGOT algorithm, 
    which is further snapshoted to the corresponding folder.
    '''
    total = len(subtasks_list)
    logger.info(f'[Core] Using {args.ncores} cores for parallel computing')
    logger.info(f'[Core] Using cell_data_file: {cell_holder.cell_data_file}')
    for idx, (cname_a, cname_b, cnum_a, cnum_b) in enumerate(subtasks_list):
        logger.info(f'[JOB - Prediction]: [{idx+1}/{total}] {cname_a} (#cells={cnum_a}) ==> {cname_b} (#cells={cnum_b})')
        job_data, origin, _ = cell_holder.subset_with_cell_name(cname_a, cname_b, times=0)
        lr_pred, pw_pred, source_df = solve_kgot_per_pair(args, job_data, origin[0], origin[1], LRPKG)
        lr_pred = lr_pred.sort_values(by='rank', ascending=True)
        pw_pred = pw_pred[~pw_pred['pathway_name'].isnull()]
        pw_pred = pw_pred.sort_values(by='rank', ascending=True)

        sub_folder = create_subfolders(args.exp_folder, cname_a, cname_b, cnum_a, cnum_b)
        snapshot(lr_pred, f'{sub_folder}/lr_predictions.csv')
        snapshot(pw_pred, f'{sub_folder}/pw_predictions.csv')
        snapshot(source_df, f'{sub_folder}/source_df.csv')
    logger.info(f'[Job - Prediction] All subtasks completed!')


def run_permutation_test(args, subtasks_list, cell_holder, LRPKG):
    '''scKGOT functionality: running permutation test
    
    We run permutation test to evaluate the significance of the results,
    given a subtask of scRNA-seq dataset.
    
    Specifically, we randomly shuffle the labels of the cells $N$ times,
    and try to predict the ligand-receptor pairs per active pathway,
    based on random labels.
    
    The p-value is the number of times that the results under random labels
    exceed those of the actual labels.
    '''
    total = len(subtasks_list)
    start_time = time.time()  # Start timing the operation
    logger.info(f'[Core] Using cell_data_file: {cell_holder.cell_data_file}')
    for idx, (cname_a, cname_b, cnum_a, cnum_b) in enumerate(subtasks_list):
        logger.info(f'[JOB - Permutation]: [{idx+1}/{total}] {cname_a} (#cells={cnum_a}) ==> {cname_b} (#cells={cnum_b})')
        job_data, _, replicates = cell_holder.subset_with_cell_name(cname_a, cname_b, times=args.times)
        lr_perm = []
        pw_perm = []
        for col_idx, (a, b) in enumerate(replicates):
            lr_df, pw_df, _ = solve_kgot_per_pair(args, job_data, a, b, LRPKG)
            # lr_df has columns of ligand, receptor, score, rank
            # pw_df has columns of pathway_name, discrepancy
            lr_df['identifier'] = f'score_{col_idx}'
            pw_df['identifier'] = f'discrepancy_{col_idx}'
            lr_perm.append(lr_df)
            pw_perm.append(pw_df)
            if col_idx % 10 == 0:  # Log a message every 10 iterations
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / (col_idx + 1)) * len(replicates)
                remaining_time = estimated_total_time - elapsed_time
                eta = str(timedelta(seconds=int(remaining_time)))
                logger.info(f'[JOB - Permutation] Progress: [{col_idx+1:03}/{len(replicates)}] ETA: {eta}')
            
        lr_perm = pd.concat(lr_perm, axis=0, ignore_index=True)
        lr_perm = lr_perm.astype({
            'src_gene': str, 
            'tgt_gene': str, 
            'score': np.float64,
            'rank': np.int32, 
            'identifier': str})
        lr_pivot = pd.pivot_table(lr_perm, values='score', index=['src_gene', 'tgt_gene'], columns='identifier')
        lr_pivot = lr_pivot.fillna(value=0)
        lr_pivot = lr_pivot.astype(pd.SparseDtype("float", 0))  # one-liner to convert to sparse df

        pw_perm = pd.concat(pw_perm, axis=0)
        pw_perm = pw_perm[~pw_perm['pathway_name'].isnull()]
        pw_perm = pw_perm.astype({
            'pathway_name': str, 
            'discrepancy': np.float64, 
            'rank': np.int32, 
            'identifier': str})
        pw_pivot = pd.pivot_table(pw_perm, values='discrepancy', index='pathway_name', columns='identifier')
        pw_pivot = pw_pivot.reset_index()
        pw_pivot = pw_pivot.fillna(value=1e9)   # the bigger the worse

        sub_folder = create_subfolders(args.exp_folder, cname_a, cname_b, cnum_a, cnum_b)
        snapshot(lr_pivot, f'{sub_folder}/lr_permutation.pkl')
        snapshot(pw_pivot, f'{sub_folder}/pw_permutation.csv')
    logger.info(f'[Job - Permutation] All subtasks completed!')
