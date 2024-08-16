import os
import argparse
import torch
import numpy as np
from utils import (
    get_pathway_mark,
    post_processing,
)
import warnings
warnings.filterwarnings('ignore')
from logger_utils import initialize_exp
np.random.seed(42)

from dataholder import CellData
from knowledge import LRPKGManager
from scheduler import TaskScheduler
from core import kgot_per_cell_pair, run_permutation_test


def get_args():
    #settings
    parser = argparse.ArgumentParser()
    # input data
    parser.add_argument('--cell_data_file', type=str, required=True)
    parser.add_argument('--cell_type_file', type=str, required=True)
    # cell level
    parser.add_argument('-c1', '--cell_type_1', type=str, nargs='+')
    parser.add_argument('-c2', '--cell_type_2', type=str, nargs='+')
    # knowledge level
    parser.add_argument('--use_directed_pathways', action='store_true', default=False)
    parser.add_argument('--add_reversed', action='store_true', default=False)
    # permutation
    parser.add_argument('--times', type=int, default=100)
    parser.add_argument('--p_threshold', type=float, default=0.01)
    # opt
    parser.add_argument('--num_iter_max', type=int, default=1000)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    parser.add_argument('--nb_dummies', type=int, default=1)
    parser.add_argument('--use_openmp', action='store_true', default=False)
    # subtasks level
    parser.add_argument('--directed_cell_pairs', action='store_true', default=False)
    parser.add_argument('--no_self_loop', action='store_true', default=False)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    # speed up
    parser.add_argument('--ncores', type=int, default=4)
    # output and save
    parser.add_argument('--dump_path', type=str, default='../results')
    parser.add_argument('--exp_name', type=str, default='test', 
                        help='Claim your purpose before running experiments.')
    parser.add_argument('--exp_id', type=str, default='',
                        help='This is used in logger utils for initializing experiment settings, including folders for storing results and dumping the training log. If not given, we generate random string as unique identifier. ')
    parser.add_argument('--mode', type=str, default='prediction', choices=['prediction', 'permutation', 'both'])
    parser.add_argument('--cell_samples_threshold', type=int, default=20,
                        help='A non-trivial subtask requires at least cell_samples_threshold (default 20) cells. In other words, a cell type will only be considered if it contains more than 20 cells in a specific cell type. This rule applies to both prediction and permutation test by default. We suggest you to increase this threshold if you have more data available.')
    parser.add_argument("--expr_drop_ratio", type=float, default=0.0,
                        help="Fraction of the lowest-expressing genes to drop from the analysis, based on total expression levels across both source and target datasets. This threshold is specified as a fraction (between 0 and 1), where 1 would mean dropping all genes and 0 means no genes are dropped. For instance, a value of 0.1 means dropping the bottom 10(%) of genes based on their total expression.")
    parser.add_argument("--edge_drop_ratio", type=float, default=0.0,
                    help="Specifies the fraction of edges to be randomly dropped from the Ligand-Receptor-Pathway Knowledge Graph. This parameter is expressed as a fraction between 0 and 1, where 0 indicates no edges are dropped and 1 would mean all edges are dropped. For example, setting this to 0.1 would drop 10(%) of the edges at random, useful for testing the robustness of network-based analyses under conditions of varying data completeness.")
    parser.add_argument("--type_drop_ratio", type=float, default=0.0,
                        help="Specifies the fraction of unique pathway types to be randomly dropped from the Ligand-Receptor-Pathway Knowledge Graph. This ratio is expressed as a fraction from 0 to 1, where 0 means no pathway types are dropped, and 1 would imply dropping all pathway types. This parameter can be used to assess the impact of pathway diversity on the performance of pathway-based analyses.")
    parser.add_argument("--cell_drop_ratio", type=float, default=0.0,
                    help="Percentage of cells to drop from the dataset (as a fraction between 0 and 1). When set to 0, no cells are dropped. This parameter allows for testing the robustness of models to variations in the number of available cells.")
    parser.add_argument("--scale", type=int, choices=[1, 2, 3, 4], default=1, help="Scaling factor for transportation mass requirement (default: 1).")

    args = parser.parse_args()

    # checking 
    if args.edge_drop_ratio > 0 and args.type_drop_ratio > 0:
        print(f"Edge Drop Ratio: {args.edge_drop_ratio}")
        print(f"Type Drop Ratio: {args.type_drop_ratio}")
        raise ValueError("Operation Forbidden: Simultaneous non-zero values for both edge drop ratio and type drop ratio are not allowed. Please set either edge drop ratio or type drop ratio to zero before proceeding.")

    return args


if __name__ == '__main__':
    # init
    args = get_args()
    args.use_gpu = args.use_gpu & torch.cuda.is_available()
    logger, exp_folder = initialize_exp(args, highlight=True)
    args.exp_folder = exp_folder
    assert args.p_threshold < 0.1, f'p should be close to zero, but got {args.p_threshold}'
    # known_pairs_file = os.path.join(base_dir, '../data', 'known_lr_pairs_in_datasets.csv')
    # known_pairs = pd.read_csv(known_pairs_file, usecols=range(0,5))
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '../data')

    cell_holder = CellData(args.cell_data_file, args.cell_type_file, args.cell_drop_ratio)
    LRPKG = LRPKGManager(
        pathway_type=get_pathway_mark(args.cell_data_file), 
        data_dir=data_dir, 
        use_directed_pathways=args.use_directed_pathways, 
        add_reversed=args.add_reversed,
        edge_drop_ratio=args.edge_drop_ratio,
        type_drop_ratio=args.type_drop_ratio
    )
    scheduler = TaskScheduler(
        cell_type_data=cell_holder.cell_type, 
        src_types=args.cell_type_1,
        tgt_types=args.cell_type_2,
        cell_samples_threshold=args.cell_samples_threshold,
        no_self_loop=args.no_self_loop
    )
    if args.mode == 'prediction':
        # compute predictions with origin data
        kgot_per_cell_pair(args, scheduler.execute_tasks, cell_holder, LRPKG)
    elif args.mode == 'permutation':
        # run permutation test with replicates
        run_permutation_test(args, scheduler.execute_tasks, cell_holder, LRPKG)
    elif args.mode == 'both':
        kgot_per_cell_pair(args, scheduler.execute_tasks, cell_holder, LRPKG)
        run_permutation_test(args, scheduler.execute_tasks, cell_holder, LRPKG)
        post_processing(args)  # merge results directly
    else:
        raise ValueError(f"Unknown mode: {args.mode}. We support modes for making prediction and/or running permutation test. "
                         f'Please check your shell script for the correct mode.')
    logger.info('Done!')

