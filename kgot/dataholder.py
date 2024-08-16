import time
import numpy as np
import pandas as pd
import pickle
import logging
logger = logging.getLogger()


class CellData(object):
    def __init__(self, cell_data_file, cell_type_file, cell_drop_ratio : float = 0, verbose=True):
        self.cell_data_file = cell_data_file
        self.cell_type_file = cell_type_file
        if not 0 <= cell_drop_ratio <= 1:
            raise ValueError("cell_drop_ratio must be between 0 and 1.")
        self.cell_drop_ratio = cell_drop_ratio
        self.cell_data, self.cell_type = self.load_data_and_celltype()
        self.verbose = verbose
    
    def __repr__(self):
        return f'CellData({self.cell_data_file} with {self.cell_data.shape})'
    
    def load_data_and_celltype(self):
        start = time.time()
        if self.cell_data_file.endswith('.csv'):
            cell_data = pd.read_csv(self.cell_data_file, index_col=0).transpose()  # cell x gene
        elif self.cell_data_file.endswith('.pkl'):
            with open(self.cell_data_file, 'rb') as file_in:
                cell_data = pickle.load(file_in).transpose()
        else:
            ext = self.cell_data_file.rsplit('.')[-1]
            raise ValueError(f"Expected file extension with .csv or .pkl, but got {ext}")
        cell_type = pd.read_csv(self.cell_type_file, index_col=0)
        logger.info(f'[CellData] Loading data in {time.time() - start:.2f} seconds...')
        return cell_data, cell_type

    def subset_with_cell_name(self, cell_name_a, cell_name_b, times : int = 0):
        """
        The ground truth data for cell-cell interaction prediction will be reordered
        in an expected order. 
        [0: num_cell_a] for the first cell type and the rest of the other cell type.
        return:
            subtask_data: all cell instances including cell type A and B (cell x gene)
            subtask_data_a: cell instances of cell type A (cell x gene)
            subtask_data_b: cell instances of cell type B (cell x gene)
            replicates: list of two cell indices of cell instance random permutation
        """
        subtask_data_a = self.cell_data.loc[self.cell_type.loc[self.cell_type['Cell_type'] == cell_name_a, 'Cell']]
        subtask_data_b = self.cell_data.loc[self.cell_type.loc[self.cell_type['Cell_type'] == cell_name_b, 'Cell']]
        subtask_data = pd.concat([subtask_data_a, subtask_data_b], axis=0)

        num_cells, _ = subtask_data.shape
        num_cell_a = (self.cell_type['Cell_type'] == cell_name_a).sum()
        num_cell_b = (self.cell_type['Cell_type'] == cell_name_b).sum()
        assert num_cells == num_cell_a + num_cell_b, 'cell number mismatch, please verify your input files'
        assert num_cell_a > 1, f'Cell Type ({cell_name_a}), not enough cell instances, got {num_cell_a}.'
        assert num_cell_b > 1, f'Cell Type ({cell_name_b}), not enough cell instances, got {num_cell_b}.'
    
        if self.cell_drop_ratio > 0:
            # Calculate number of cells to keep from each partition
            num_keep_a = int(num_cell_a * (1 - self.cell_drop_ratio))
            num_keep_b = int((num_cells - num_cell_a) * (1 - self.cell_drop_ratio))

            # Directly choose which indices to keep
            keep_indices_a = np.random.choice(num_cell_a, size=num_keep_a, replace=False)
            keep_indices_b = np.random.choice(num_cells - num_cell_a, size=num_keep_b, replace=False) + num_cell_a

            # Create origin using the selected indices to keep
            origin = [
                sorted(keep_indices_a.tolist()),  # Sort to maintain order
                sorted(keep_indices_b.tolist())
            ]
            if self.verbose:
                logging.info(f"[CellData] Droping cells ({self.cell_drop_ratio * 100} %). {cell_name_a} (#cells={len(origin[0])}) ==> {cell_name_b} (#cells={len(origin[1])})")
        else:
            # Default origin without dropping any cells
            origin = [list(range(0, num_cell_a)), list(range(num_cell_a, num_cells))]

        # Prepare replicates
        replicates = []
        if times > 0:
            for i in range(times):
                perm = np.random.permutation(num_cells)
                part_a = perm[:num_cell_a]
                part_b = perm[num_cell_a:]

                if self.cell_drop_ratio > 0:
                    # Select to keep directly in replicates
                    part_a_kept = np.random.choice(part_a, size=num_keep_a, replace=False)
                    part_b_kept = np.random.choice(part_b, size=num_keep_b, replace=False)
                    replicates.append([sorted(part_a_kept.tolist()), sorted(part_b_kept.tolist())])
                else:
                    replicates.append([part_a.tolist(), part_b.tolist()])
        else:
            replicates = []

        return subtask_data, origin, replicates


class TaskData:
    def __init__(self, src_data, tgt_data, expr_drop_ratio=0, verbose=True):
        """
        Initializes the TaskData object with source and target datasets along with optional parameters for expression ratio
        threshold and verbosity. Prepares the data by removing genes with zero expression and applying an expression
        level threshold.

        Parameters:
        src_data (pd.DataFrame, cell x gene): DataFrame containing gene expression data for the source dataset.
        tgt_data (pd.DataFrame, cell x gene): DataFrame containing gene expression data for the target dataset.
        expr_ratio_threshold (float): Threshold for filtering based on expression ratio (not used in this version).
        verbose (bool): If True, print detailed log messages about the processing steps.
        """
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.expr_drop_ratio = expr_drop_ratio
        self.verbose = verbose
        self.preprocess_data()

    def preprocess_data(self):
        # Remove genes with zero total expression
        self.remove_zero_expression_genes()  
        # Apply expression threshold filter
        if self.expr_drop_ratio > 0:
            self.filter_by_total_expression()

    def remove_zero_expression_genes(self):
        """
        Removes genes that have zero total expression in both the source and target datasets.
        This helps in reducing the dimensionality and improving the analysis focus on genes that are expressed.
        """
        # Calculate the sum of expressions for each gene across both datasets
        src_sum = self.src_data.sum(axis=0)
        tgt_sum = self.tgt_data.sum(axis=0)
        # Filter out genes that have zero expression in both datasets
        # nonzero_genes = (src_sum > 0) | (tgt_sum > 0)
        self.src_data = self.src_data.loc[:, src_sum > 0]
        self.tgt_data = self.tgt_data.loc[:, tgt_sum > 0]
        if self.verbose:
            logging.info(f"[TaskData] Removing zero-expression genes. Remaining genes - src: {self.src_data.shape[1]}, tgt: {self.tgt_data.shape[1]}")

    def filter_by_total_expression(self):
        """
        Filters out the bottom fraction of genes based on total expression levels across both source and target datasets.
        This method focuses on retaining genes that are most active across the datasets.
        """
        # Calculate total expression across both datasets for threshold determination
        total_expression = self.src_data.sum(axis=0).add(self.tgt_data.sum(axis=0), fill_value=0)  # avoiding 500 + NaN = NaN
        # Determine the threshold for the bottom 'fraction' of expression
        # Convert fraction to percentile
        threshold = np.percentile(total_expression, self.expr_drop_ratio * 100)  
        # Filter out genes below this threshold
        filter_mask = total_expression >= threshold
        self.src_data = self.src_data.loc[:, filter_mask]
        self.tgt_data = self.tgt_data.loc[:, filter_mask]
        
        if self.verbose:
            logging.info(f"[TaskData] Filtered by total expression. {self.expr_drop_ratio * 100}% threshold applied. Remaining genes - src: {self.src_data.shape[1]}, tgt: {self.tgt_data.shape[1]}")

    def filter_by_pathway(self, pathway_info):
        """
        Filters source and target datasets to include only genes listed in the provided pathway information.
        This ensures that the analysis focuses on genes involved in specified pathways.

        Parameters:
        pathway_info (pd.DataFrame): DataFrame containing information about pathways, including 'source' and 'target' gene columns.
        """
        assert isinstance(pathway_info, pd.DataFrame), 'Pathway information must be provided in a Pandas DataFrame.'
        # Combine lists of source and target genes from the pathway information
        gene_list = pathway_info['source'].tolist() + pathway_info['target'].tolist()
        gene_list = sorted(set(gene_list))  # Remove duplicates and sort

        # Filter source and target data to only include genes in the pathway list
        partial_a = self.src_data.loc[:, self.src_data.columns.isin(gene_list)]
        partial_b = self.tgt_data.loc[:, self.tgt_data.columns.isin(gene_list)]

        return partial_a, partial_b
