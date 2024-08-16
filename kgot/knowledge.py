import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import logging
logger = logging.getLogger()


class LRPKGManager(object):
    def __init__(self, pathway_type,
                 data_dir='../data',
                 use_directed_pathways=True,
                 add_reversed=False,
                 edge_drop_ratio : float = 0,
                 type_drop_ratio : float = 0,
                 verbose=True):
        assert pathway_type in ['human', 'mouse']
        self.pathway_type = pathway_type        
        self.data_dir = data_dir
        self.use_directed_pathways = use_directed_pathways
        self.add_reversed = add_reversed
        self.edge_drop_ratio = edge_drop_ratio
        self.type_drop_ratio = type_drop_ratio
        self.verbose = verbose

        self.pathways, self.name2pathway = self._load_pathways()
        self.known_lr_pairs, self.ligands, self.receptors = self._load_lr_pairs()
        
    def _load_pathways(self):
        # pathway_file = os.path.join(self.data_dir, 'pathways', f'{mark}_pathways.csv')
        pathway_file = os.path.join(self.data_dir, f'{self.pathway_type}_pathways_v2.csv')
        pathways = self._get_pathways(pathway_file)
        if self.use_directed_pathways:
            pathways = pathways[pathways['direction'] == 'directed']
        if self.edge_drop_ratio > 0:
            pathways = self.drop_edges(pathways, self.edge_drop_ratio)
        if self.type_drop_ratio > 0:
            pathways = self.drop_pathway_types(pathways, self.type_drop_ratio)
            
        # name2pathway
        if self.edge_drop_ratio > 0:
            pathway_pkl = os.path.join(self.data_dir, 'pickle_data', f'{self.pathway_type}_pathways_v2_edgedrop_{self.edge_drop_ratio}.pkl')
        elif self.type_drop_ratio > 0:
            pathway_pkl = os.path.join(self.data_dir, 'pickle_data', f'{self.pathway_type}_pathways_v2_typedrop_{self.type_drop_ratio}.pkl')
        else:
            pathway_pkl = os.path.join(self.data_dir, 'pickle_data', f'{self.pathway_type}_pathways_v2.pkl')
            
        if os.path.exists(pathway_pkl):
            with open(pathway_pkl, 'rb') as file_in:
                name2pathway = pickle.load(file_in)
        else:
            name2pathway = {}
            for name in tqdm(pathways['pathway'].unique().tolist()):
                name2pathway[name] = pathways[pathways['pathway'] == name]
            with open(pathway_pkl, 'wb') as file_out:
                pickle.dump(name2pathway, file_out)
        return pathways, name2pathway

    def _get_pathways(self, file_path):
        pathways = pd.read_csv(file_path, index_col=0).reset_index(drop=True)
        pathways.columns = ['source', 'target', 'pathway', 'type', 'direction', 'database']
        if self.add_reversed:
            reversed_pathway = pathways[pathways.direction == 'undirected']
            reversed_pathway.columns = ['target', 'source', 'pathway', 'type', 'direction', 'database']
            pathways = pd.concat([pathways, reversed_pathway], axis=0, ignore_index=True)
        # TODO Skip ['Metabolic pathways', 'Metabolism']
        return pathways

    def _load_lr_pairs(self):
        lr_pairs_file = os.path.join(self.data_dir, 'lr_pairs', f'{self.pathway_type}_lr_pair.csv')
        lr_pairs = pd.read_csv(lr_pairs_file, index_col=0)[['ligand', 'receptor']].reset_index(drop=True)
        ligands = frozenset(lr_pairs.ligand.tolist())
        receptors = frozenset(lr_pairs.receptor.tolist())
        return lr_pairs, ligands, receptors

    def get_pathway_name_with_high_coverage(self, genes_in_cell_data):
        high_coverages = []
        for pgroup_name, pgroup_value in self.pathways.groupby('pathway'):
            # pathway_name = pgroup_name.replace('/', '_')
            genes_in_pathway = set(pgroup_value.source.tolist() + pgroup_value.target.tolist())
            shared = len(genes_in_cell_data & genes_in_pathway)
            origin = len(genes_in_pathway)
            if origin >= 50 and shared / origin >= 0.3:
                high_coverages.append(pgroup_name)
        return high_coverages

    def drop_edges(self, df_pathways: pd.DataFrame, percent: float = 0.1):
        """
        Randomly drops a percentage of edges (rows) from the DataFrame.

        Parameters:
        - df_pathways: DataFrame containing pathway data.
        - percent: Percentage of rows to drop (default is 10%).

        Returns:
        - DataFrame with rows dropped.
        """
        # Calculate the number of rows to drop
        num_to_drop = int(len(df_pathways) * percent)
        # Create a random sample of indices to drop
        drop_indices = np.random.choice(df_pathways.index, num_to_drop, replace=False)
        # Drop the rows and return the new DataFrame
        df_dropped = df_pathways.drop(drop_indices)
        if self.edge_drop_ratio:
            logging.info(f"[LRP-KG] Dropping {self.edge_drop_ratio * 100} % edges ({len(df_pathways)} => {len(df_dropped)})")
        return df_dropped

    def drop_pathway_types(self, df_pathways: pd.DataFrame, percent: float = 0.1):
        """
        Drops entries based on a percentage of unique pathway types.

        Parameters:
        - df_pathways: DataFrame containing pathway data.
        - percent: Percentage of pathway types to drop (default is 10%).

        Returns:
        - DataFrame with selected pathway types dropped.
        """
        # Find unique pathway types
        unique_types = df_pathways['pathway'].unique()
        # Calculate the number of types to drop
        num_types_to_drop = int(len(unique_types) * percent)
        # Randomly select types to drop
        types_to_drop = np.random.choice(unique_types, num_types_to_drop, replace=False)
        # Drop rows that match the selected types
        df_dropped = df_pathways[~df_pathways['pathway'].isin(types_to_drop)]
        if self.type_drop_ratio:
            logging.info(f"[LRP-KG] Dropping {self.type_drop_ratio * 100} % pathways ({len(unique_types)} => {len(unique_types) - num_types_to_drop})")
        return df_dropped

        