import pandas as pd
import logging
logger = logging.getLogger()


class TaskScheduler(object):
    def __init__(self, cell_type_data, src_types, tgt_types, 
                 cell_samples_threshold=5,
                 no_self_loop=True):
        self.cell_type_data = cell_type_data
        self.src_types = src_types
        self.tgt_types = tgt_types
        self.cell_samples_threshold = cell_samples_threshold
        self.no_self_loop = no_self_loop

        self.default_tasks = self.schedule_all_tasks()
        self.execute_tasks = self.task_filter(self.default_tasks)
        
    def schedule_all_tasks(self):
        """
        As of Pandas 1.2.0, 
        there is a how='cross' option in pandas.merge() 
        that produces the Cartesian product of the columns.
        """
        cell_count = self.cell_type_data.groupby('Cell_type').count()
        cell_count = cell_count.reset_index(drop=False)
        cell_count_cross = pd.merge(cell_count, cell_count, how='cross', suffixes=['_a', '_b'])
        cell_count_cross['key'] = cell_count_cross.apply(lambda x: frozenset(x[['Cell_type_a', 'Cell_type_b']]), axis=1)
        if self.no_self_loop:
            cell_count_cross = cell_count_cross.loc[
                    cell_count_cross['key'].apply(lambda x: len(x) == 2), :]
            cell_count_cross = cell_count_cross.reset_index(drop=True)
        # reorder and strip useless columns
        cell_count_cross = cell_count_cross.loc[:, ['Cell_type_a', 'Cell_type_b', 'Cell_a', 'Cell_b']]
        return cell_count_cross.values.tolist()

    def task_filter(self, tasks):
        if self.src_types is not None and self.tgt_types is not None:
            # if cell name specified, skip others
            final_tasks = []
            for task in tasks:
                cell_type_a, cell_type_b, cell_num_a, cell_num_b = task
                if cell_type_a in self.src_types and cell_type_b in self.tgt_types:
                    if cell_num_a <= self.cell_samples_threshold or cell_num_b <= self.cell_samples_threshold:
                        logger.info(f'Skipping {cell_type_a} ==> {cell_type_b} due to the lack of the number of cells (<={self.cell_samples_threshold})')
                        continue
                    final_tasks.append(task)
            if not len(final_tasks):
                cts = list(sorted(set(self.cell_type_data["Cell_type"].tolist())))
                raise ValueError(f'No cell pairs found for the specified cell types. '
                                f'We expected the following cell types: {cts}, '
                                f'but got {self.src_types} and {self.tgt_types}.')
            return final_tasks
        elif self.src_types is not None or self.tgt_types is not None:
            raise ValueError('Please specify both cell types for a specific subtask.')
        else:
            logger.info('No cell type specified, we will run all subtasks.')
            return tasks
    