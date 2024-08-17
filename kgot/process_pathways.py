import argparse
import os
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from dataholder import CellData, TaskData
from knowledge import LRPKGManager
from scheduler import TaskScheduler
from utils import get_pathway_mark
from vis_helper import *
import plotly.io as pio

# Set seeds and output settings
random.seed(10086)
pio.kaleido.scope.mathjax = None

# ==============================
# Data Loading Functions
# ==============================

def load_pw_results(result_path):
    """Load pathway results from a CSV file."""
    return pd.read_csv(f'{result_path}/pw_final_results.csv')

def load_cell_data(cell_data_file, cell_type_file):
    """Load cell data and return an instance of CellData."""
    return CellData(cell_data_file, cell_type_file)

# ==============================
# Calculation Functions
# ==============================

def calculate_expression_sum(row, expr_in_gene_a, expr_in_gene_b):
    """Calculate the sum of gene expression for a given pathway."""
    source_expression = expr_in_gene_a.get(row['source'], 0)
    target_expression = expr_in_gene_b.get(row['target'], 0)
    return source_expression + target_expression

def calculate_pathway_stats(task_holder, LRPKG):
    """Calculate pathway statistics including expression sums and gene counts."""
    expr_in_gene_a = task_holder.src_data.sum(axis=0).to_dict()
    expr_in_gene_b = task_holder.tgt_data.sum(axis=0).to_dict()

    pathways = LRPKG.pathways.drop_duplicates(subset=['source', 'target', 'pathway'])
    pathways['expr_sum'] = pathways.apply(calculate_expression_sum, axis=1, 
                                          expr_in_gene_a=expr_in_gene_a, expr_in_gene_b=expr_in_gene_b)

    pathway_stats = pathways.groupby('pathway').agg({
        'expr_sum': 'sum',
        'source': 'nunique',
        'target': 'nunique'
    }).reset_index()

    pathway_stats['gene_count'] = pathway_stats['source'] + pathway_stats['target']
    pathway_stats['avg_expr'] = pathway_stats['expr_sum'] / pathway_stats['gene_count']
    return pathway_stats

# ==============================
# Plotting Functions
# ==============================

def plot_expression_distribution(pathway_stats, pred_pathway_stats, output_path, dpi):
    """Plot and save the expression distribution."""
    print('plot_expression_distribution START!')
    
    plt.figure(figsize=(12, 8))

    sns.histplot(pathway_stats['avg_expr'], bins=20, kde=True, color='#636EFA', label='Pathways in LRP-KG', alpha=0.6)
    sns.histplot(pred_pathway_stats['avg_expr'], bins=20, kde=False, color='#FFA15A', label='Pathways Predicted by scKGOT', alpha=0.8)

    mean_avg_expr = pathway_stats['avg_expr'].mean()
    mean_pred_avg_expr = pred_pathway_stats['avg_expr'].mean()

    plt.axvline(mean_avg_expr, color='#FFA15A', linestyle='dashed', linewidth=2)
    plt.axvline(mean_pred_avg_expr, color='#636EFA', linestyle='dashed', linewidth=2)

    plt.text(mean_avg_expr, plt.ylim()[1] * 0.9, f'mean = {mean_avg_expr:.2f}', color='#FFA15A', fontsize=16, ha='left')
    plt.text(mean_pred_avg_expr, plt.ylim()[1] * 0.8, f'mean = {mean_pred_avg_expr:.2f}', color='#636EFA', fontsize=16, ha='left')

    plt.legend(loc='upper right', fontsize=18)
    plt.xlabel('Average Gene Expression Level among Pathway', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(f'{output_path}/pathway_expression_distribution.pdf', dpi=dpi)
    plt.savefig(f'{output_path}/pathway_expression_distribution.png', dpi=dpi)
    plt.clf()
    print('plot_expression_distribution DONE!')

def plot_pathway_expression_and_discrepancy(selected_pathways, output_path, dpi):
    """Plot and save the pathway expression and knowledge discrepancy."""
    print('plot_pathway_expression_and_discrepancy START!')
    
    fig, ax1 = plt.subplots(figsize=(12, 16))

    bar_width = 0.35
    bar_gap = 0.1
    total_width = bar_width * 2 + bar_gap

    y_labels = selected_pathways['pathway']
    y_pos = np.arange(len(y_labels)) * total_width

    ax1.barh(y_pos - bar_width / 2, selected_pathways['gene_count'], color='#636EFA', edgecolor='black', alpha=0.6, height=bar_width, label='#Genes in a Pathway')
    
    ax2 = ax1.twiny()
    ax2.barh(y_pos + bar_width / 2, selected_pathways['weight'], color='#FFA15A', edgecolor='black', alpha=0.8, height=bar_width, label='Knowledge Discrepancy (percentile)')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, fontsize=20)

    ax1.set_xlabel('#Genes in a Pathway', fontsize=18, color='#636EFA')
    ax2.set_xlabel('Knowledge Discrepancy', fontsize=18, color='#FFA15A')

    ax1.set_xticks([0, 750, 1500, 2250, 3000])
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])

    ax1.set_ylim(min(y_pos) - total_width / 2, max(y_pos) + total_width / 2)
    ax2.grid(color='gray', linestyle='-', linewidth=0.5)

    ax1.tick_params(axis='x', which='major', labelsize=16)
    ax2.tick_params(axis='x', which='major', labelsize=16)
    plt.tight_layout()

    plt.savefig(f'{output_path}/pathway_expression_and_discrepancy.png', dpi=dpi)
    plt.savefig(f'{output_path}/pathway_expression_and_discrepancy.pdf', dpi=dpi)
    plt.clf()
    print('plot_pathway_expression_and_discrepancy DONE!')

# ==============================
# Sankey Diagram Functions
# ==============================

def load_and_process_data(data, lr_pathway_file, size=0):
    """Load and process pathway interaction data."""
    lr_pathway = pd.read_csv(lr_pathway_file, index_col=0)
    
    ligands = list(set(lr_pathway['ligand'].tolist()))
    receptors = list(set(lr_pathway['receptor'].tolist()))

    data = data[(data['src_gene'].isin(ligands)) & (data['tgt_gene'].isin(receptors))].copy(deep=True)
    
    pathway2count = data['name'].value_counts().to_dict()
    large_pathways = [k for k, v in pathway2count.items() if v > size]
    print(len(large_pathways))
    data = data[(data['name'].isin(large_pathways))].copy(deep=True)
    
    data['src_name'] = 'src_' + data['name']
    data['tgt_name'] = 'tgt_' + data['name']
    return data

def create_sankey_diagram(subdata):
    """Create a Sankey diagram for pathway interactions."""
    colormap = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    vis_df_1 = subdata[['src_name', 'src_gene', 'name', 'ratio']]
    vis_df_1['ratio'] = 1
    vis_df_1.columns = ['source', 'target', 'pathway_name', 'ratio']

    vis_df_2 = subdata[['src_gene', 'tgt_gene', 'name', 'ratio']]
    vis_df_2['ratio'] *= 80
    vis_df_2.columns = ['source', 'target', 'pathway_name', 'ratio']
    
    vis_df_3 = subdata[['tgt_gene', 'tgt_name', 'name', 'ratio']]
    vis_df_3['ratio'] = 1
    vis_df_3.columns = ['source', 'target', 'pathway_name', 'ratio']
    
    vis_df = pd.concat([vis_df_1, vis_df_2, vis_df_3])

    vis_ligands = set(subdata['src_gene'])
    vis_receptors = set(subdata['tgt_gene'])

    node2id = {n: idx for idx, n in enumerate(set(vis_df['source']) | set(vis_df['target']))}
    
    pathway_colors = {pathway: colormap[i % len(colormap)] for i, pathway in enumerate(vis_df['pathway_name'].unique())}
    
    vis_df['source_id'] = vis_df['source'].map(node2id)
    vis_df['target_id'] = vis_df['target'].map(node2id)
    vis_df['link_color'] = vis_df['pathway_name'].map(lambda x: pathway_colors[x])

    label = [i.replace('src_', '').replace('tgt_', '') for i in list(node2id.keys())] 
    node_colors = []
    
    for node in label:
        if node in pathway_colors:
            node_colors.append(pathway_colors[node]) 
        elif node in vis_ligands:
            related_pathways = vis_df[vis_df['target'] == node]['pathway_name'].unique()
            if len(related_pathways) == 1:
                node_colors.append(pathway_colors[related_pathways.item()])
            else:
                node_colors.append(random.choice(colormap))
        elif node in vis_receptors:
            related_pathways = vis_df[vis_df['source'] == node]['pathway_name'].unique()
            if len(related_pathways) == 1:
                node_colors.append(pathway_colors[related_pathways.item()])
            else:
                node_colors.append(random.choice(colormap))

    node = dict(
        label=label,
        pad=15,
        thickness=30,
        color=node_colors
    )

    link = dict(
        source=vis_df['source_id'].tolist(),
        target=vis_df['target_id'].tolist(),
        value=vis_df['ratio'].tolist(),
        color=vis_df['link_color'].tolist()
    )

    fig = go.Figure(go.Sankey(link=link, node=node))

    fig.update_layout(
        font_size=12,
        height=800,
        width=1500,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    fig.add_annotation(x=0.05, y=1.05, showarrow=False, text='<b>Pathway</b>', font_size=12)
    fig.add_annotation(x=0.35, y=1.05, showarrow=False, text='<b>Ligand</b>', font_size=12)
    fig.add_annotation(x=0.65, y=1.05, showarrow=False, text='<b>Receptor</b>', font_size=12)
    fig.add_annotation(x=0.95, y=1.05, showarrow=False, text='<b>Pathway</b>', font_size=12)

    return fig


def process_pathways(result_path, args, dpi=300):
    """Main function to process pathway data and generate visualizations."""
    print("Reading files START!")
    output_path = '../vis_output'
    os.makedirs(output_path, exist_ok=True)
    
    # Load pathway and cell data
    pw_results = load_pw_results(result_path)
    final_pathway_name = pw_results['pathway_name'].tolist()

    # Initialize and load cell data
    cell_holder = load_cell_data(args.cell_data_file, args.cell_type_file)
    LRPKG = LRPKGManager(pathway_type=get_pathway_mark(args.cell_data_file), data_dir='../data',
                         use_directed_pathways=args.use_directed_pathways, add_reversed=args.add_reversed)
    
    # Scheduler and task holder
    scheduler = TaskScheduler(cell_type_data=cell_holder.cell_type, src_types=args.cell_type_1, 
                              tgt_types=args.cell_type_2, cell_samples_threshold=args.cell_samples_threshold, 
                              no_self_loop=args.no_self_loop)
    
    # Subset data for the task
    job_data, origin, _ = cell_holder.subset_with_cell_name(args.cell_type_1, args.cell_type_2, times=0)
    task_holder = TaskData(job_data.iloc[origin[0]], job_data.iloc[origin[1]], verbose=False)

    # Calculate pathway statistics
    pathway_stats = calculate_pathway_stats(task_holder, LRPKG)

    # Plot expression distribution
    pred_pathway_stats = pathway_stats.loc[pathway_stats['pathway'].isin(final_pathway_name)]
    plot_expression_distribution(pathway_stats, pred_pathway_stats, output_path, dpi)

    # Load additional data and plot pathway expression and discrepancy
    source_df = pd.read_csv(f'{result_path}/source_df.csv')
    mapper_df = source_df[['name', 'weight']].drop_duplicates().sort_values('weight', ascending=False)
    pathway_stats['weight'] = pathway_stats['pathway'].map(dict(zip(mapper_df.name, mapper_df.weight)))
    pathway_stats['knowledge_discrepancy'] = pathway_stats['pathway'].map(dict(zip(pw_results.pathway_name, pw_results.discrepancy)))

    selected_pathways = pathway_stats.loc[pathway_stats['pathway'].isin(final_pathway_name)]
    selected_pathways = selected_pathways.sort_values('weight', ascending=True)[:35]
    plot_pathway_expression_and_discrepancy(selected_pathways, output_path, dpi)

    # Generate and save Sankey diagram
    lr_pathway_file = '../data/lr_pairs/human_lr_pair.csv'
    data = load_and_process_data(source_df, lr_pathway_file, size=args.size)
    fig = create_sankey_diagram(data)

    fig.write_image(f"{output_path}/pathway_interaction_sankey_diagram_{args.size}.png")
    fig.write_image(f"{output_path}/pathway_interaction_sankey_diagram_{args.size}.pdf")
    print("Sankey diagram saved!")


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process and visualize pathway data.")
    
    # Add arguments
    parser.add_argument('--result_path', type=str, required=True, help='Path to the result directory.')
    parser.add_argument('--dataset', type=str, default='human_placenta_20218', help='Dataset name (default: human_placenta_20218)')
    parser.add_argument('--cell_type_1', type=str, default='lymphatic endothelial cell', help='sender cell type (default: lymphatic endothelial cell)')
    parser.add_argument('--cell_type_2', type=str, default='villous cytotrophoblast', help='receiver cell type (default: villous cytotrophoblast)')
    parser.add_argument('--use_directed_pathways', action='store_true', default=True, help='Flag to use directed pathways (default: True)')
    parser.add_argument('--add_reversed', action='store_true', default=False, help='Flag to add reversed pathways (default: False)')
    parser.add_argument('--no_self_loop', action='store_true', default=True, help='Remove self-loops (default: True)')
    parser.add_argument('--cell_samples_threshold', type=int, default=1, help='Threshold for cell samples (default: 1)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for the generated figures (default: 300)')
    parser.add_argument('--size', type=int, default=1, help='Size parameter for filtering pathways (default: 1)')  # typical size is from 0 to 10, bigger size results in a cleaner sankey diagram 

    # Parse arguments
    args = parser.parse_args()

    # Set paths for cell data files based on dataset
    args.cell_data_file = f'../data/pickle_data/{args.dataset}_data.pkl'
    args.cell_type_file = f'../data/revised_data/{args.dataset}_celltype.csv'

    # Run the main pathway processing function
    process_pathways(result_path=args.result_path, args=args, dpi=args.dpi)


if __name__ == "__main__":
    main()

