# scKGOT: Intercellular Signaling Inference with Knowledge Graph Optimal Transport for Single-cell Transcriptomics

## Overview

scKGOT is a framework designed to help infer intercellular signaling from single-cell RNA sequencing (scRNA-seq) data. By leveraging a Ligand-Receptor-Pathway Knowledge Graph (LRP-KG) and Optimal Transport (OT) theory, scKGOT models the ligand-receptor-signaling pathways between sender and receiver cells. The goal is to provide interpretable predictions of intercellular communication.

scKGOT seeks to integrate gene expression data with pathway knowledge in an effort to better predict the pathways that mediate communication between different cell types. It aims to help reconstruct signaling pathways and quantify the significance of ligand-receptor interactions, facilitating a deeper understanding of cellular mechanisms involved in tissue homeostasis, development, and disease.


## Key Features
- Utilizes LRP-KG, a knowledge graph specifically built for ligand-receptor-pathway interactions, integrating biological priors to enhance model precision.
- Employs optimal transport to model the signal transduction process through pathways, connecting ligands and receptors.
- Benchmarks against multiple state-of-the-art methods, outperforming them in accuracy and efficiency.
- Provides comprehensive visualizations including heatmaps and Sankey diagrams for intuitive exploration of signaling pathways.

## Installation

To install scKGOT and its dependencies, follow these steps:

### Prerequisites
- Python >= 3.9
- POT (Python Optimal Transport) == 0.9.3
- torch == 2.0.1
- Install dependencies via `requirements.txt`.

### Setup
1. Clone the repository:  
   `git clone https://github.com/yourusername/scKGOT.git`  
   `cd scKGOT`

2. Install dependencies:  
   `pip install -r requirements.txt`

3. You are now ready to start using scKGOT!

## Example Scripts

The `./scripts` folder contains example scripts that demonstrate how to run the scKGOT model on specific datasets. For example:

`bash ./scripts/test_maintable_human.sh`

This script processes multiple datasets with pre-defined sender and receiver cell types, adjusting various model parameters such as the number of iterations, number of cores, and thresholds.


## Project Structure

Below is an overview of the main files and directories:

```
scKGOT/
│
├── data/
│ ├── lr_pairs/     # Directory for ligand-receptor pair data
│ ├── revised_data/ # Directory for scRNA-seq datasets
│ ├── pickle_data/  # Directory for pickled data files
│ ├── human_pathways.csv # Human pathways data
│ └── mouse_pathways.csv # Mouse pathways data
│
├── kgot/
│ ├── main.py          # Main script to run the scKGOT pipeline
│ ├── dataholder.py    # Handles data loading and processing
│ ├── knowledge.p      # Functionality of LRP-KG
│ ├── scheduler.py     # Scheduler for subtasks
│ ├── core.py          # Core implementation of the scKGOT algorithm
│ ├── ot_solver.py     # Implements the optimal transport solver for LR pairs
│ ├── logger_utils.py    # Utility for logging model progress and results
│ ├── utils.py           # General utility functions
│ └── post_processing.py # Post-processes model results, including pathway analysis
│
├── results/ # Directory for storing output results
├── vis_output/ # Output directory for visualization results
└── requirements.txt # List of dependencies for the project
```

### Key Modules
- **core.py**: Implements the core scKGOT architecture, integrating LRP-KG and optimal transport theory.
- **ot_solver.py**: Solves the optimal transport problem to model the transfer of biological signals between ligands and receptors.
- **main.py**: The entry point for running the scKGOT pipeline, with configurable settings in YAML format.
- **knowledge.py**: Builds and processes the Ligand-Receptor-Pathway Knowledge Graph (LRP-KG) used for pathway inference.

## Data Requirements

scKGOT requires two primary inputs:  
1. **Single-cell RNA sequencing (scRNA-seq) data**: Gene expression across different cell types.  
2. **Ligand-Receptor-Pathway Knowledge Graph (LRP-KG)**: Pre-built knowledge graph with curated information on ligand-receptor pairs and associated signaling pathways.

Ensure your scRNA-seq data and the LRP-KG are properly formatted.

## Core Methodology

### Ligand-Receptor-Pathway Knowledge Graph (LRP-KG)
scKGOT employs a Ligand-Receptor-Pathway Knowledge Graph (LRP-KG) to model intercellular communication. The LRP-KG encodes prior biological knowledge about gene interactions, ligand-receptor pairs, and signaling pathways, integrating data from well-known databases like KEGG and Reactome.

### Gene Importance Score and Pathway Knowledge Discrepancy
At the core of scKGOT is a mechanism that evaluates both gene importance and pathway knowledge discrepancy:

Gene Importance Score: This score quantifies the role of specific genes in the context of ligand-receptor communication within pathways. By calculating the probability of functional interactions between genes, scKGOT identifies the key players involved in mediating intercellular signaling. These importance scores are integrated into the optimal transport framework to prioritize pathways and gene pairs that are most likely involved in communication between sender and receiver cells.

Pathway Knowledge Discrepancy (KD): To ensure that predicted signaling pathways align with known biological data, scKGOT incorporates pathway knowledge discrepancy. KD measures the difference between the predicted pathway activity and the established structure of the LRP-KG. This allows the model to adjust predictions based on the degree of consistency with prior biological knowledge, ensuring that the pathways identified as active are not only statistically significant but also biologically plausible.

### Optimal Transport
The model frames the inference of ligand-receptor communication as a transportation problem. scKGOT solves the Gromov-Wasserstein (GW) optimal transport distance to model the flow of biological signals through various pathways. This formulation allows scKGOT to map ligands to receptors by transporting gene importance scores through the most relevant pathways, based on both gene expression data and the LRP-KG structure. The combination of gene importance and KD ensures that the pathways identified are robust, interpretable, and consistent with known biology.

This methodology enables scKGOT to identify not only ligand-receptor pairs but also the signaling pathways they activate, offering deeper insights into the mechanisms of intercellular communication compared to conventional methods.

## License

This project is licensed under the MIT License.

## Contact

For any questions, feedback, or suggestions, feel free to reach out:

- **Name**: Haihong Yang 
- **Email**: haihong825@zju.edu.cn
