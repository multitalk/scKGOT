#!/bin/bash

# Set constants
dataset="human_placenta_20218"
n_times=100
n_cores=8
num_iter_max=1000
tolerance=0.0001
cell_samples_threshold=1

# Directories
cell_data_dir="../data/pickle_data"
cell_type_dir="../data/revised_data"
dump_path="../results/example_experiment"

# Cell types
declare -a sender=("lymphatic endothelial cell")
declare -a receiver=("villous cytotrophoblast")

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for index in ${!sender[@]}; do
    python "${SCRIPT_DIR}/../kgot/main.py" \
        --cell_data_file "${SCRIPT_DIR}/${cell_data_dir}/${dataset}_data.pkl" \
        --cell_type_file "${SCRIPT_DIR}/${cell_type_dir}/${dataset}_celltype.csv" \
        --cell_type_1 "${sender[$index]}" \
        --cell_type_2 "${receiver[$index]}" \
        --ncores $n_cores \
        --times $n_times \
        --num_iter_max $num_iter_max \
        --tolerance $tolerance \
        --no_self_loop \
        --use_directed_pathways \
        --cell_samples_threshold $cell_samples_threshold \
        --expr_drop_ratio 0.0 \
        --edge_drop_ratio 0.0 \
        --type_drop_ratio 0.0 \
        --cell_drop_ratio 0.0 \
        --mode "both" \
        --dump_path "${SCRIPT_DIR}/$dump_path" \
        --exp_name "${dataset}_example"
done


