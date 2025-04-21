# $1: dataset
# $2: weight_decay
# $3: n_layers

EXP_DIR=creativity_results/creativity_data/sibling.5.500.10.50000/train.json/train/checkpoint_outputs

DATA_DIR=/data/locus/project_data/project_data2/chenwu2/creativity_data

python eval_qa.py --dir $EXP_DIR --dataset $1 --data_dir $DATA_DIR
