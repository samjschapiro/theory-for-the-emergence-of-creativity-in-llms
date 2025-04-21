# $1: dataset
# $2: weight_decay
# $3: n_layers

EXP_DIR=../creativity
DATA_DIR=/data/locus/project_data/project_data2/chenwu2/creativity_data

python eval_qa.py --dir $EXP_DIR/$1_$2_$3 --dataset $1 --data_dir $DATA_DIR
