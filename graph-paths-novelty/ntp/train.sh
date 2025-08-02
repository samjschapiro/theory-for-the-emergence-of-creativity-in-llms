MODEL_PATH=gpt2

DATASET=/data/locus/project_data/project_data2/chenwu2/creativity_data/$1/
WEIGHT_DECAY=$2
N_LAYERS=$3
GPU=$4

EXP_DIR=../creativity

OUTPUT_DIR=$EXP_DIR/$1_$2_$3

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --data_dir $DATASET \
    --model_name_or_path ${MODEL_PATH} \
    --weight_decay $WEIGHT_DECAY \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 128 \
    --max_length 128 \
    --block_size 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --save_step 40000 \
    --save_step_dense 20000 \
    --max_steps 800000 \
    --do_train \
    --scheduler constant_schedule_with_warmup \
    --fp16 \
    --evaluate_during_training \
    --predict_during_training \
    --init_weights \
    --add_tokens \
    --n_layer $N_LAYERS
