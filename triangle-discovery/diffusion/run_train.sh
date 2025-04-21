python train.py \
    noise.type=loglinear \
    graph.type=absorb \
    model=small \
    training.accum=1 \
    data.train=creativity_data/triangle.0/train.json \
    data.valid=creativity_data/triangle.0/valid.json \
    add_vocab=creativity_data/triangle.0/vocab.json \
    hydra.run.dir=/data/locus/project_data/project_data2/chenwu2/creativity_results/creativity_data/triangle.0/train.json/train

python test.py \
    --model_checkpoint_dir creativity_results/creativity_data/triangle.0/train.json/train \
    --dataset creativity_data/triangle.0 \
    --add_vocab creativity_data/triangle.0/vocab.json