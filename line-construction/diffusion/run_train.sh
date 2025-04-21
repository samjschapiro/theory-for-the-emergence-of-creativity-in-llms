python train.py \
    noise.type=loglinear \
    graph.type=absorb \
    model=small \
    training.accum=1 \
    data.train=creativity_data/line.10.9.0.10000/train.json \
    data.valid=creativity_data/line.10.9.0.10000/valid.json \
    add_vocab=creativity_data/line.10.9.0.10000/vocab.json \
    hydra.run.dir=/data/locus/project_data/project_data2/chenwu2/creativity_results/creativity_data/line.10.9.0.10000/train.json/train

python test.py \
    --model_checkpoint_dir creativity_results/creativity_data/line.10.9.0.10000/train.json/train \
    --dataset creativity_data/line.10.9.0.10000 \
    --add_vocab creativity_data/line.10.9.0.10000/vocab.json