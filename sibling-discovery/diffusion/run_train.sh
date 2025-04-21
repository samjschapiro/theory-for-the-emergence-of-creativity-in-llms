python train.py \
    noise.type=loglinear \
    graph.type=absorb \
    model=small \
    training.accum=1 \
    data.train=creativity_data/sibling.5.500.10.50000/train.json \
    data.valid=creativity_data/sibling.5.500.10.50000/valid.json \
    add_vocab=creativity_data/sibling.5.500.10.50000/vocab.json \
    hydra.run.dir=/data/locus/project_data/project_data2/chenwu2/creativity_results/creativity_data/sibling.5.500.10.50000/train.json/train

python test.py \
    --model_checkpoint_dir creativity_results/creativity_data/sibling.5.500.10.50000/train.json/train \
    --dataset creativity_data/sibling.5.500.10.50000 \
    --add_vocab creativity_data/sibling.5.500.10.50000/vocab.json