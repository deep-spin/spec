#!/usr/bin/env bash


python3 -m spec train \
    --seed 42 \
    --gpu-id 0  \
    --output-dir "runs/bow-sst/" \
    --save "data/saved-models/bow-sst/" \
    --print-parameters-per-layer \
    --final-report \
    \
    --corpus sst \
    --train-path "data/corpus/sst/train.txt" \
    --dev-path "data/corpus/sst/dev.txt" \
    --test-path "data/corpus/sst/test.txt" \
    --max-length 9999999 \
    --min-length 0 \
    \
    --vocab-size 9999999 \
    --vocab-min-frequency 1 \
    --keep-rare-with-vectors \
    --add-embeddings-vocab \
    \
    --embeddings-format "text" \
    --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.sst" \
    --embeddings-binary \
    --emb-dropout 0.0 \
    --freeze-embeddings \
    \
    --model linear_bow \
    \
    --rnn-type lstm \
    --hidden-size 128 \
    --bidirectional \
    --rnn-dropout 0.0 \
    \
    --loss-weights "same" \
    --train-batch-size 8 \
    --dev-batch-size 8 \
    --epochs 10 \
    --optimizer "adamw" \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --save-best-only \
    --early-stopping-patience 5 \
    --restore-best-model



python3 -m spec predict \
    --gpu-id 0  \
    --prediction-type classes \
    --load "data/saved-models/bow-sst/" \
    --corpus sst \
    --test-path "data/corpus/sst/test.txt" \
    --output-dir "data/predictions/bow-sst/" \
    --dev-batch-size 4


