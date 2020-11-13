#!/usr/bin/env bash

python3 -m spec train \
    --seed 42 \
    --gpu-id 0  \
    --output-dir "runs/bow-agnews/" \
    --save "data/saved-models/bow-agnews/" \
    --print-parameters-per-layer \
    --final-report \
    \
    --corpus agnews \
    --train-path "data/corpus/agnews/train.xml" \
    --dev-path "data/corpus/agnews/dev.xml" \
    --test-path "data/corpus/agnews/test.xml" \
    --max-length 9999999 \
    --min-length 0 \
    \
    --vocab-size 9999999 \
    --vocab-min-frequency 1 \
    --keep-rare-with-vectors \
    --add-embeddings-vocab \
    \
    --embeddings-format "text" \
    --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.agnews" \
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
    --train-batch-size 16 \
    --dev-batch-size 16 \
    --epochs 5 \
    --optimizer "adamw" \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --save-best-only \
    --early-stopping-patience 3 \
    --restore-best-model


python3 -m spec predict \
    --gpu-id 0  \
    --prediction-type classes \
    --load "data/saved-models/bow-agnews/" \
    --corpus agnews \
    --test-path "data/corpus/agnews/test.xml" \
    --output-dir "data/predictions/bow-agnews/" \
    --dev-batch-size 4
