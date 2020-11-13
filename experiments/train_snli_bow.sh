#!/usr/bin/env bash


python3 -m spec train \
    --seed 42 \
    --gpu-id 0  \
    --output-dir "runs/bow-snli/" \
    --save "data/saved-models/bow-snli/" \
    --print-parameters-per-layer \
    --final-report \
    \
    --corpus snli \
    --train-path "data/corpus/snli/snli_1.0_train.jsonl" \
    --dev-path "data/corpus/snli/snli_1.0_dev.jsonl" \
    --test-path "data/corpus/snli/snli_1.0_test.jsonl" \
    --max-length 9999999 \
    --min-length 0 \
    \
    --vocab-size 9999999 \
    --vocab-min-frequency 1 \
    --keep-rare-with-vectors \
    --add-embeddings-vocab \
    \
    --embeddings-format "text" \
    --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.snli" \
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
    --train-batch-size 64 \
    --dev-batch-size 64 \
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
    --load "data/saved-models/bow-snli/" \
    --corpus snli \
    --test-path "data/corpus/snli/snli_1.0_test.jsonl" \
    --output-dir "data/predictions/bow-snli/" \
    --dev-batch-size 4
