#!/usr/bin/env bash

train_spec(){
  local activation=$1
  python3 -m spec train \
      --seed 42 \
      --gpu-id 0  \
      --output-dir "runs/emb-test-sst-${activation}/" \
      --save "data/saved-models/emb-test-sst-${activation}/" \
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
      --model rnn_attn_emb \
      \
      --rnn-type lstm \
      --hidden-size 128 \
      --bidirectional \
      --rnn-dropout 0.0 \
      \
      --attn-type "regular" \
      --attn-scorer "add" \
      --attn-dropout 0.0 \
      --attn-max-activation "${activation}" \
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
}

predict_spec(){
  local activation=$1
  python3 -m spec predict \
      --gpu-id 0  \
      --prediction-type classes \
      --load "data/saved-models/emb-test-sst-${activation}/" \
      --corpus sst \
      --test-path "data/corpus/sst/test.txt" \
      --output-dir "data/predictions/emb-test-sst-${activation}/" \
      --dev-batch-size 4
}


train_spec "softmax"
predict_spec "softmax"

train_spec "sparsemax"
predict_spec "sparsemax"

train_spec "entmax15"
predict_spec "entmax15"
