#!/usr/bin/env bash

train_spec(){
  local activation=$1
  python3 -m spec train \
      --seed 42 \
      --gpu-id 0  \
      --output-dir "runs/test-imdb-${activation}/" \
      --save "data/saved-models/test-imdb-${activation}/" \
      --print-parameters-per-layer \
      --final-report \
      \
      --corpus imdb \
      --train-path "data/corpus/imdb/train/" \
      --dev-path "data/corpus/imdb/dev/" \
      --test-path "data/corpus/imdb/test/" \
      --max-length 9999999 \
      --min-length 0 \
      \
      --vocab-size 9999999 \
      --vocab-min-frequency 1 \
      --keep-rare-with-vectors \
      --add-embeddings-vocab \
      \
      --embeddings-format "text" \
      --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.imdb" \
      --embeddings-binary \
      --emb-dropout 0.0 \
      --freeze-embeddings \
      \
      --model rnn_attn \
      \
      --rnn-type lstm \
      --hidden-size 128 \
      --bidirectional \
      --rnn-dropout 0.0 \
      \
      --attn-type "regular" \
      --attn-scorer "self_add" \
      --attn-dropout 0.0 \
      --attn-max-activation "${activation}" \
      \
      --loss-weights "same" \
      --train-batch-size 16 \
      --dev-batch-size 16 \
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
      --load "data/saved-models/test-imdb-${activation}/" \
      --corpus imdb \
      --test-path "data/corpus/imdb/test/" \
      --output-dir "data/predictions/test-imdb-${activation}/" \
      --dev-batch-size 4
}


train_spec "softmax"
predict_spec "softmax"

train_spec "sparsemax"
predict_spec "sparsemax"

train_spec "entmax15"
predict_spec "entmax15"
