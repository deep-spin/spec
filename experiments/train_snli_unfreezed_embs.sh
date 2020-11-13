#!/usr/bin/env bash

train_spec(){
  local activation=$1
  python3 -m spec train \
      --seed 42 \
      --gpu-id 3  \
      --output-dir "runs/test-snli-unfreezed-embs-${activation}/" \
      --save "data/saved-models/test-snli-unfreezed-embs-${activation}/" \
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
      \
      --model rnn_attn_entailment \
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
      --train-batch-size 64 \
      --dev-batch-size 64 \
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
      --load "data/saved-models/test-snli-unfreezed-embs-${activation}/" \
      --corpus snli \
      --test-path "data/corpus/snli/snli_1.0_test.jsonl" \
      --output-dir "data/predictions/test-snli-unfreezed-embs-${activation}/" \
      --dev-batch-size 4
}


train_spec "softmax"
predict_spec "softmax"

train_spec "sparsemax"
predict_spec "sparsemax"

train_spec "entmax15"
predict_spec "entmax15"
