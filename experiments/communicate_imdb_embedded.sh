#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/communicate-embedded-imdb-${activation}/" \
                          --load "data/saved-models/test-imdb-${activation}/" \
                          --save "data/saved-models/communicate-embedded-imdb-${activation}/" \
                          --print-parameters-per-layer \
                          --final-report \
                          \
                          --corpus imdb \
                          --train-path "data/corpus/imdb/train/" \
                          --dev-path "data/corpus/imdb/dev/" \
                          --test-path "data/corpus/imdb/test/" \
                          --max-length 9999999 \
                          --min-length 0 \
                          --word-embeddings-size 300 \
                          \
                          --explainer encoded_attn \
                          --message-type bow \
                          --explainer-attn-top-k "${topk}" \
                          --layman linear \
                          --freeze-classifier-params \
                          --freeze-explainer-params \
                          \
                          --loss-weights "same" \
                          --train-batch-size 16 \
                          --dev-batch-size 16 \
                          --epochs 10 \
                          --optimizer "adam" \
                          --learning-rate 0.001 \
                          --save-best-only \
                          --early-stopping-patience 3 \
                          --restore-best-model
}

communicate_spec "sparsemax" 99999
communicate_spec "entmax15" 99999
