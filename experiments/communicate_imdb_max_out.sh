#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 3  \
                          --output-dir "runs/max-out-communicate-imdb-${activation}/" \
                          --load "data/saved-models/test-imdb-${activation}/" \
                          --save "data/saved-models/max-out-communicate-imdb-${activation}/" \
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
                          --explainer recursive_max_out \
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

#communicate_spec "softmax" 1
#communicate_spec "softmax" 3
communicate_spec "softmax" 5
#communicate_spec "softmax" 1000

#communicate_spec "sparsemax" 1
#communicate_spec "sparsemax" 3
communicate_spec "sparsemax" 5
#communicate_spec "sparsemax" 10
#
#communicate_spec "entmax15" 1
#communicate_spec "entmax15" 3
communicate_spec "entmax15" 5
#communicate_spec "entmax15" 10
