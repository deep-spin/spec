#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  local random_type=$3
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/random-communicate-sst-${activation}/" \
                          --load "data/saved-models/test-sst-${activation}/" \
                          --save "data/saved-models/random-communicate-sst-${activation}/" \
                          --print-parameters-per-layer \
                          --final-report \
                          \
                          --corpus sst \
                          --train-path "data/corpus/sst/train.txt" \
                          --dev-path "data/corpus/sst/dev.txt" \
                          --test-path "data/corpus/sst/test.txt" \
                          --max-length 9999999 \
                          --min-length 0 \
                          --word-embeddings-size 300 \
                          \
                          --explainer "random_attn" \
                          --message-type "bow" \
                          --explainer-attn-top-k "${topk}" \
                          --explainer-random-type "${random_type}" \
                          --layman "linear" \
                          --freeze-classifier-params \
                          --freeze-explainer-params \
                          \
                          --loss-weights "same" \
                          --train-batch-size 8 \
                          --dev-batch-size 8 \
                          --epochs 10 \
                          --optimizer "adam" \
                          --learning-rate 0.001 \
                          --save-best-only \
                          --early-stopping-patience 3 \
                          --restore-best-model
}


communicate_spec "softmax" 5 "uniform"
communicate_spec "softmax" 5 "beta"
communicate_spec "softmax" 5 "shuffle"
communicate_spec "softmax" 5 "zero_max_out"
communicate_spec "softmax" 5 "first_states"
communicate_spec "softmax" 5 "last_states"
#communicate_spec "softmax" 5 "mid_states"

communicate_spec "sparsemax" 5 "uniform"
communicate_spec "sparsemax" 5 "beta"
communicate_spec "sparsemax" 5 "shuffle"
communicate_spec "sparsemax" 5 "zero_max_out"
communicate_spec "sparsemax" 5 "first_states"
communicate_spec "sparsemax" 5 "last_states"
#communicate_spec "sparsemax" 5 "mid_states"

communicate_spec "entmax15" 5 "uniform"
communicate_spec "entmax15" 5 "beta"
communicate_spec "entmax15" 5 "shuffle"
communicate_spec "entmax15" 5 "zero_max_out"
communicate_spec "entmax15" 5 "first_states"
communicate_spec "entmax15" 5 "last_states"
#communicate_spec "entmax15" 5 "mid_states"
