#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  local random_type=$3
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 1  \
                          --output-dir "runs/random-communicate-yelp-${activation}/" \
                          --load "data/saved-models/test-yelp-${activation}/" \
                          --save "data/saved-models/random-communicate-yelp-${activation}/" \
                          --print-parameters-per-layer \
                          --final-report \
                          \
                          --corpus yelp \
                          --train-path "data/corpus/yelp/review_train.json" \
                          --dev-path "data/corpus/yelp/review_dev.json" \
                          --test-path "data/corpus/yelp/review_test.json" \
                          --max-length 9999999 \
                          --min-length 0 \
                          --word-embeddings-size 300 \
                          --lazy-loading \
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
                          --train-batch-size 98 \
                          --dev-batch-size 98 \
                          --epochs 3 \
                          --optimizer "adam" \
                          --learning-rate 0.003 \
                          --save-best-only \
                          --early-stopping-patience 2 \
                          --restore-best-model
}

communicate_spec "softmax" 10 "uniform"
#communicate_spec "softmax" 10 "beta"
communicate_spec "softmax" 10 "shuffle"
communicate_spec "softmax" 10 "zero_max_out"
#communicate_spec "softmax" 10 "first_states"
#communicate_spec "softmax" 10 "last_states"
#communicate_spec "softmax" 10 "mid_states"

communicate_spec "sparsemax" 10 "uniform"
#communicate_spec "sparsemax" 10 "beta"
communicate_spec "sparsemax" 10 "shuffle"
communicate_spec "sparsemax" 10 "zero_max_out"
#communicate_spec "sparsemax" 10 "first_states"
#communicate_spec "sparsemax" 10 "last_states"
#communicate_spec "sparsemax" 10 "mid_states"

communicate_spec "entmax15" 10 "uniform"
#communicate_spec "entmax15" 10 "beta"
communicate_spec "entmax15" 10 "shuffle"
communicate_spec "entmax15" 10 "zero_max_out"
#communicate_spec "entmax15" 10 "first_states"
#communicate_spec "entmax15" 10 "last_states"
#communicate_spec "entmax15" 10 "mid_states"

