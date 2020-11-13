#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/communicate-embedded-yelp-${activation}/" \
                          --load "data/saved-models/test-yelp-${activation}/" \
                          --save "data/saved-models/communicate-embedded-yelp-${activation}/" \
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
                          --explainer embedded_attn \
                          --message-type bow \
                          --explainer-attn-top-k "${topk}" \
                          --layman linear \
                          --freeze-classifier-params \
                          --freeze-explainer-params \
                          \
                          --loss-weights "same" \
                          --train-batch-size 112 \
                          --dev-batch-size 112 \
                          --epochs 3 \
                          --optimizer "adam" \
                          --learning-rate 0.003 \
                          --save-best-only \
                          --early-stopping-patience 2 \
                          --restore-best-model
}


communicate_spec "sparsemax" 99999
communicate_spec "entmax15" 99999
