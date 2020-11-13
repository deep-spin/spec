#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/communicate-embedded-agnews-${activation}/" \
                          --load "data/saved-models/test-agnews-${activation}/" \
                          --save "data/saved-models/communicate-embedded-agnews-${activation}/" \
                          --print-parameters-per-layer \
                          --final-report \
                          \
                          --corpus agnews \
                          --train-path "data/corpus/agnews/train.xml" \
                          --dev-path "data/corpus/agnews/dev.xml" \
                          --test-path "data/corpus/agnews/test.xml" \
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
