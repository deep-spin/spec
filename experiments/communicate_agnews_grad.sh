#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/grad-new-communicate-agnews-${activation}/" \
                          --load "data/saved-models/test-agnews-${activation}/" \
                          --save "data/saved-models/grad-new-communicate-agnews-${activation}/" \
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
                          --explainer "gradient_magnitude" \
                          --message-type "bow" \
                          --explainer-attn-top-k "${topk}" \
                          --layman "linear" \
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

communicate_spec "softmax" 1
communicate_spec "softmax" 3
communicate_spec "softmax" 5
communicate_spec "softmax" 10

communicate_spec "sparsemax" 1
communicate_spec "sparsemax" 3
communicate_spec "sparsemax" 5
communicate_spec "sparsemax" 10

communicate_spec "entmax15" 1
communicate_spec "entmax15" 3
communicate_spec "entmax15" 5
communicate_spec "entmax15" 10
