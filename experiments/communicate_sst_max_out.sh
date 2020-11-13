#!/usr/bin/env bash

# if you want to load a trained communication (which usually means a trained
# layman, since only posthoc explainers have parameters to be trained, the rest
# only pools attention weights from the classifier)
# --load-communication "data/saved-models/debug-communicate-sst-${activation}/" \

communicate_spec(){
  local activation=$1
  local topk=$2
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/max-out-communicate-sst-${activation}/" \
                          --load "data/saved-models/test-sst-${activation}/" \
                          --save "data/saved-models/max-out-communicate-sst-${activation}/" \
                          --save-explanations "data/explanations/max_out_sst_test.txt" \
                          --max-explanations 2000 \
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
                          --explainer recursive_max_out \
                          --message-type bow \
                          --explainer-attn-top-k "${topk}" \
                          --layman linear \
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

#communicate_spec "softmax" 1
#communicate_spec "softmax" 3
communicate_spec "softmax" 9999

#communicate_spec "sparsemax" 1
#communicate_spec "sparsemax" 3
communicate_spec "sparsemax" 5

#communicate_spec "entmax15" 1
#communicate_spec "entmax15" 3
communicate_spec "entmax15" 5
