#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  local cheat_ratio=$3
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/posthoc-communicate-yelp-${activation}-${cheat_ratio}/" \
                          --load "data/saved-models/test-yelp-softmax/" \
                          --save "data/saved-models/posthoc-communicate-yelp-${activation}-${cheat_ratio}/" \
                          --save-explanations "data/debug-explanations/yelp-cosloss-${activation}-${cheat_ratio}.txt" \
                          --print-parameters-per-layer \
                          --final-report \
                          \
                          --corpus yelp \
                          --train-path "data/corpus/yelp/review_train.json" \
                          --dev-path "data/corpus/yelp/review_dev.json" \
                          --test-path "data/corpus/yelp/review_test.json" \
                          --max-length 9999999 \
                          --min-length 0 \
                          \
                          --embeddings-format "text" \
                          --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.yelp" \
                          --embeddings-binary \
                          --freeze-embeddings \
                          --word-embeddings-size 300 \
                          --lazy-loading \
                          \
                          --explainer "post_hoc" \
                          --message-type "bow" \
                          --explainer-attn-top-k "${topk}" \
                          --explainer-cheat-ratio "${cheat_ratio}" \
                          --layman "linear" \
                          --freeze-classifier-params \
                          --train-explainer \
                          \
                          --rnn-type "lstm" \
                          --hidden-size 128 \
                          --bidirectional \
                          --attn-dropout 0.0 \
                          --attn-max-activation "${activation}" \
                          \
                          --loss-weights "same" \
                          --train-batch-size 64 \
                          --dev-batch-size 64 \
                          --epochs 3 \
                          --optimizer "adam" \
                          --learning-rate 0.003 \
                          --save-best-only \
                          --early-stopping-patience 2 \
                          --restore-best-model
}

#communicate_spec "softmax" 5 0.0
#communicate_spec "softmax" 5 0.1
communicate_spec "softmax" 5 0.2
#communicate_spec "softmax" 5 0.5
#communicate_spec "softmax" 5 1.0

#communicate_spec "sparsemax" 5 0.0
#communicate_spec "sparsemax" 5 0.1
communicate_spec "sparsemax" 5 0.2
#communicate_spec "sparsemax" 5 0.5
#communicate_spec "sparsemax" 5 1.0

#communicate_spec "entmax15" 5 0.0
#communicate_spec "entmax15" 5 0.1
communicate_spec "entmax15" 5 0.2
#communicate_spec "entmax15" 5 0.5
#communicate_spec "entmax15" 5 1.0

