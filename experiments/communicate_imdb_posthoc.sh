#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  local cheat_ratio=$3
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/posthoc-communicate-imdb-${activation}-${cheat_ratio}/" \
                          --load "data/saved-models/test-imdb-softmax/" \
                          --save "data/saved-models/posthoc-communicate-imdb-${activation}-${cheat_ratio}/" \
                          --save-explanations "data/debug-explanations/imdb-cosloss-${activation}-${cheat_ratio}.txt" \
                          --print-parameters-per-layer \
                          --final-report \
                          \
                          --corpus imdb \
                          --train-path "data/corpus/imdb/train/" \
                          --dev-path "data/corpus/imdb/dev/" \
                          --test-path "data/corpus/imdb/test/" \
                          --max-length 9999999 \
                          --min-length 0 \
                          \
                          --embeddings-format "text" \
                          --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.imdb" \
                          --embeddings-binary \
                          --freeze-embeddings \
                          --word-embeddings-size 300 \
                          \
                          --explainer "post_hoc" \
                          --message-type "bow" \
                          --explainer-attn-top-k "${topk}" \
                          --explainer-cheat-ratio "${cheat_ratio}" \
                          --explainer-lambda 1.0 \
                          --explainer-second-loss "pred" \
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
                          --train-batch-size 16 \
                          --dev-batch-size 16 \
                          --epochs 10 \
                          --optimizer "adam" \
                          --learning-rate 0.001 \
                          --save-best-only \
                          --early-stopping-patience 3 \
                          --restore-best-model
}

#communicate_spec "softmax" 5 0.0
#communicate_spec "softmax" 5 0.1
communicate_spec "softmax" 5 0.2
#communicate_spec "softmax" 5 0.5
#communicate_spec "softmax" 5 1.0
#
#communicate_spec "sparsemax" 5 0.0
#communicate_spec "sparsemax" 5 0.1
#communicate_spec "sparsemax" 5 0.2
#communicate_spec "sparsemax" 5 0.5
#communicate_spec "sparsemax" 5 1.0
#
#communicate_spec "entmax15" 5 0.0
#communicate_spec "entmax15" 5 0.1
#communicate_spec "entmax15" 5 0.2
#communicate_spec "entmax15" 5 0.5
#communicate_spec "entmax15" 5 1.0
