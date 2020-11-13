#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  local cheat_ratio=$3
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/posthoc-communicate-snli-cosloss-both-with-emb-${activation}-${cheat_ratio}/" \
                          --load "data/saved-models/test-snli-softmax/" \
                          --save "data/saved-models/posthoc-communicate-snli-cosloss-both-with-emb-${activation}-${cheat_ratio}/" \
                          --save-explanations "data/debug-explanations/snli-cosloss-both-${activation}-${cheat_ratio}.txt" \
                          --print-parameters-per-layer \
                          --final-report \
                          \
                          --corpus snli \
                          --train-path "data/corpus/snli/snli_1.0_train.jsonl" \
                          --dev-path "data/corpus/snli/snli_1.0_dev.jsonl" \
                          --test-path "data/corpus/snli/snli_1.0_test.jsonl" \
                          --max-length 9999999 \
                          --min-length 0 \
                          \
                          --embeddings-format "text" \
                          --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.snli" \
                          --embeddings-binary \
                          --freeze-embeddings \
                          --word-embeddings-size 300 \
                          \
                          --explainer "post_hoc_entailment" \
                          --message-type "bow" \
                          --explainer-attn-top-k "${topk}" \
                          --explainer-cheat-ratio "${cheat_ratio}" \
                          --layman "linear_entailment" \
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
                          --train-batch-size 32 \
                          --dev-batch-size 32 \
                          --epochs 10 \
                          --optimizer "adam" \
                          --learning-rate 0.001 \
                          --save-best-only \
                          --early-stopping-patience 3 \
                          --restore-best-model
}

#communicate_spec "softmax" 4 0.0
#communicate_spec "softmax" 4 0.1
#communicate_spec "softmax" 4 0.2
#communicate_spec "softmax" 4 0.5
#communicate_spec "softmax" 4 1.0

communicate_spec "sparsemax" 4 0.0
communicate_spec "sparsemax" 4 0.1
communicate_spec "sparsemax" 4 0.2
communicate_spec "sparsemax" 4 0.5
communicate_spec "sparsemax" 4 1.0

#communicate_spec "entmax15" 4 0.0
#communicate_spec "entmax15" 4 0.1
#communicate_spec "entmax15" 4 0.2
#communicate_spec "entmax15" 4 0.5
#communicate_spec "entmax15" 4 1.0

