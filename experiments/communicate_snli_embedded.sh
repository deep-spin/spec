#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 2  \
                          --output-dir "runs/communicate-embedded-snli-with-emb-${activation}/" \
                          --load "data/saved-models/test-snli-${activation}/" \
                          --save "data/saved-models/communicate-embedded-snli-with-emb-${activation}/" \
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
                          --explainer encoded_attn \
                          --message-type bow \
                          --explainer-attn-top-k "${topk}" \
                          --layman linear_entailment \
                          --freeze-classifier-params \
                          --freeze-explainer-params \
                          \
                          --bidirectional \
                          --rnn-type "lstm" \
                          --hidden-size 128 \
                          \
                          --loss-weights "same" \
                          --train-batch-size 64 \
                          --dev-batch-size 64 \
                          --epochs 10 \
                          --optimizer "adam" \
                          --learning-rate 0.001 \
                          --save-best-only \
                          --early-stopping-patience 3 \
                          --restore-best-model
}

communicate_spec "sparsemax" 99999
communicate_spec "entmax15" 99999
