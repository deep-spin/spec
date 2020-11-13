#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  python3 -m spec communicate \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/communicate-esnli-expl-with-emb-${activation}/" \
                          --load "data/saved-models/test-snli-${activation}/" \
                          --save "data/saved-models/communicate-esnli-expl-with-emb-${activation}/" \
                          --print-parameters-per-layer \
                          --final-report \
                          \
                          --corpus esnli \
                          --train-path "data/corpus/esnli/esnli_train.csv" \
                          --dev-path "data/corpus/esnli/esnli_dev.csv" \
                          --test-path "data/corpus/esnli/esnli_test.csv" \
                          --max-length 9999999 \
                          --min-length 0 \
                          \
                          --model rnn_attn_entailment \
                          \
                          --embeddings-format "text" \
                          --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.snli" \
                          --embeddings-binary \
                          --freeze-embeddings \
                          --word-embeddings-size 300 \
                          \
                          --use-gold-as-clf \
                          --explainer embedded_attn_supervised \
                          --message-type bow \
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

communicate_spec "softmax"
#communicate_spec "sparsemax"
#communicate_spec "entmax15"
