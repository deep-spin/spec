#!/usr/bin/env bash

communicate_spec(){
  local activation=$1
  local topk=$2
  python3 -m spec communicate_translation \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/communicate-embedded-iwslt-${activation}-${topk}/" \
                          --save "data/saved-models/communicate-embedded-iwslt-${activation}-${topk}/" \
                          --save-explanations "data/explanations/iwslt_embedded_${activation}-${topk}.txt" \
                          --max-explanations 2000 \
                          --print-parameters-per-layer \
                          --final-report \
                          \
                          --corpus iwslt \
                          --train-path "data/saved-translation-models/iwslt-ende-bahdanau-${activation}-new/attn-train/train" \
                          --dev-path "data/saved-translation-models/iwslt-ende-bahdanau-${activation}-new/attn-dev/dev" \
                          --test-path "data/saved-translation-models/iwslt-ende-bahdanau-${activation}-new/attn-test/test" \
                          --max-length 100 \
                          --min-length 0 \
                          \
                          --vocab-size 9999999 \
                          --vocab-min-frequency 1 \
                          \
                          --embeddings-format "text" \
                          --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.iwslt.en" \
                          --embeddings-binary \
                          --embeddings-format-target "fasttext" \
                          --embeddings-path-target "data/embs/fasttext/cc.de.300.bin.gz" \
                          --embeddings-binary-target \
                          --emb-dropout 0.0 \
                          \
                          --word-embeddings-size 300 \
                          --hidden-size 256 \
                          \
                          --explainer encoded_attn_translation \
                          --message-type "embs" \
                          --explainer-attn-top-k "${topk}" \
                          --layman linear_translation \
                          --freeze-classifier-params \
                          --freeze-explainer-params \
                          \
                          --loss-weights "same" \
                          --train-batch-size 16 \
                          --dev-batch-size 16 \
                          --epochs 10 \
                          --optimizer "adam" \
                          --learning-rate 0.003 \
                          --save-best-only \
                          --early-stopping-patience 5 \
                          --restore-best-model
}

#communicate_spec "sparsemax" 22  # max attn nonzeros
#communicate_spec "entmax15" 44  # max attn nonzeros

communicate_spec "softmax" 154  # max len
communicate_spec "sparsemax" 154  # max len
communicate_spec "entmax15" 154  # max len
