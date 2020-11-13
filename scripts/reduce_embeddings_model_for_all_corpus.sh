#!/usr/bin/env bash

python3 reduce_embeddings_model.py --corpus agnews \
                                   --data-paths ../data/corpus/agnews/newsspace200.xml \
                                   --emb-path ../data/embs/glove/glove.840B.300d.txt \
                                   --output-path ../data/embs/glove/glove.840B.300d.small.raw.pickle.agnews \
                                   --format glove  \
                                   --binary


python3 reduce_embeddings_model.py --corpus imdb \
                                   --data-paths ../data/corpus/imdb/train/ \
                                                ../data/corpus/imdb/dev/ \
                                                ../data/corpus/imdb/test/ \
                                   --emb-path ../data/embs/glove/glove.840B.300d.txt \
                                   --output-path ../data/embs/glove/glove.840B.300d.small.raw.pickle.imdb \
                                   --format glove  \
                                   --binary


python3 reduce_embeddings_model.py --corpus mnli \
                                   --data-paths ../data/corpus/mnli/multinli_1.0_train.jsonl \
                                                ../data/corpus/mnli/multinli_1.0_dev_matched.jsonl \
                                                ../data/corpus/mnli/multinli_0.9_test_matched_unlabeled.jsonl \
                                   --emb-path ../data/embs/glove/glove.840B.300d.txt \
                                   --output-path ../data/embs/glove/glove.840B.300d.small.raw.pickle.mnli \
                                   --format glove  \
                                   --binary


python3 reduce_embeddings_model.py --corpus snli \
                                   --data-paths ../data/corpus/snli/snli_1.0_train.jsonl \
                                                ../data/corpus/snli/snli_1.0_dev.jsonl \
                                                ../data/corpus/snli/snli_1.0_test.jsonl \
                                   --emb-path ../data/embs/glove/glove.840B.300d.txt \
                                   --output-path ../data/embs/glove/glove.840B.300d.small.raw.pickle.snli \
                                   --format glove  \
                                   --binary

python3 reduce_embeddings_model.py --corpus sst \
                                   --data-paths ../data/corpus/sst/train.txt \
                                                ../data/corpus/sst/test.txt \
                                                ../data/corpus/sst/dev.txt \
                                   --emb-path ../data/embs/glove/glove.840B.300d.txt \
                                   --output-path ../data/embs/glove/glove.840B.300d.small.raw.pickle.sst \
                                   --format glove  \
                                   --binary

# instead: set --emb-path to a portuguese glove model (google for nilc embeddings)
#python3 reduce_embeddings_model.py --corpus ttsbr \
#                                   --data-paths ../data/corpus/ttsbr/trainTT.txt \
#                                                ../data/corpus/ttsbr/testTT.txt \
#                                   --emb-path ../data/embs/glove/glove.840B.300d.txt \
#                                   --output-path ../data/embs/glove/glove.840B.300d.small.raw.pickle.ttsbr \
#                                   --format glove  \
#                                   --binary

python3 reduce_embeddings_model.py --corpus yelp \
                                   --data-paths ../data/corpus/yelp/review_train.json \
                                                ../data/corpus/yelp/review_dev.json \
                                                ../data/corpus/yelp/review_test.json \
                                   --emb-path ../data/embs/glove/glove.840B.300d.txt \
                                   --output-path ../data/embs/glove/glove.840B.300d.small.raw.pickle.yelp \
                                   --format glove  \
                                   --binary

python3 reduce_embeddings_model.py --corpus iwslt \
                                   --data-paths ../data/saved-translation-models/iwslt-ende-bahdanau-softmax-new/attn-train/train \
                                                ../data/saved-translation-models/iwslt-ende-bahdanau-softmax-new/attn-dev/dev \
                                                ../data/saved-translation-models/iwslt-ende-bahdanau-softmax-new/attn-test/test \
                                   --emb-path ../data/embs/glove/glove.840B.300d.txt \
                                   --output-path ../data/embs/glove/glove.840B.300d.small.raw.pickle.iwslt.en \
                                   --format glove  \
                                   --binary
