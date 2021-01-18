#!/usr/bin/env bash

# generate train and test data for AGNEWS
python3 scripts/partition_agnews_corpus.py \
        'data/corpus/agnews/newsspace200.xml' \
        'data/corpus/agnews/train.xml' \
        'data/corpus/agnews/test.xml'

# generate dev data from the training data for AGNEWS
python3 scripts/partition_agnews_corpus.py \
        'data/corpus/agnews/train.xml' \
        'data/corpus/agnews/train.xml' \
        'data/corpus/agnews/dev.xml'


# make sure to rename your imdb/train to imdb/train-original to don't collapse
# the original training file
mv data/corpus/imdb/train data/corpus/imdb/train-original
mkdir data/corpus/imdb/train
mkdir data/corpus/imdb/dev

# generate dev data from the training data for IMDB
python3 scripts/partition_imdb_corpus.py \
        'data/corpus/imdb/train-original/' \
        'data/corpus/imdb/train/data.txt' \
        'data/corpus/imdb/dev/data.txt'


# generate train and test data for YELP
#python3 scripts/partition_yelp_corpus.py \
#        'data/corpus/yelp/review.json' \
#        'data/corpus/yelp-small/review_train.json' \
#        'data/corpus/yelp-small/review_test.json'

# generate dev data from the training data for YELP
#python3 scripts/partition_yelp_corpus.py \
#        'data/corpus/yelp/review_train.json' \
#        'data/corpus/yelp/review_train.json' \
#        'data/corpus/yelp/review_dev.json'

# generate train and test data for the small version of YELP
mkdir data/corpus/yelp-small
python3 scripts/partition_yelp_corpus_small.py  \
        'data/corpus/yelp/review.json' \
        'data/corpus/yelp-small/review_train.json' \
        'data/corpus/yelp-small/review_test.json' \
        'data/corpus/yelp-small/review_dev.json'
