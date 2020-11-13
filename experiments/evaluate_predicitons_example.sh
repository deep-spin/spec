# AGNEWS
###############

python3 scripts/evaluate_predictions.py \
                --corpus agnews \
                --corpus-path data/corpus/agnews/test.xml \
                --predictions-path data/predictions/test-agnews-softmax/predictions.txt \
                --average macro

python3 scripts/evaluate_predictions.py \
                --corpus agnews \
                --corpus-path data/corpus/agnews/test.xml \
                --predictions-path data/predictions/test-agnews-sparsemax/predictions.txt \
                --average macro

python3 scripts/evaluate_predictions.py \
                --corpus agnews \
                --corpus-path data/corpus/agnews/test.xml \
                --predictions-path data/predictions/test-agnews-entmax15/predictions.txt \
                --average macro


# IMDB
###############

python3 scripts/evaluate_predictions.py \
                --corpus imdb \
                --corpus-path data/corpus/imdb/test/ \
                --predictions-path data/predictions/test-imdb-softmax/predictions.txt \
                --average macro

python3 scripts/evaluate_predictions.py \
                --corpus imdb \
                --corpus-path data/corpus/imdb/test/ \
                --predictions-path data/predictions/test-imdb-sparsemax/predictions.txt \
                --average macro

python3 scripts/evaluate_predictions.py \
                --corpus imdb \
                --corpus-path data/corpus/imdb/test/ \
                --predictions-path data/predictions/test-imdb-entmax15/predictions.txt \
                --average macro



# SNLI
###############

python3 scripts/evaluate_predictions.py \
                --corpus snli \
                --corpus-path data/corpus/snli/snli_1.0_test.jsonl \
                --predictions-path data/predictions/test-snli-softmax/predictions.txt \
                --average macro

python3 scripts/evaluate_predictions.py \
                --corpus snli \
                --corpus-path data/corpus/snli/snli_1.0_test.jsonl \
                --predictions-path data/predictions/test-snli-sparsemax/predictions.txt \
                --average macro

python3 scripts/evaluate_predictions.py \
                --corpus snli \
                --corpus-path data/corpus/snli/snli_1.0_test.jsonl \
                --predictions-path data/predictions/test-snli-entmax15/predictions.txt \
                --average macro


# SST
###############

python3 scripts/evaluate_predictions.py \
                --corpus sst \
                --corpus-path data/corpus/sst/test.txt \
                --predictions-path data/predictions/test-sst-softmax/predictions.txt \
                --average macro

python3 scripts/evaluate_predictions.py \
                --corpus sst \
                --corpus-path data/corpus/sst/test.txt \
                --predictions-path data/predictions/test-sst-sparsemax/predictions.txt \
                --average macro

python3 scripts/evaluate_predictions.py \
                --corpus sst \
                --corpus-path data/corpus/sst/test.txt \
                --predictions-path data/predictions/test-sst-entmax15/predictions.txt \
                --average macro



# YELP
###############

python3 scripts/evaluate_predictions.py \
                --corpus yelp \
                --corpus-path data/corpus/yelp/review_test.json \
                --predictions-path data/predictions/test-yelp-softmax/predictions.txt \
                --average macro

python3 scripts/evaluate_predictions.py \
                --corpus yelp \
                --corpus-path data/corpus/yelp/review_test.json \
                --predictions-path data/predictions/test-yelp-sparsemax/predictions.txt \
                --average macro

python3 scripts/evaluate_predictions.py \
                --corpus yelp \
                --corpus-path data/corpus/yelp/review_test.json \
                --predictions-path data/predictions/test-yelp-entmax15/predictions.txt \
                --average macro
