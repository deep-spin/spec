from spec.dataset.corpora.agnews import AGNewsCorpus
from spec.dataset.corpora.imdb import IMDBCorpus
from spec.dataset.corpora.mnli import MNLICorpus
from spec.dataset.corpora.snli import SNLICorpus
from spec.dataset.corpora.esnli import ESNLICorpus
from spec.dataset.corpora.sst import SSTCorpus
from spec.dataset.corpora.ttsbr import TTSBRCorpus
from spec.dataset.corpora.yelp import YelpCorpus
from spec.dataset.corpora.iwslt import IWSLTCorpus

available_corpora = {
    'agnews': AGNewsCorpus,
    'imdb': IMDBCorpus,
    'esnli': ESNLICorpus,
    'snli': SNLICorpus,
    'mnli': MNLICorpus,
    'sst': SSTCorpus,
    'ttsbr': TTSBRCorpus,
    'yelp': YelpCorpus,
    'iwslt': IWSLTCorpus
}
