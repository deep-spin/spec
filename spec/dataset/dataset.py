import warnings

from torchtext.data import interleave_keys

from spec.dataset.corpora import available_corpora
from spec.dataset.corpora.text import TextCorpus
from spec.dataset.corpora.text_pair import TextPairCorpus
from spec.dataset.corpora.text_pair_with_marks import TextPairWithMarksCorpus
from spec.dataset.modules.dataset import LazyDataset


def build(path, fields_tuples, options):
    def filter_len(x):
        return options.min_length <= len(x.words) <= options.max_length
    corpus_cls = available_corpora[options.corpus]
    corpus = corpus_cls(fields_tuples, lazy=options.lazy_loading)
    examples = corpus.read(path)
    return TextDataset(examples, fields_tuples, filter_pred=filter_len)


def build_texts(texts, fields_tuples, options):
    def filter_len(x):
        return options.min_length <= len(x.words) <= options.max_length
    corpus = TextCorpus(fields_tuples, lazy=options.lazy_loading)
    examples = corpus.read(texts)
    return TextDataset(examples, fields_tuples, filter_pred=filter_len)


def build_pair_texts(texts_ab, fields_tuples, options):
    def filter_len(x):
        return options.min_length <= len(x.words) <= options.max_length
    corpus = TextPairCorpus(fields_tuples, lazy=options.lazy_loading)
    examples = corpus.read(texts_ab)
    return TextDataset(examples, fields_tuples, filter_pred=filter_len)


def build_pair_texts_with_marks(texts_abc, fields_tuples, options):
    def filter_len(x):
        return options.min_length <= len(x.words) <= options.max_length
    corpus = TextPairWithMarksCorpus(fields_tuples, lazy=options.lazy_loading)
    examples = corpus.read(texts_abc)
    return TextDataset(examples, fields_tuples, filter_pred=filter_len)


class TextDataset(LazyDataset):
    """Defines a dataset for TextClassification"""

    @staticmethod
    def sort_key(ex):
        """Use the number of words as the criterion for sorting a batch."""
        if hasattr(ex, 'words_hyp'):
            return interleave_keys(len(ex.words), len(ex.words_hyp))
        return len(ex.words)

    def __init__(self, examples, fields_tuples, filter_pred=None):
        """Create a dataset from a list of Examples and Fields.

        Arguments:
            examples: A list or a generator of examples. Usually, the output
                of corpus.read()
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default: None.
        """
        is_lazy = hasattr(examples, 'lazy') and examples.lazy is True
        super().__init__(examples, fields_tuples, filter_pred, not is_lazy)

    def get_loss_weights(self):
        from sklearn.utils.class_weight import compute_class_weight
        target_vocab = self.fields['target'].vocab.stoi
        y = [target_vocab[t] for ex in self.examples for t in ex.target]
        classes = list(set(y))
        return compute_class_weight('balanced', classes, y)

    def __len__(self):
        try:
            return len(self.examples)
        except ValueError:
            warnings.warn("Corpus loaded in lazy mode and its length was not "
                          "determined yet. Returning 0 for now since in order "
                          "to calculate this number we'd have to go through "
                          "the entire dataset at least once, which can be very "
                          "expensive for large datasets.")
            return 0
