import logging
from collections import defaultdict
from pathlib import Path

import torch
from torchtext.data import Field

from spec import constants
from spec.dataset.vectors import available_embeddings
from spec.dataset.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def load_vectors(options, side='source'):
    vectors = None
    # load the word embeddings only if a correct format is provided
    if side == 'source' and options.embeddings_format is not None:
        assert options.embeddings_format in available_embeddings.keys()
        logger.info('Loading {} word embeddings from: {}'.format(
            options.embeddings_format,
            options.embeddings_path)
        )
        emb_cls = available_embeddings[options.embeddings_format]
        vectors = emb_cls(options.embeddings_path,
                          binary=options.embeddings_binary)
    elif side == 'target' and options.embeddings_format is not None:
        logger.info('Loading {} word embeddings from: {}'.format(
            options.embeddings_format_target,
            options.embeddings_path_target)
        )
        emb_cls = available_embeddings[options.embeddings_format_target]
        vectors = emb_cls(options.embeddings_path_target,
                          binary=options.embeddings_binary_target)
    return vectors


def build_vocabs(fields_tuples, train_dataset, all_datasets, options):
    # transform fields_tuples to a dict in order to access fields easily
    dict_fields = defaultdict(lambda: None)
    dict_fields.update(dict(fields_tuples))

    # build vocab for words based on the training set
    words_field = dict_fields['words']

    # build vocab based on all datasets
    target_field = dict_fields['target']

    if options.corpus == 'iwslt':
        vectors = load_vectors(options, side='source')
        words_field.build_vocab(
            train_dataset,
            vectors=vectors,
            max_size=options.vocab_size,
            min_freq=options.vocab_min_frequency,
        )
        vectors = load_vectors(options, side='target')
        target_field.build_vocab(
            *all_datasets,
            vectors=vectors,
            max_size=options.vocab_size,
            min_freq=options.vocab_min_frequency,
            specials_first=False
        )
    else:
        vectors = load_vectors(options)
        words_field.build_vocab(
            train_dataset,
            vectors=vectors,
            max_size=options.vocab_size,
            min_freq=options.vocab_min_frequency,
            keep_rare_with_vectors=options.keep_rare_with_vectors,
            add_vectors_vocab=options.add_embeddings_vocab
        )
        target_field.build_vocab(*all_datasets, specials_first=False)

    # set global constants to their correct value
    constants.PAD_ID = dict_fields['words'].vocab.stoi[constants.PAD]

    # set target pad id (useful for seq classification)
    if constants.PAD in target_field.vocab.stoi:
        constants.TARGET_PAD_ID = target_field.vocab.stoi[constants.PAD]


def load_vocabs(path, fields_tuples):
    vocab_path = Path(path, constants.VOCAB)

    # load vocabs for each field and transform it to dict to access it easily
    vocabs = torch.load(str(vocab_path),
                        map_location=lambda storage, loc: storage)
    vocabs = dict(vocabs)

    # set field.vocab to its correct vocab object
    for name, field in fields_tuples:
        if field.use_vocab:
            if name == 'words_expl':
                field.vocab = vocabs['words']
                continue
            field.vocab = vocabs[name]

    # transform fields_tuples to a dict in order to access fields easily
    dict_fields = dict(fields_tuples)

    # ensure global constants to their correct value
    words_field = dict_fields['words']
    target_field = dict_fields['target']
    constants.PAD_ID = words_field.vocab.stoi[constants.PAD]
    if constants.PAD in target_field.vocab.stoi:
        constants.TARGET_PAD_ID = target_field.vocab.stoi[constants.PAD]


def save_vocabs(path, fields_tuples):
    # list of fields name and their vocab
    vocabs = []
    for name, field in fields_tuples:
        if field.use_vocab:
            vocabs.append((name, field.vocab))

    # save vectors in a temporary dict and save the vocabs
    vectors = {}
    for name, vocab in vocabs:
        vectors[name] = vocab.vectors
        vocab.vectors = None
    vocab_path = Path(path, constants.VOCAB)
    torch.save(vocabs, str(vocab_path))

    # restore vectors -> useful if we want to use fields later
    for name, vocab in vocabs:
        vocab.vectors = vectors[name]


class WordsField(Field):
    """Defines a field for word tokens with default
       values from constant.py and with the vocabulary
       defined in vocabulary.py."""

    def __init__(self, **kwargs):
        super().__init__(unk_token=constants.UNK,
                         pad_token=constants.PAD,
                         batch_first=True,
                         **kwargs)
        self.vocab_cls = Vocabulary


class TagsField(Field):
    """Defines a field for text tags with default
       values from constant.py and with the vocabulary
       defined in vocabulary.py."""

    def __init__(self, **kwargs):
        super().__init__(unk_token=None,
                         pad_token=constants.PAD,
                         sequential=True,
                         is_target=True,
                         batch_first=True,
                         **kwargs)
        self.vocab_cls = Vocabulary


class LabelField(Field):
    """Defines a field for text labels. Equivalent to
    torchtext's LabelField but with my own vocabulary."""

    def __init__(self, **kwargs):
        super().__init__(unk_token=None,
                         pad_token=None,
                         sequential=False,
                         is_target=True,
                         batch_first=True,
                         **kwargs)
        self.vocab_cls = Vocabulary


class AttentionField(Field):
    """Defines a field for saved attention weights."""
    def __init__(self, **kwargs):
        super().__init__(unk_token=None,
                         pad_token=None,
                         batch_first=True,
                         use_vocab=False,
                         sequential=False,
                         **kwargs)

    def process(self, batch, device=None):
        """ Just return a tensor of the original batch."""
        max_len_src = max([b.shape[0] for b in batch])
        max_len_trg = max([b.shape[1] for b in batch])
        new_batch = torch.zeros(len(batch), max_len_src, max_len_trg)
        for i, b in enumerate(batch):
            new_batch[i, :b.shape[0], :b.shape[1]] = torch.from_numpy(b)
        return new_batch.to(device)


class MarkIndexesField(Field):
    """Defines a field for highlighted indexes (list of ints)."""
    def __init__(self, **kwargs):
        super().__init__(unk_token=None,
                         pad_token=None,
                         batch_first=True,
                         use_vocab=False,
                         sequential=False,
                         **kwargs)

    def process(self, batch, device=None):
        """ Just return a tensor of the original batch."""
        max_expl_len = max([len(b) for b in batch])
        max_marks_len = max([len(b2) for b in batch for b2 in b])
        new_batch = torch.zeros(len(batch), max_expl_len, max_marks_len)
        new_batch.fill_(-1)  # -1 are considered pad positions
        for i, b in enumerate(batch):
            for j, b2 in enumerate(b):
                new_batch[i, j, :len(b2)] = torch.tensor(b2)
        return new_batch.to(device).long()


class SpectrogramField(Field):
    """
    Defines a field for a audio spectrogram.
    """
    def __init__(self, **kwargs):
        super().__init__(unk_token=None,
                         pad_token=None,
                         batch_first=True,
                         use_vocab=False,
                         sequential=False,
                         **kwargs)

    def process(self, batch, device=None):
        """ Just return a tensor of the original batch."""
        max_len_src = max([b.shape[0] for b in batch])
        max_len_trg = max([b.shape[1] for b in batch])
        new_batch = torch.zeros(len(batch), max_len_src, max_len_trg)
        lengths = []
        for i, b in enumerate(batch):
            new_batch[i, :b.shape[0], :b.shape[1]] = b
            lengths.append(b.shape[0])
        lengths = torch.tensor(lengths)
        return new_batch.to(device), lengths.to(device)
