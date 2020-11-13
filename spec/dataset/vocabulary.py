import warnings
from collections import defaultdict

from torchtext.vocab import Vocab

from spec.constants import UNK_ID, UNK, PAD, START, STOP


class Vocabulary(Vocab):
    """Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(
        self,
        counter,
        max_size=None,
        min_freq=1,
        specials=None,
        vectors=None,
        unk_init=None,
        vectors_cache=None,
        keep_rare_with_vectors=True,
        add_vectors_vocab=False,
        specials_first=True,
    ):
        """Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocab word
                vectors to zero vectors; can be any function that takes in a
                Tensor and returns a Tensor of the same size.
                Default: torch.Tensor.zero_
            vectors_cache: dir for cached vectors. Default: '.vector_cache'
            keep_rare_with_vectors: if True and a vectors object is passed,
                then it will add words that appears less than min_freq
                but are in vectors vocabulary. Default: True.
            add_vectors_vocab: by default, the vocab is built using only words
            from the provided datasets. If this flag is true, the vocab will
            add words that are not in the datasets but are in the vectors vocab
            (e.g. words from polyglot vectors). Default: False
        """
        if specials is None:
            specials = ['<pad>']

        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = []
        if specials_first:
            self.itos = list(specials)
            max_size = None if max_size is None else max_size + len(self.itos)

        # frequencies of special tokens are not counted when building vocab
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        if not isinstance(vectors, list) and vectors is not None:
            vectors = [vectors]

        # add words that appears less than min_freq but are in embeddings
        # vocabulary
        for word, freq in words_and_frequencies:
            if freq < min_freq:
                if vectors is not None and keep_rare_with_vectors:
                    for v in vectors:
                        if word in v.stoi:
                            self.itos.append(word)
                else:
                    break
            elif len(self.itos) == max_size:
                break
            else:
                self.itos.append(word)

        self.orig_stoi = defaultdict(self._default_unk_index)
        self.orig_stoi.update({tok: i for i, tok in enumerate(self.itos)})

        if add_vectors_vocab and vectors is not None:
            if max_size is not None and \
                sum(len(v.stoi) for v in vectors) + \
                    len(self.itos) > max_size:
                warnings.warn('Adding the vectors vocabulary will make '
                              'len(vocab) > max_vocab_size!')
            vset = set()
            for v in vectors:
                vset.update(v.stoi.keys())
            v_itos = vset - set(self.itos)
            self.itos.extend(list(v_itos))

        if '<unk>' in specials:  # hard-coded for now
            self.stoi = defaultdict(self._default_unk_index)
        else:
            self.stoi = defaultdict()

        if not specials_first:
            self.itos.extend(list(specials))

        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectors = None
        self.vectors_words = set()
        if vectors is not None:
            for v in vectors:
                self.vectors_words.update(v.stoi.keys())
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def _default_unk_index(self):
        return UNK_ID  # should be zero

    def __setstate__(self, state):
        stoi = defaultdict(self._default_unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)


def merge_vocabularies(
    vocab_a,
    vocab_b,
    max_size=None,
    vectors=None,
    **kwargs
):
    merged = vocab_a.freqs + vocab_b.freqs
    return Vocabulary(merged,
                      specials=[UNK, PAD, START, STOP],
                      max_size=max_size,
                      vectors=vectors,
                      **kwargs)
