import logging

logger = logging.getLogger(__name__)


class Corpus:

    def __init__(self, fields_tuples, lazy=False):
        """
        Base class for a Corpus.

        Args:
            fields_tuples (list of (str, torchtext.field)): a list of tuples
                where the first element is an attr name and the second is a
                torchtext's Field object.
            lazy (bool): whether to read this dataset lazily
        """
        # list of name of attrs and their corresponding torchtext fields
        fields_dict = dict(fields_tuples)
        # hack for torchtext -.-
        self.fields_dict = dict(zip(fields_dict.keys(), fields_dict.items()))
        # lazy loading properties
        self.lazy = lazy
        self.corpus_path = None
        self._current_line = 0
        self._nb_examples = 0
        self.closed = False
        self.read_once = False
        self.file = None

    def _read(self, *files):
        """
        The method used to read the dataset.

        Args:
            files (io.Object): one or more instances of a corpus file
        """
        raise NotImplementedError

    def make_torchtext_example(self, *args, **kwargs):
        """
        Create a new torch.data.Example from args and kwargs.
        """
        raise NotImplementedError

    @staticmethod
    def create_fields_tuples():
        """
        Create torchtext.data.Fields for this specific corpus
        """
        raise NotImplementedError

    def read(self, corpus_path):
        """
        Args:
            corpus_path (str): path to a corpus (file).
            If you want a different logic you should subclass this method.
        Returns:
            A generator of torchtext.data.Example if `self.lazy` is true or a
            list of torchtext.data.Example otherwise. Check __iter__
        """
        self.corpus_path = corpus_path
        self.open(self.corpus_path)
        if self.lazy is True:
            return self
        else:
            return list(self)

    def __iter__(self):
        for ex in self._read(self.file):
            self._current_line += 1
            yield ex
        self.start_over()
        self.read_once = True

    def open(self, corpus_path):
        self._current_line = 0
        self.closed = False
        self.file = open(corpus_path, 'r', encoding='utf8')
        return self.file

    def start_over(self):
        self._nb_examples = self._current_line
        self._current_line = 0
        self.file.seek(0)

    def close(self):
        self.file.close()
        self.file = None
        self.closed = True

    def __del__(self):
        if self.file is not None:
            self.close()

    @property
    def nb_examples(self):
        if self.lazy is True and self.read_once is False:
            raise ValueError('You should read the entire file at least once to '
                             'know the number of examples in it.')
        return self._nb_examples

    def __len__(self):
        return self.nb_examples
