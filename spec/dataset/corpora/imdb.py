from itertools import chain
from pathlib import Path

import nltk
import torchtext

from spec.dataset import fields
from spec.dataset.corpora.corpus import Corpus


class IMDBCorpus(Corpus):

    @staticmethod
    def create_fields_tuples():
        # if you choose tokenizer='spacy', please install the en package:
        # python3 -m spacy download en
        tokenizer = nltk.WordPunctTokenizer()
        # tokenizer = nltk.TreebankWordTokenizer()
        fields_tuples = [
            ('words', fields.WordsField(tokenize=tokenizer.tokenize)),
            ('target', fields.TagsField())
        ]
        return fields_tuples

    def read(self, corpus_path):
        new_file_path = Path(corpus_path, 'data.txt')
        if not new_file_path.exists():
            paths = chain(sorted(Path(corpus_path, 'neg').glob('*.txt')),
                          sorted(Path(corpus_path, 'pos').glob('*.txt')))
            new_file = new_file_path.open('w', encoding='utf8')
            for file_path in paths:
                content = file_path.read_text().strip()
                # import ipdb; ipdb.set_trace()
                content = content.replace('<br>', ' <br> ')
                content = content.replace('<br >', ' <br> ')
                content = content.replace('<br />', ' <br> ')
                content = content.replace('<br/>', ' <br> ')
                label = '1' if 'pos' in str(file_path) else '0'
                new_file.write(label + ' ' + content + '\n')
            new_file.seek(0)
            new_file.close()
        self.corpus_path = str(new_file_path)
        self.open(self.corpus_path)
        if self.lazy is True:
            return self
        else:
            return list(self)

    def _read(self, file):
        for line in file:
            line = line.strip().split()
            if line:
                label = line[0]
                text = ' '.join(line[1:])
                yield self.make_torchtext_example(text, label)

    def make_torchtext_example(self, text, label=None):
        ex = {'words': text, 'target': label}
        if 'target' not in self.fields_dict.keys():
            del ex['target']
        assert ex.keys() == self.fields_dict.keys()
        return torchtext.data.Example.fromdict(ex, self.fields_dict)


if __name__ == '__main__':
    from spec.dataset.corpora.test_corpus import quick_test
    quick_test(
        IMDBCorpus,
        '../../../data/corpus/imdb/test/',
        lazy=True,
    )
