import csv
import logging
import re
import string

import nltk
import torchtext


from spec.dataset import fields
from spec.dataset.corpora.corpus import Corpus

logger = logging.getLogger(__name__)


def filter_text_with_marks(text):
    text = text.replace('"', "'").replace('\\', '')
    # this is too dangerous but fix a particular annoying case:
    # text = text.replace('**', '*')
    puncts = re.escape(string.punctuation)
    text = re.sub(r'([%s])(\*)' % puncts, r'\g<1> \g<2>', text)
    text = re.sub(r'(\*)([%s])' % puncts, r'\g<1> \g<2>', text)
    tokenizer = nltk.WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)
    new_text = []
    new_marks = []
    b = False
    for i in range(len(tokens)):
        tk_i = tokens[i]
        if tk_i != '*':
            if '*' in tk_i:
                print(tk_i, tokens)
            new_text.append(tk_i)
        if tk_i == '*' and b is False:
            b = True
        elif tk_i == '*' and b is True:
            b = False
        if tk_i == '*' and b is True:
            new_marks.append(len(new_text))
    new_text = ' '.join(new_text).replace('*', '')
    return new_text, new_marks


class ESNLICorpus(Corpus):

    @staticmethod
    def create_fields_tuples():
        words_field = fields.WordsField()
        fields_tuples = [
            ('words', words_field),
            ('words_hyp', words_field),
            ('words_expl', words_field),
            ('marks', fields.MarkIndexesField()),
            ('marks_hyp', fields.MarkIndexesField()),
            ('target', fields.TagsField())
        ]
        return fields_tuples

    def _read(self, file):
        # important: the cases where the annotators disagreed are
        # already removed from this dataset, so there are 549367 examples here
        # these are the columns of the csv file:
        # 'pairID': 0
        # 'gold_label': 1
        # 'Sentence1': 2
        # 'Sentence2': 3
        # 'Explanation_1': 4
        # 'Sentence1_marked_1': 5
        # 'Sentence2_marked_1': 6
        # 'Sentence1_Highlighted_1': 7
        # 'Sentence2_Highlighted_1': 8
        # 'Explanation_2': 9
        # 'Sentence1_marked_2': 10
        # 'Sentence2_marked_2': 11
        # 'Sentence1_Highlighted_2': 12
        # 'Sentence2_Highlighted_2': 13
        # 'Explanation_3': 14
        # 'Sentence1_marked_3': 15
        # 'Sentence2_marked_3': 16
        # 'Sentence1_Highlighted_3': 17
        # 'Sentence2_Highlighted_3': 18

        # For entailment pairs, we required at least one word in the premise to
        # be highlighted. For contradiction pairs, we required highlighting at
        # least one word in both the premise and the hypothesis. For neutral
        # pairs, we only allowed highlighting words in the hypothesis, in order
        # to strongly emphasize the asymmetry in this relation and to prevent
        # workers from confusing the premise with the hypothesis.
        csv_reader = csv.reader(file, delimiter=',')
        for ell, line in enumerate(csv_reader):
            if ell == 0:  # header
                continue
            # pair_id = line[0]
            label = line[1]
            premise = line[2]
            hypothesis = line[3]
            explanation = line[4]
            # worked_id = line[5]

            if len(line) == 10:   # training data
                tokenizer = nltk.WordPunctTokenizer()
                orig_prem_tokens = tokenizer.tokenize(
                    premise.replace('"', "'").replace('\\', ''))
                orig_hyp_tokens = tokenizer.tokenize(
                    hypothesis.replace('"', "'").replace('\\', ''))

                explanation, _ = filter_text_with_marks(explanation)

                # the file is not very well aligned, so instead I decided to
                # parse the * marks from scratch instead
                premise_marked = line[6].strip()
                hypothesis_marked = line[7].strip()
                if premise_marked == '' or hypothesis_marked == '':
                    msg = "Example without explanation at line n {}: {}"
                    logger.debug(msg.format(ell, line))
                    continue

                premise, premise_marks = filter_text_with_marks(premise_marked)
                premise_marks = [premise_marks]
                assert (orig_prem_tokens == premise.split())

                hypothesis, hypothesis_marks = filter_text_with_marks(hypothesis_marked)  # noqa
                hypothesis_marks = [hypothesis_marks]
                assert (orig_hyp_tokens == hypothesis.split() or ell == 102709)

                # old code:
                # premise_marks = [[]]
                # hypothesis_marks = [[]]
                # try:
                #     if line[8] != '{}':
                #         premise_marks = [
                #             [int(i) for i in line[8].split(',')]
                #         ]
                #     if line[9] != '{}':
                #         hypothesis_marks = [
                #             [int(i) for i in line[9].split(',')]
                #         ]
                # except ValueError:
                #     # there are some garbages: for instance, line 88026.
                #     # there is no hypothesis and no explanation marks.
                #     # in total, 25 examples in the training set are removed
                #     msg = "Example without explanation at line n {}: {}"
                #     logger.debug(msg.format(ell, line))
                #     continue

            else:  # test data (3 explanations)
                tokenizer = nltk.WordPunctTokenizer()
                orig_prem_tokens = tokenizer.tokenize(
                    premise.replace('"', "'").replace('\\', ''))
                orig_hyp_tokens = tokenizer.tokenize(
                    hypothesis.replace('"', "'").replace('\\', ''))

                explanation = line[4]
                # explanation = line[9]
                # explanation = line[14]
                explanation, _ = filter_text_with_marks(explanation)

                premise_marks = []
                hypothesis_marks = []
                prem_marks_line_idxs = [5, 10, 15]
                hyp_marks_line_idxs = [6, 11, 16]
                for j in prem_marks_line_idxs:
                    premise_marked = line[j].strip()
                    if premise_marked == '':
                        continue
                    premise, marks = filter_text_with_marks(premise_marked)
                    premise_marks.append(marks)
                assert (orig_prem_tokens == premise.split())

                for j in hyp_marks_line_idxs:
                    hypothesis_marked = line[j].strip()
                    if hypothesis_marked == '':
                        continue
                    hypothesis, marks = filter_text_with_marks(hypothesis_marked)  # noqa
                    hypothesis_marks.append(marks)
                assert (orig_hyp_tokens == hypothesis.split())

                if label in ['entailment', 'contradiction']:
                    assert len(premise_marks) > 0

                # ignore neutral examples in the test set
                if label == 'neutral' and 'test' in file.name:
                    continue

            yield self.make_torchtext_example(premise,
                                              hypothesis,
                                              explanation,
                                              premise_marks,
                                              hypothesis_marks,
                                              label)

    def make_torchtext_example(
        self, prem, hyp, expl, prem_marks, hyp_marks, label
    ):
        ex = {'words': prem,
              'words_hyp': hyp,
              'words_expl': expl,
              'marks': prem_marks,
              'marks_hyp': hyp_marks,
              'target': label}
        if 'target' not in self.fields_dict.keys():
            del ex['target']
        assert ex.keys() == self.fields_dict.keys()
        return torchtext.data.Example.fromdict(ex, self.fields_dict)


if __name__ == '__main__':
    fields_tuples = ESNLICorpus.create_fields_tuples()
    words_field = fields_tuples[0][1]
    target_field = fields_tuples[-1][1]
    lazy = True
    corpus_path = '../../../data/corpus/esnli/esnli_test.csv'

    train_corpus = ESNLICorpus(fields_tuples, lazy=lazy)
    train_examples = train_corpus.read(corpus_path)
    labels_count = {}
    avg_len = 0
    avg_len_hyp = 0
    avg_len_marks = 0
    avg_len_marks_hyp = 0
    avg_len_marks_x = [0, 0, 0]
    avg_len_marks_hyp_x = [0, 0, 0]
    total = 0
    tp, th = [0, 0, 0], [0, 0, 0]
    puncts = "!\"#$%&'()*+,-/:;<=>?@[\\]^_`{|}~"
    for i, ex in enumerate(train_examples):
        if ex.target[0] == 'neutral':
            continue

        if ex.target[0] not in labels_count:
            labels_count[ex.target[0]] = 0
        labels_count[ex.target[0]] += 1
        for punc in puncts:
            if punc in ex.words:
                total += 1
                break

        avg_len += len(ex.words)
        avg_len_hyp += len(ex.words_hyp)
        avg_len_marks += sum(len(x) for x in ex.marks) / len(ex.marks)
        avg_len_marks_hyp += sum(len(x) for x in ex.marks_hyp) / len(ex.marks_hyp)  # noqa
        if len(ex.marks[0]) > 0 and max(ex.marks[0]) >= len(ex.words):
            import ipdb; ipdb.set_trace()
        # sel = ' '.join([ex.words[j] for j in ex.marks[0]])
        # print(' '.join(ex.words), '|', sel)
        # print(' '.join(ex.words_hyp), '   |   ', ' '.join(ex.words))
        for j, x in enumerate(ex.marks):
            tp[j] += int(len(x) > 0)
            avg_len_marks_x[j] += len(x)

        for j, x in enumerate(ex.marks_hyp):
            th[j] += int(len(x) > 0)
            avg_len_marks_hyp_x[j] += len(x)

    print(labels_count)
    print(total)
    print('avg_len:', avg_len / len(train_examples))
    print('avg_len_hyp:', avg_len_hyp / len(train_examples))
    print('avg_len_marks:', avg_len_marks / sum(tp))
    print('avg_len_marks_hyp:', avg_len_marks_hyp / sum(th))
    for i, a in enumerate(avg_len_marks_x):
        print('avg_len_marks {}: {}'.format(i, a/tp[i]))
    for i, a in enumerate(avg_len_marks_hyp_x):
        print('avg_len_marks_hyp_x {}: {}'.format(i, a/th[i]))
    print('nb examples:', len(train_examples))
    print('')
