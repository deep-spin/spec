import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from spec import constants
from spec.explainers.explainer import Explainer
from spec.explainers.utils import filter_word_ids_with_non_zero_probability
from spec.initialization import init_xavier
from spec.modules.attention import Attention
from spec.modules.scorer import SelfAdditiveScorer, OperationScorer


class PostHocExplainer(Explainer):

    def __init__(self, fields_tuples, options):
        super().__init__(fields_tuples)
        self.words_vocab_size = len(self.fields_dict['words'].vocab)
        self.explainer_attn_top_k = options.explainer_attn_top_k
        self.message_type = options.message_type
        self.emb_size = options.word_embeddings_size
        self.explainer_cheat_ratio = options.explainer_cheat_ratio
        self.explainer_idf = options.explainer_idf
        self.explainer_ignore_top_words = options.explainer_ignore_top_words
        self.valid_top_word_ids = None

        # create an embedding layer to reencode the words
        embeddings_weight = None
        if self.fields_dict['words'].vocab.vectors is not None:
            embeddings_weight = self.fields_dict['words'].vocab.vectors
            options.word_embeddings_size = embeddings_weight.size(1)

        self.word_emb_explainer = nn.Embedding(
            num_embeddings=len(self.fields_dict['words'].vocab),
            embedding_dim=options.word_embeddings_size,
            padding_idx=constants.PAD_ID,
            _weight=embeddings_weight,
        )
        if options.freeze_embeddings:
            self.word_emb_explainer.weight.requires_grad = False

        hidden_size = options.hidden_size[0]
        self.lstm_explainer = nn.LSTM(options.word_embeddings_size,
                                      hidden_size,
                                      batch_first=True,
                                      bidirectional=True)

        n = 2 if options.bidirectional else 1
        concat_features_size = n * options.hidden_size[0]
        concat_features_size += self.nb_classes
        self.scorer = OperationScorer(concat_features_size,
                                      n * options.hidden_size[0],
                                      concat_features_size // 4,
                                      op='add',
                                      scaled=True)

        self.attn = Attention(self.scorer,
                              dropout=options.attn_dropout,
                              max_activation=options.attn_max_activation)

        # save for later
        self.lstm_hidden = None
        self.lstm_out = None

        init_xavier(self.lstm_explainer, dist='uniform')
        init_xavier(self.attn, dist='uniform')

        self.rec_layer_ae = nn.Linear(self.words_vocab_size, 100)
        self.rec_layer_ae_2 = nn.Linear(100, self.words_vocab_size)
        self.rec_layer_pred = nn.Linear(n * hidden_size, n * hidden_size)

        self.nb_documents = 0
        self.idf = None
        if self.explainer_idf in ['embs', 'scores']:
            self.calc_idf(options)

    def calc_idf(self, options):
        from spec.dataset.corpora import available_corpora
        self.idf = torch.zeros(self.words_vocab_size)
        self.idf.requires_grad = False
        if options.gpu_id is not None:
            self.idf = self.idf.to(options.gpu_id)

        corpus_cls = available_corpora[options.corpus]
        self.nb_documents = 0

        stop_words = ["i", "me", "my", "myself", "we", "our", "ours",
                      "ourselves", "you", "your", "yours", "yourself",
                      "yourselves", "he", "him", "his", "himself", "she",
                      "her", "hers", "herself", "it", "its", "itself", "they",
                      "them", "their", "theirs", "themselves", "what", "which",
                      "who", "whom", "this", "that", "these", "those", "am",
                      "is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "having", "do", "does", "did",
                      "doing", "a", "an", "the", "and", "but", "if", "or",
                      "because", "as", "until", "while", "of", "at", "by",
                      "for", "with", "about", "against", "between", "into",
                      "through", "during", "before", "after", "above", "below",
                      "to", "from", "up", "down", "in", "out", "on", "off",
                      "over", "under", "again", "further", "then", "once",
                      "here", "there", "when", "where", "why", "how", "all",
                      "any", "both", "each", "few", "more", "most", "other",
                      "some", "such", "no", "nor", "not", "only", "own",
                      "same", "so", "than", "too", "very", "s", "t", "can",
                      "will", "just", "don", "should", "now"]

        # capitalize stopwords (in case we didnt lowercased the corpus)
        stop_words += [w.capitalize() for w in stop_words]

        # add some punctuations
        stop_words += [',', '.', '/', ':', ';', '(', ')', '[', ']', '{', '}',
                       '"', "'", '<', '>', '-']
        stop_words_ids = [self.fields_dict['words'].vocab.stoi[w]
                          for w in stop_words
                          if w in self.fields_dict['words'].vocab.stoi]
        stop_words_ids = torch.tensor(stop_words_ids).long().to(options.gpu_id)

        if self.explainer_ignore_top_words > 0:
            pids = stop_words_ids
            # pids = self.idf.flatten().topk(self.explainer_ignore_top_words)[1]
            self.idf = torch.ones(self.words_vocab_size).to(options.gpu_id)
            self.idf.requires_grad = False
            self.idf[pids] = 0  # will be set to -inf by log(idf) in attention

        else:
            # smooth directly
            # self.idf[constants.UNK_ID] = 1
            # self.idf[self.idf.int() == 0] = 1
            # assert (torch.all(self.idf >= 1))
            # div = torch.div(self.nb_documents, self.idf)
            # assert (torch.all(div >= 1))
            # self.idf = torch.log(div).unsqueeze(0)
            # import ipdb; ipdb.set_trace()

            # with smooth for all
            self.idf[constants.UNK_ID] = 0  # unnecessary, just to be sure
            div = torch.div(self.nb_documents, self.idf + 1)
            self.idf = torch.log(div).unsqueeze(0) + 1

            # normalize idf to be between 0 and 1
            self.idf = self.idf / (1 + math.log(self.nb_documents))
            self.idf = self.idf.squeeze().detach()

    def build_loss(self, loss_weights=None):
        self._loss = None

    def bag_of_probas(self, words, probas, normalize=False):
        bs, ts = words.shape
        mask = torch.ne(words, constants.PAD_ID)
        bids = torch.arange(bs).unsqueeze(-1).expand(-1, ts).flatten()
        bids = bids.to(words.device)
        idxs = torch.stack((bids, words.flatten()), dim=0)
        vals = mask.int().to(words.device).float() * probas
        vals = vals.flatten()
        size = torch.Size([bs, self.words_vocab_size])
        bow = torch.sparse.FloatTensor(idxs, vals, size).to_dense()
        bow = bow.to(words.device)
        # normalize words by the number of their occurrence within the sentence
        if normalize:
            for i in range(bs):
                z = words[i].bincount(minlength=self.words_vocab_size).float()
                z[z == 0] = float('inf')
                bow[i] = bow[i] / z.to(words.device)
        # bow = bow * self.idf
        return bow

    @staticmethod
    def extract_hidden(classifier):
        hidden = classifier.hidden
        rnn_type = classifier.rnn_type
        is_bidir = classifier.is_bidir
        if rnn_type == 'lstm':
            new_hidden = hidden[0].detach()
        else:
            new_hidden = hidden.detach()
        if is_bidir:
            new_hidden = [new_hidden[0], new_hidden[1]]
        else:
            new_hidden = [new_hidden[0]]
        return torch.cat(new_hidden, dim=-1).unsqueeze(1)

    def get_clf_pred_and_hidden(self, batch, classifier, p=0):
        if self.explainer_cheat_ratio > 1:
            p = 1
        batch_size = batch.words.shape[0]
        clf_hidden = None
        clf_pred_classes = None
        do_batota = bool(torch.rand(1).item() < self.explainer_cheat_ratio)
        do_p = bool(torch.rand(1).item() < p)
        # do_p = bool(p > 0.8)
        do_batota_h = bool(torch.rand(1).item() < self.explainer_cheat_ratio)
        # do_p_h = bool(torch.rand(1).item() < p)

        # cheat explainer:
        # get pred_probas and hidden states from the classifier
        if not self.training or do_batota or do_batota_h:
            print('\t\t\tbatota', end='\r')

            # do cheat if do_batota is true or if we are at test time
            if not self.training or (do_batota and do_p):
                clf_hidden = self.extract_hidden(classifier)

                clf_pred_classes = classifier.predict_classes(batch).detach()

                # convert pred classes to one hot vectors
                one_hot = torch.zeros(batch_size, self.nb_classes,
                                      device=batch.words.device)
                clf_pred_classes = one_hot.scatter(1, clf_pred_classes, 1)
                clf_pred_classes = clf_pred_classes.unsqueeze(1)

        # blind explainer:
        # create zeros tensors as pred_classes and hidden states
        if clf_hidden is None:
            n = 2 if classifier.is_bidir else 1
            hdim = n * classifier.rnn.hidden_size
            clf_hidden = torch.zeros(batch_size, 1, hdim,
                                     device=batch.words.device)

        if clf_pred_classes is None:
            print('\t\t\tno batota', end='\r')
            clf_pred_classes = torch.zeros(batch_size, 1, self.nb_classes,
                                           device=batch.words.device)

        return clf_pred_classes, clf_hidden

    def get_clf_pred_and_hidden_inside(self, batch, classifier, p=0):
        if self.explainer_cheat_ratio > 1:
            p = 1
        batch_size = batch.words.shape[0]
        do_batota = torch.rand(batch_size) < self.explainer_cheat_ratio
        do_p = torch.rand(batch_size) < p
        do_batota_h = torch.rand(batch_size) < self.explainer_cheat_ratio

        clf_hidden = self.extract_hidden(classifier)
        clf_pred_classes = classifier.predict_classes(batch).detach()
        one_hot = torch.zeros(batch_size, self.nb_classes,
                              device=batch.words.device)
        clf_pred_classes = one_hot.scatter(1, clf_pred_classes, 1)
        clf_pred_classes = clf_pred_classes.unsqueeze(1)

        # blind explainer:
        n = 2 if classifier.is_bidir else 1
        hdim = n * classifier.rnn.hidden_size
        b_clf_hidden = torch.zeros(batch_size, 1, hdim, device=batch.words.device)
        b_clf_pred_classes = torch.zeros(batch_size, 1, self.nb_classes, device=batch.words.device)

        m = torch.tensor(not self.training) | (do_batota & do_p)
        m = m.float().unsqueeze(-1).unsqueeze(-1).to(batch.words.device)
        clf_pred_classes = m * clf_pred_classes + (1 - m) * b_clf_pred_classes
        # m = torch.tensor(not self.training) | (do_batota_h & do_p)
        # m = m.float().unsqueeze(-1).unsqueeze(-1).to(batch.words.device)
        clf_hidden = m * clf_hidden + (1 - m) * b_clf_hidden

        return clf_pred_classes, clf_hidden

    def forward(self, batch, classifier, p=0):
        bs, ts = batch.words.shape
        mask = torch.ne(batch.words, constants.PAD_ID)
        lengths = mask.int().sum(dim=-1)

        # get word embeddings
        explainer_embs = self.word_emb_explainer(batch.words)

        # apply idf
        if self.explainer_idf == 'embs':
            w_idf = self.idf[batch.words].unsqueeze(-1)
            explainer_embs = explainer_embs * w_idf

        # forward and backward hidden states
        lstm_vecs = pack(
            explainer_embs, lengths, batch_first=True, enforce_sorted=False)
        lstm_vecs, lstm_hidden = self.lstm_explainer(lstm_vecs)
        lstm_vecs, _ = unpack(lstm_vecs, batch_first=True)

        # recover the classifier predictions and its hidden states
        # clf_pred_classes, clf_hidden = self.get_clf_pred_and_hidden(
        #     batch, classifier, p=p
        # )
        clf_pred_classes, clf_hidden = self.get_clf_pred_and_hidden_inside(
            batch, classifier, p=p
        )

        # concat the lstm hidden states of the hypothesis and create a time dim
        hidden_states = [lstm_hidden[0][0], lstm_hidden[0][1]]
        lstm_hidden = torch.cat(hidden_states, dim=-1).unsqueeze(1)
        self.lstm_hidden = lstm_hidden
        self.lstm_out = lstm_vecs

        # concat clf output and hidden reps with lstm hidden
        # lstm_hidden = torch.cat(
        #   (lstm_hidden, clf_hidden, clf_pred_classes), dim=-1
        # )
        lstm_hidden = torch.cat((lstm_hidden, clf_pred_classes), dim=-1)

        # attention over explainer embs
        # message_emb, attn_weights = self.attn(lstm_hidden,
        #                                       explainer_embs,
        #                                       values=explainer_embs,
        #                                       mask=mask)

        # set score weights as idfs
        if self.explainer_idf == 'scores':
            s_weights = self.idf[batch.words].unsqueeze(-1)
        else:
            s_weights = None

        # attention over lstm vecs
        message_emb, attn_weights = self.attn(lstm_hidden,
                                              lstm_vecs,
                                              values=lstm_vecs,
                                              mask=mask,
                                              s_weights=s_weights)
        # self.lstm_hidden = message_emb

        # (bs, 1, ts) -> (bs, ts)
        attn_weights = attn_weights.squeeze()
        self.attn_weights = attn_weights

        # try to use the straight-through gumbel softmax
        # argmaxes = torch.nn.functional.gumbel_softmax(
        #     torch.log(attn_weights), tau=0.8, hard=True
        # )
        # top_word_ids = batch.words[argmaxes.int().bool()]

        # create message
        # use bag of probas during training
        if self.training:  # training time
            # k = min(self.explainer_attn_top_k, attn_weights.shape[-1])
            # top_probas, top_idxs = torch.topk(attn_weights, k, dim=-1)
            # top_word_ids = batch.words.gather(1, top_idxs)
            # probas = torch.zeros_like(attn_weights)
            # probas.scatter_(1, top_idxs, top_probas)
            # message = self.bag_of_probas(batch.words, probas, normalize=True)
            # message = message_emb.squeeze()
            # message = torch.sum(explainer_embs * attn_weights.unsqueeze(-1), 1)
            message = self.bag_of_probas(batch.words, attn_weights,
                                         normalize=False)

        # bag of words during test
        else:  # test time
            k = min(self.explainer_attn_top_k, attn_weights.shape[-1])
            top_probas, top_idxs = torch.topk(attn_weights, k, dim=-1)
            top_word_ids = batch.words.gather(1, top_idxs)

            # this is not a part of the computation graph, it is just for saving
            # the valid top word ids in case we need to access them later:
            self.valid_top_word_ids = filter_word_ids_with_non_zero_probability(
                top_word_ids, top_probas, pad_id=constants.PAD_ID
            )

            message = self.bag_of_probas(top_word_ids, torch.ceil(top_probas),
                                         normalize=False)
            # top_probas = top_probas / top_probas.sum(1).unsqueeze(-1)
            # message = self.bag_of_probas(top_word_ids, top_probas,
            #                              normalize=True)
            message = message / message.sum(-1).unsqueeze(-1)

        # create a time dimension of size 1
        message = message.unsqueeze(1)

        return message, message_emb

    def get_second_loss(self, batch, message, message_emb, classifier=None, kind='ae'):
        if kind == 'ae':
            x_gold = self.bag_of_probas(batch.words, 1.0)
            x_gold = x_gold / x_gold.sum(-1).unsqueeze(-1)
            x_tilde = self.rec_layer_ae_2(torch.tanh(self.rec_layer_ae(message)))
            x_tilde = torch.log_softmax(x_tilde, dim=-1)
            return torch.nn.functional.kl_div(
                x_tilde.squeeze(), x_gold.squeeze(), reduction='batchmean'
            )

        elif kind == 'pred':
            hid_gold = classifier.lstm_out.detach().mean(1)
            hid_pred = self.lstm_out.mean(1)
            hid_pred = self.rec_layer_pred(hid_pred)
            # hid_gold = self.extract_hidden(classifier)
            # hid_pred = torch.tanh(self.rec_layer_pred(message_emb))
            return torch.nn.functional.mse_loss(
                hid_pred.squeeze(), hid_gold.squeeze()
            )

        elif kind == 'conicity':
            hid_gold = classifier.lstm_out.detach()
            hid_gold_avg_vec = hid_gold.mean(1).unsqueeze(1)
            hid_pred = self.lstm_out
            # hid_pred = self.rec_layer_pred(hid_pred)
            hid_pred_avg_vec = hid_pred.mean(1).unsqueeze(1)

            gold_cos = torch.nn.functional.cosine_similarity(hid_gold_avg_vec, hid_gold, dim=-1)
            pred_cos = torch.nn.functional.cosine_similarity(hid_pred_avg_vec, hid_pred, dim=-1)
            return torch.pow(gold_cos - pred_cos, 2).sum(1).mean(0)

        elif kind == 'lstm':
            hid_gold = self.extract_hidden(classifier)
            hid_pred = self.lstm_hidden.detach()
            return torch.nn.functional.cosine_embedding_loss(
                hid_pred.squeeze(),
                hid_gold.squeeze(),
                torch.ones(hid_pred.shape[:2], device=hid_pred.device).float(),
                margin=0.0,
            )
            # other_loss = torch.nn.functional.pairwise_distance(
            #     hid_pred.squeeze(),
            #     hid_gold.squeeze(),
            #     p=2.0
            # ).mean()
            # other_loss = torch.nn.functional.mse_loss(
            #     hid_pred.squeeze(),
            #     hid_gold.squeeze()
            # )

        return torch.zeros(1)
