import torch

from spec import constants
from spec.explainers.explainer import Explainer
from spec.explainers.utils import filter_word_ids_with_non_zero_probability


class RandomAttentionExplainer(Explainer):

    def __init__(self, fields_tuples, options):
        super().__init__(fields_tuples)
        self.words_vocab_size = len(self.fields_dict['words'].vocab)
        self.explainer_attn_top_k = options.explainer_attn_top_k
        self.message_type = options.message_type
        # options.word_embeddings_size is updated in the classifier constructor
        # when a path to pretrained embeddings is passed
        self.emb_size = options.word_embeddings_size
        self.random_type = options.explainer_random_type
        self.valid_top_word_ids = None

    def build_loss(self, loss_weights=None):
        self._loss = None

    def forward(self, batch, classifier):
        # generate random attn_weights
        if self.random_type == 'beta':
            mask = torch.ne(batch.words, constants.PAD_ID)
            beta = torch.distributions.beta.Beta(5.0, 5.0)
            attn_weights = beta.sample(batch.words.shape)
            attn_weights = attn_weights.squeeze(-1).to(batch.words.device)
            attn_weights[mask == 0] = 0

        elif self.random_type == 'uniform':
            mask = torch.ne(batch.words, constants.PAD_ID)
            attn_weights = torch.rand(batch.words.shape).to(batch.words.device)
            attn_weights = attn_weights / attn_weights.sum(-1).unsqueeze(-1)
            attn_weights[mask == 0] = 0

        elif self.random_type == 'zero_max_out':
            _ = classifier(batch)
            attn_weights = classifier.attn_weights.squeeze()
            arange = torch.arange(attn_weights.shape[0]).to(attn_weights.device)
            # maybe we can try zero out k max?
            _, max_idxs = torch.topk(attn_weights, k=1, dim=-1)
            attn_weights[arange, max_idxs.squeeze()] = 0

        elif self.random_type == 'first_states':
            mask = torch.ne(batch.words, constants.PAD_ID)
            _ = classifier(batch)
            bs, ts = batch.words.shape
            attn_weights = torch.arange(ts, 0, -1).repeat(bs, 1).float()
            attn_weights = attn_weights.to(batch.words.device)
            attn_weights = attn_weights / ts
            attn_weights[mask == 0] = 0

        elif self.random_type == 'last_states':
            mask = torch.ne(batch.words, constants.PAD_ID)
            _ = classifier(batch)
            bs, ts = batch.words.shape
            attn_weights = torch.arange(1, ts + 1).repeat(bs, 1).float()
            attn_weights = attn_weights.to(batch.words.device)
            attn_weights = attn_weights / ts
            attn_weights[mask == 0] = 0

        elif self.random_type == 'mid_states':
            mask = torch.ne(batch.words, constants.PAD_ID)
            lengths = mask.int().sum(-1).tolist()
            bs, ts = batch.words.shape
            attn_weights = torch.zeros(bs, ts).to(batch.words.device)
            for i, ell in enumerate(lengths):
                attn_weight_left = torch.arange(1, ell // 2 + 1)
                attn_weight_right = torch.arange(ell // 2, 0, -1)
                w = [attn_weight_left]
                if ell % 2 != 0:
                    attn_weight_mid = torch.tensor([(ell + 1) // 2])
                    w.append(attn_weight_mid)
                w.append(attn_weight_right)
                concat_tensors = torch.cat(w).to(attn_weights.device)
                attn_weights[i, :ell] = concat_tensors
            attn_weights = attn_weights.float()

        else:  # shuffle
            _ = classifier(batch)
            attn_weights = classifier.attn_weights.squeeze()
            mask = torch.ne(batch.words, constants.PAD_ID)
            lengths = mask.int().sum(-1).tolist()
            for i in range(attn_weights.shape[0]):
                valid_random_idx = torch.arange(attn_weights.shape[1])
                idx = torch.randperm(lengths[i])
                valid_random_idx[:lengths[i]] = idx
                attn_weights[i] = attn_weights[i, valid_random_idx]

        # find the topk attn weights using 1 < k < seq_len
        k = min(self.explainer_attn_top_k, attn_weights.shape[-1])
        top_probas, top_idxs = torch.topk(attn_weights, k, dim=-1)

        # recover the word ids from the top indexes
        top_word_ids = batch.words.gather(1, top_idxs)

        # what to do when top ids map to pad ids? or when
        # it returns instances zeroed out by sparsity?
        # for now, hard coded in pure python: filter out these entries
        valid_top_word_ids = filter_word_ids_with_non_zero_probability(
            top_word_ids, top_probas, pad_id=constants.PAD_ID
        )
        # save for getting the words later
        self.valid_top_word_ids = valid_top_word_ids

        # create the message
        message = self.make_message(
            valid_top_word_ids, top_probas, classifier.word_emb
        )

        # create a time dimension of size 1
        message = message.unsqueeze(1)

        return message
