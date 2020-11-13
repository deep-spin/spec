import torch

from spec import constants
from spec.explainers.explainer import Explainer


class TranslationEncodedAttentionExplainer(Explainer):

    def __init__(self, fields_tuples, options):
        super().__init__(fields_tuples)
        self.words_vocab_size = len(self.fields_dict['words'].vocab)
        self.explainer_attn_top_k = options.explainer_attn_top_k
        self.message_type = options.message_type
        self.emb_size = 0
        self.valid_top_word_ids = None
        self.is_full = bool('154' in options.output_dir)

        if self.message_type == 'embs':
            emb_weight = self.fields_dict['words'].vocab.vectors
            self.emb_size = emb_weight.shape[-1]
            self.word_emb = torch.nn.Embedding.from_pretrained(
                emb_weight, freeze=True, padding_idx=constants.PAD_ID
            )

    def get_output_size(self):
        if self.message_type == 'bow':
            return self.words_vocab_size
        else:
            return self.explainer_attn_top_k * self.emb_size

    def build_loss(self, loss_weights=None):
        """This is not a trainable module, so it does not have a loss."""
        self._loss = None

    def forward(self, batch, classifier):
        # recover attn_weights
        # clf_attn_weights = classifier.attn_weights.squeeze()
        bs = batch.words.shape[0]
        src_len = batch.words.shape[-1]
        hyp_len = batch.words_hyp.shape[-1]
        clf_attn_weights = batch.attn[:, :src_len, :hyp_len]
        expanded_words = batch.words.unsqueeze(1).expand(bs, hyp_len, src_len)

        # (bs, source, target) -> (bs, target, source)
        clf_attn_weights = clf_attn_weights.transpose(1, 2)

        # random:
        # device = batch.words.device
        # clf_attn_weights = torch.rand(bs, hyp_len, src_len).to(device)
        # mask = torch.ne(expanded_words, constants.PAD_ID).to(device)
        # clf_attn_weights = clf_attn_weights * mask.float()
        # clf_attn_weights /= clf_attn_weights.sum(-1).unsqueeze(-1)

        # find the topk attn weights using 1 < k < seq_len
        k = min(self.explainer_attn_top_k, clf_attn_weights.shape[-1])
        top_probas, top_idxs = torch.topk(clf_attn_weights, k, dim=-1)

        # recover the word ids from the top indexes
        top_word_ids = expanded_words.gather(2, top_idxs)

        # save for getting the words later
        self.valid_top_word_ids = top_word_ids.tolist()

        if self.message_type == 'bow':
            if k == 0:
                return torch.zeros(bs, hyp_len, self.words_vocab_size,
                                   device=batch.words.device)

            # create a masked bag of words
            flat_top_word_ids = top_word_ids.view(bs * hyp_len, k)
            mask = torch.ne(flat_top_word_ids, constants.PAD_ID)
            bids = torch.arange(bs*hyp_len).to(flat_top_word_ids.device)
            bids = bids.unsqueeze(-1).expand(-1, k).flatten()
            idxs = torch.stack((bids, flat_top_word_ids.flatten()), dim=0)
            vals = mask.float().to(flat_top_word_ids.device)
            valid = (top_probas.view(bs * hyp_len, k) > 0).float()
            vals = vals * valid
            vals = vals.flatten()
            size = torch.Size([bs*hyp_len, self.words_vocab_size])
            bow = torch.sparse.FloatTensor(idxs, vals, size).to_dense()
            bow = bow.to(flat_top_word_ids.device)
            bow = bow.view(bs, hyp_len, self.words_vocab_size)
            # bow = torch.zeros_like(clf_attn_weights)
            message = bow
        else:

            if self.is_full:
                top_word_ids_pad = top_word_ids
                if k < self.explainer_attn_top_k:
                    top_word_ids_pad = torch.zeros(bs,
                                                   hyp_len,
                                                   self.explainer_attn_top_k,
                                                   device=batch.words.device).long()
                    top_word_ids_pad += constants.PAD_ID
                    for i, t1 in enumerate(top_word_ids):
                        for j, t2 in enumerate(t1):
                            top_word_ids_pad[i, j, :t2.shape[-1]] = t2

            else:
                top_word_ids_pad = top_word_ids
                v = (top_probas > 0).long()
                top_word_ids_pad = v * top_word_ids_pad + (1 - v) * constants.PAD_ID
                top_word_ids_pad = top_word_ids_pad.long()

                if not self.training:
                    self.valid_top_word_ids = []
                    for x in top_word_ids_pad.tolist():
                        self.valid_top_word_ids.append(
                            [w for w in x if w != constants.PAD_ID]
                        )

                if k < self.explainer_attn_top_k:
                    top_word_ids_pad = torch.zeros(bs,
                                                   hyp_len,
                                                   self.explainer_attn_top_k,
                                                   device=batch.words.device).long()
                    top_word_ids_pad += constants.PAD_ID
                    for i, t1 in enumerate(top_word_ids):
                        for j, t2 in enumerate(t1):
                            top_word_ids_pad[i, j, :t2.shape[-1]] = t2

            # recover glove embeddings for each top word:
            message = self.word_emb(top_word_ids_pad)
            # concat embeddings:
            message = message.view(bs, hyp_len, -1)

        return message


