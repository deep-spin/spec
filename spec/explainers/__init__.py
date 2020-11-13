from pathlib import Path

from spec import constants
from spec import opts
from spec.explainers.embedded_attention import EmbeddedAttentionExplainer
from spec.explainers.encoded_attention import EncodedAttentionExplainer
from spec.explainers.embedded_attention_supervised import \
    EmbeddedAttentionSupervisedExplainer
from spec.explainers.encoded_attention_translation import \
    TranslationEncodedAttentionExplainer
from spec.explainers.eraser_max_out import EraserMaxOutExplainer
from spec.explainers.gradient_magnitude import GradientMagnitudeExplainer
from spec.explainers.leave_one_out import LeaveOneOutExplainer
from spec.explainers.post_hoc import PostHocExplainer
from spec.explainers.post_hoc_entailment import PostHocEntailmentExplainer
from spec.explainers.prototype import PrototypeExplainer
from spec.explainers.random_attention import RandomAttentionExplainer
from spec.explainers.recursive_max_out import RecursiveMaxOutExplainer

available_explainers = {
    'encoded_attn': EncodedAttentionExplainer,
    'gradient_magnitude': GradientMagnitudeExplainer,
    'post_hoc': PostHocExplainer,
    'post_hoc_entailment': PostHocEntailmentExplainer,
    'prototype': PrototypeExplainer,
    'random_attn': RandomAttentionExplainer,
    'encoded_attn_translation': TranslationEncodedAttentionExplainer,
    'embedded_attn': EmbeddedAttentionExplainer,
    'embedded_attn_supervised': EmbeddedAttentionSupervisedExplainer,
    'recursive_max_out': RecursiveMaxOutExplainer,
    'eraser_max_out': EraserMaxOutExplainer,
    'leave_one_out': LeaveOneOutExplainer
}


def build(options, fields_tuples, loss_weights):
    explainer_cls = available_explainers[options.explainer]
    explainer = explainer_cls(fields_tuples, options)
    explainer.build_loss(loss_weights)
    if options.gpu_id is not None:
        explainer = explainer.cuda(options.gpu_id)
    return explainer


def load_state(path, explainer):
    explainer_path = Path(path, constants.EXPLAINER)
    explainer.load(explainer_path)


def load(path, fields_tuples, current_gpu_id):
    options = opts.load(path, name=constants.COMMUNICATION_CONFIG)

    # set gpu device to the current device
    options.gpu_id = current_gpu_id

    # hack: set dummy loss_weights (the correct values are going to be loaded)
    target_field = dict(fields_tuples)['target']
    loss_weights = None
    if options.loss_weights == 'balanced':
        loss_weights = [0] * (len(target_field.vocab) - 1)

    explainer = build(options, fields_tuples, loss_weights)
    load_state(path, explainer)
    return explainer


def save(path, explainer):
    explainer_path = Path(path, constants.EXPLAINER)
    explainer.save(explainer_path)
