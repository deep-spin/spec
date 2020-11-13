from spec.explainers.explainer import Explainer


class PrototypeExplainer(Explainer):

    def __init__(self, fields_tuples, options):
        super().__init__(fields_tuples)
        pass

    def build_loss(self, loss_weights=None):
        self._loss = None

    def forward(self, batch, classifier):
        pass
