from . import BaseExplainer
from alibi.explainers import ALE

class AleExplainer(BaseExplainer):
    def __init__(self):
        super().__init__()

    def _fit(
            self,
            model,
            X_train,
            y_train
        ):
        self.explainer = ALE(model.predict, X_train.columns, [y_train.name])

    def _explain(
            self,
            X_test,
            y_test
        ):
        return self.explainer.explain(X_test.to_numpy()) # TODO: Verify that using X_test makes sense.