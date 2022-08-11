from . import BaseExplainer
from alibi.explainers import TreeShap

class TreeShapExplainer(BaseExplainer):

    def __init__(self):
        super().__init__()

    def _fit(
        self,
        model,
        X_train,
        y_train,
    ):
        self.explainer = TreeShap(model)
        self.explainer.fit(
            X_train,
            model_output='raw'
        )

    def _explain(
        self,
        X_test,
        y_test
    ):
        return self.explainer.explain(X_test)