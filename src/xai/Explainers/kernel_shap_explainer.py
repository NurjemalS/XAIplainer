from . import BaseExplainer
from alibi.explainers import KernelShap

class KernelShapExplainer(BaseExplainer):

    def __init__(self):
        super().__init__()

    def _fit(
        self,
        model,
        X_train,
        y_train,
    ):
        self.explainer = KernelShap(model.predict)
        self.explainer.fit(
            X_train,
            summarise_background='auto' # X_train too big. This option allows model to "summarize"
        )

    def _explain(
        self,
        X_test,
        y_test
    ):
        return self.explainer.explain(X_test)