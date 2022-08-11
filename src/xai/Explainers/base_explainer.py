
class BaseExplainer:
    """
    Base Explainer class is subclassed by
    specific explainer methods.
    """
    def __init__(self):
        self.explainer_fit = False

    def fit(self, model, X_train, y_train):
        self._fit(model, X_train, y_train)
        self.explainer_fit = True

    def explain(self, X_test, y_test):
        if not self.explainer_fit:
            raise Exception("Expaliner not fit. Use 'fit' method before calling 'explain'.")
        return self._explain(X_test, y_test)

    def _explain(self):
        raise Exception("'_explain' method should be overridden.")

    def _fit(self):
        raise Exception("'_fit' method should be overridden.")
        