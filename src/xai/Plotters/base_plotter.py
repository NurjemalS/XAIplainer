from ..Explainers import BaseExplainer

class BasePlotter:

    def get_plots(
            self,
            explainer: BaseExplainer,
            X_test,
            y_test
        ):
            raise Exception("'get_plots' must be overridden.")    