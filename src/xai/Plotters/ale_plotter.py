from ..Explainers import BaseExplainer
from . import BasePlotter
from alibi.explainers import plot_ale

class AlePlotter(BasePlotter):

    def __ale_plot(self, ale_explanation):
        plot_ale(
            ale_explanation
            #fig_kw={'figwidth':10, 'figheight': 10}
        )

    def get_plots(
            self,
            explainer: BaseExplainer,
            X_test,
            y_test
        ):

        ale_explanation = explainer.explain(X_test, y_test)
        return [
            {
                "header": "ALE Plot",
                "plotter": lambda: self.__ale_plot(ale_explanation),
                "summary": "ALE plot summary." #TODO: add summary
            }
        ]