from ..Explainers import BaseExplainer
from . import BasePlotter
import shap

class ShapPlotter(BasePlotter):

    def __local_force_plot(
            self,
            explainer,
            explanation,
            X_test,
            index=2
        ):

        instance = X_test.iloc[index, :]

        shap.force_plot(
            explainer.explainer.expected_value,
            explanation.shap_values[0][index, :],
            instance,
            X_test.columns,
            matplotlib = True
        )

    def __multioutput_decision_plot(
            self,
            explainer,
            explanation,
            X_test,
            index=2
        ):

        shap.multioutput_decision_plot(
            [explainer.explainer.expected_value],
            explanation.shap_values,
            index,
            feature_names=list(X_test.columns),
            feature_order='importance',
            return_objects=True
        )

    def __mean_summary_plot(self, explanation, X_test):

        shap.summary_plot(
            shap_values = explanation.shap_values,
            feature_names = list(X_test.columns),
            features = X_test
        )

    def __summary_plot(self, explanation, X_test):
        shap.summary_plot(
            shap_values = explanation.shap_values[0],
            feature_names = list(X_test.columns),
            features = X_test
        )

    def __global_force_plot(
            self,
            explainer,
            explanation,
            X_test
        ):
        """
        NOT SUPPORTED.
        
        Shap can not draw force plots with multiple samples using
        Matplotlib.
        """

        return shap.force_plot(
            base_value = explainer.expected_value,
            shap_values = explanation.shap_values[0],
            feature_names = list(X_test.columns),
            features = X_test[:50],
            matplotlib = True
        )


    def get_plots(
            self,
            explainer: BaseExplainer,
            X_test,
            y_test
        ):

        explanation = explainer.explain(X_test, y_test)
        return [
            {
                "header": "Local Force Plot",
                "plotter": lambda: self.__local_force_plot(explainer, explanation, X_test),
                "summary": "Local Force Plot summary." #TODO: add summary
            },
            {
                "header": "Multioutput Decision Plot",
                "plotter": lambda: self.__multioutput_decision_plot(explainer, explanation, X_test),
                "summary": "Local Multioutput Decision summary." #TODO: add summary
            },
            {
                "header": "Mean Summary Plot",
                "plotter": lambda: self.__mean_summary_plot(explanation, X_test),
                "summary": "Global Mean Summary Plot summary." #TODO: add summary
            },
            {
                "header": "Summary Plot",
                "plotter": lambda: self.__summary_plot(explanation, X_test),
                "summary": "Global Summary Plot summary." #TODO: add summary
            }
        ]