import streamlit as st
import pandas as pd
from datetime import datetime
from xai import get_available_explainers_for_model, load_explainer_and_plotter

def app():
    st.header("Reports")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if "model"  not in st.session_state:
           st.write("first you need to predict")
    else:
        model = st.session_state["model"]

        X_test, y_test = st.session_state["X_test"], st.session_state["y_test"]
        list_of_explainers = []

        for explainer_name in get_available_explainers_for_model(model.__class__.__name__):
            st.write("---")
            start_for_each_explainer=datetime.now()
            st.subheader(explainer_name)
            explainer, plotter = load_explainer_and_plotter(explainer_name)
            explainer.fit(model, X_test, y_test)
            plots = plotter.get_plots(explainer, X_test, y_test)

            time_for_explanation = (datetime.now()-start_for_each_explainer).total_seconds()
            list_of_explainers.append({
                "Name" : explainer_name,
                "Time (in seconds)" : time_for_explanation
            })
            for plot in plots:
                st.write(plot["header"])
                plot["plotter"]()
                st.pyplot(bbox_inches='tight', dpi=300,pad_inches=0)

        explainers= ""
        time_for_total_explanation = 0
        for e in list_of_explainers:
            explainers = explainers + "  " + e["Name"]
            time_for_total_explanation +=  e["Time (in seconds)"]

        st.write("---")
        st.write("---")

        st.subheader("Statistics: ")
        benchmark_total= {
            "Model" : [model.__class__.__name__],
            "Explainer names": explainers, 
            "Total Time for Explanation (in seconds)": [time_for_total_explanation],
            "Total Time for Prediction (in seconds)": [st.session_state["total_time_for_prediction"]]
        }

        for error in st.session_state["errors"]:
            benchmark_total[error["error_name"]] = [error["error_value"]]

        df_total = pd.DataFrame.from_dict(benchmark_total)
        df_explainers = pd.DataFrame.from_dict(list_of_explainers)
        st.table(df_total)
        st.write("Benchmarking for Explainers:")
        st.table(df_explainers)

app()
