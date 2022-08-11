from typing import Tuple, List
import streamlit as st
import pandas as pd
import pickle
import os
import tempfile
from datetime import datetime
from sklearn.model_selection import train_test_split

from utils import list_datasets, read_data, render_widget_options
from xai import get_model_names, load_model, get_model_widget_options, config, load_class

import mlflow

def pick_model_section() -> str:

    st.header("Select Model")
    model_name = st.selectbox("Choose the model", options = get_model_names(), index = 0 )
    return model_name

def pick_common_model_args_section(
        model_name: str
    ) -> dict:
    
    st.subheader("Model Configuration Part 1: Common parameters")
    widget_options = get_model_widget_options(model_name)
    if len(widget_options) == 0:
        st.write("""No common parameters available for this model.
        Update 'widget_options' attribute of the model in the config
        file in order to add common parameters to the model.""")
        return {}
    else:
        return render_widget_options(widget_options)

def pick_extra_model_args_section() -> dict:

    st.subheader("Model Configuration Part 2: Extra parameters")
    st.write("Below, you can enter paramaters in addition to the common ones present above.")

    extra_widget_options = st.session_state.get("extra_widget_options", [])

    param_name_col, param_value_col, param_dtype_col = st.columns(3)

    param_name = param_name_col.text_input('Name of the parameter')
    param_dtype = param_dtype_col.selectbox("Type of the parameter", options=["text", "number"])
    if param_dtype == "text":
        param_value = param_value_col.text_input('Value of the parameter')
    else:
        param_value = param_value_col.number_input('Value of the parameter')

    # Add extra parameter button

    add = st.button('ADD')

    if add:
        if len(param_name) != 0:

            if param_dtype == "text" and len(param_value) == 0:
                st.error("Enter the argument value properly")
            else:
                option = {
                        "label": param_name,
                        "name": param_name,
                        "value": param_value,
                        "is_text_input": True,
                        "data_type": str
                    }
                
                if param_dtype != "text":
                    option["data_type"] = float
                
                extra_widget_options.append(option)
                st.session_state["extra_widget_options"] = extra_widget_options
        else:
            st.error("Enter the argument name properly")

    # Remove added extra parameter selectbox

    extra_parameter_list = ["None"]
    extra_parameter_list.extend([option["name"] for option in extra_widget_options])

    st.write("Using the select box below, you can remove an extra parameter you have added.")
    remove = st.selectbox("Remove Parameter", options = extra_parameter_list, index = 0)
    if remove != "None":
        extra_widget_options = list(filter(lambda option: option["name"] != remove, extra_widget_options))
        st.session_state["extra_widget_options"] = extra_widget_options

    if len(extra_widget_options) == 0:
        st.info("There are no extra parameters added.")
        return {}
    else:
        st.write("Below, you can view and edit the extra parameters added.")
        return render_widget_options(extra_widget_options)

def model_args_preview_section(model_args: dict):

    st.subheader("Model Configuration Preview")
    st.write(model_args)

def pick_dataset_section() -> pd.DataFrame:

    # Set the session_state to the initial state for dataset selection
    if "dataset_selected" not in st.session_state:
        st.session_state["dataset_selected"] = None
    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = type(None)

    st.header("Choose Data")
    st.markdown("""
    Below, you can see two ways of selecting data:
    - To the left, you can see a list of datasets available in the app. 
    - To the right, you can see a way of uploading your own dataset.
    
    Keep in mind that uploading has the priority. If you upload a file, model will be trained with the uploaded file. If you want to get rid of the uploaded file, simply click the cross next to the uploaded file.""")
    sel_col, upload_col = st.columns(2)
    dataset = list_datasets()
    dataset_selected = sel_col.selectbox("Datasets", options = dataset, index = 0)
    uploaded_file = upload_col.file_uploader("Upload your dataset", type=["csv"])

    if uploaded_file:
        # file uploaded. Use it.
        df = pd.read_csv(uploaded_file)
    else:
        # no file uploaded, use the selected dataset
        df = read_data(dataset_selected)
    
    st.write("Using Data: ", dataset_selected)
    return df

def data_configuration_section(
        df: pd.DataFrame
    ) -> Tuple[str, List[str], float]:

    target_feature = st.selectbox("Target Feature", options = df.columns, index = 0)

    features = list(df.columns)
    features.remove(target_feature)
    selected_features = st.multiselect(label = "select features", options = features)

    size = st.slider("Select split size", min_value = 0.10, max_value= 0.90, value = 0.20, step = 0.10 )

    return target_feature, selected_features, size

def predict_section(
        model_name: str,
        model_args: dict,
        df: pd.DataFrame,
        selected_features: list,
        target_feature: str,
        size: float
    ):

    if st.button("PREDICT"):
        if len(selected_features) == 0:
            st.error("Select features")
        else:
            model = load_model(model_name, model_args)

            X = pd.DataFrame(df, columns = selected_features)
            X_train, X_test, y_train, y_test = train_test_split(X, df[target_feature], test_size = size)

            model.fit(X_train, y_train)

            #st.write(y_test)
            start=datetime.now()
            predictions = model.predict(X_test)
            st.subheader("Predictions")
            st.write(predictions)
            total_time_for_prediction = (datetime.now()-start).total_seconds()
            st.session_state["total_time_for_prediction"] = total_time_for_prediction


            # create artifact

            artifact_dictionary = {
                "model": model,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test
            }

            artifact = pickle.dumps(artifact_dictionary)
            st.download_button("Download Artifact Pickle", data=artifact, file_name="xai-artifact.pkl")

            # log to mlflow
            errors = []
            with mlflow.start_run():

                with tempfile.TemporaryDirectory() as tmp:
                    artifact_path = os.path.join(tmp, "run_artifact.pkl")
                    with open(artifact_path, "wb") as f:
                        pickle.dump(artifact_dictionary, f)                        
                    mlflow.log_artifact(artifact_path)

                mlflow.sklearn.log_model(model, f"{model.__class__.__name__}_model")
                mlflow.log_params(model_args)

                y_train_pred = model.predict(X_train)
                y_test_pred  = model.predict(X_test)                
                for metric in config["sklearn_metrics"]:
                    scorer = load_class("sklearn.metrics", metric)

                    train_error = scorer(y_train, y_train_pred)
                    test_error  = scorer(y_test,  y_test_pred)
                    errors.append({
                        "error_name" : metric,
                        "error_value" : test_error
                    })

                    mlflow.log_metric(f"{metric}_train", train_error)
                    mlflow.log_metric(f"{metric}_test",  test_error)

                # save session state
                st.session_state["errors"] = errors
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test
                st.session_state["model"] = model

def app():

    model_name = pick_model_section()
    common_model_args = pick_common_model_args_section(model_name)
    extra_model_args  = pick_extra_model_args_section()

    model_args = {}
    model_args.update(common_model_args)
    model_args.update(extra_model_args)

    model_args_preview_section(model_args)

    st.write("---")

    df = pick_dataset_section()
    target_feature, selected_features, size = data_configuration_section(df)

    predict_section(model_name,
                    model_args,
                    df,
                    selected_features,
                    target_feature, 
                    size)

app()