import pickle
import streamlit as st
import pandas as pd

def app():
    st.header("Upload Artifact")

    st.write("On this page, you can upload an artifact generated with this website to generate plots.")

    file = st.file_uploader("Upload Artifact", type=["pkl"])

    if file:
        artifact = pickle.load(file)

        model = artifact["model"]

        st.write("Go to the reports page to view the reports generated with the artifact you uploaded.")
        st.header("Artifact Information")
        st.write("Model:", model.__class__.__name__)

        st.subheader("Set Shapes")
        l = ["X_train", "X_test", "y_train", "y_test"]
        st.dataframe({"Set": l, "Shape": [str(artifact[i].shape) for i in l]})

        st.subheader("Training Set")        
        st.write(pd.concat([artifact["X_train"], artifact["y_train"]], axis=1))
        st.write("Target column:", artifact["y_train"].name)

        # Save artifacts to the session state
        st.session_state["X_test"] = artifact["X_test"]
        st.session_state["y_test"] = artifact["y_test"]
        st.session_state["model"] = model

app()