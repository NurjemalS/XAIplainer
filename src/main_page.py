import streamlit as st

st.title("XAIplainer")

st.markdown("""
You can use the sidebar to navigate the website.

In order to load a model to the app, you can do one of the following:
- Go to 'train predict' page and train a model
- Go to 'artifact upload' page and uplaod an artifact generated in 'train predict'

Once the model is uploaded, you can go to 'reports' page and view the reports.
""")