import streamlit as st

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Ingestion.data_loader import DataLoader
from analysis.dataset_analyzer import DatasetAnalyzer


st.set_page_config(
    page_title="AutoML Pipeline",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AutoML Pipeline")

st.write("Upload a CSV file to analyze the dataset.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        loader = DataLoader()

        df = loader.load_data(uploaded_file)

        analyzer = DatasetAnalyzer(df)

        report = analyzer.generate_report()

        st.success("Dataset Loaded Successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(loader.preview_data(df))

        st.subheader("Dataset Shape")

        rows, cols = report["Shape"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", rows)

        with col2:
            st.metric("Columns", cols)

        st.subheader("Column Names")
        st.write(report["Columns_name"])

        st.subheader("Column Types")
        st.write(report["Columns_type"])

        st.subheader("Missing Values")
        st.dataframe(report["Missing_values"])

        st.subheader("Duplicate Count")
        st.write(report["Duplicate_count"])

        st.subheader("Numerical Columns")
        st.write(report["Numerical_columns"])

        st.subheader("Categorical Columns")
        st.write(report["Categorical_columns"])

    except Exception as e:
        st.error(f"Error: {e}")