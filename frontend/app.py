import streamlit as st
import os
import sys

# =====================================================
# PATH SETUP
# =====================================================

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(project_root)

# =====================================================
# IMPORTS
# =====================================================

from Ingestion.data_loader import DataLoader
from analysis.dataset_analyzer import DatasetAnalyzer
from preprocessing.preprocessor import Preprocessor

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ModelForge",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ModelForge")
st.caption("Build Machine Learning Pipelines Faster")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# =====================================================
# MAIN APP
# =====================================================

if uploaded_file is not None:

    try:

        loader = DataLoader()

        df = loader.load_data(uploaded_file)

        analyzer = DatasetAnalyzer(df)

        report = analyzer.generate_report()

        st.success("Dataset Loaded Successfully!")

        # =================================================
        # TABS
        # =================================================

        tab1, tab2 = st.tabs([
            "📊 Dataset Analysis",
            "🛠️ Preprocessing"
        ])

        # =================================================
        # DATASET ANALYSIS TAB
        # =================================================

        with tab1:

            st.subheader("Dataset Preview")

            st.dataframe(
                loader.preview_data(df),
                use_container_width=True
            )

            rows, cols = report["Shape"]

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Rows", rows)

            with c2:
                st.metric("Columns", cols)

            with c3:
                st.metric(
                    "Duplicates",
                    report["Duplicate_count"]
                )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Column Names")
                st.write(report["Columns_name"])

                st.subheader("Numerical Columns")
                st.write(report["Numerical_columns"])

            with col2:

                st.subheader("Column Types")
                st.write(report["Columns_type"])

                st.subheader("Categorical Columns")
                st.write(report["Categorical_columns"])

            st.subheader("Missing Values")

            st.dataframe(
                report["Missing_values"],
                use_container_width=True
            )

        # =================================================
        # PREPROCESSING TAB
        # =================================================

        with tab2:

            st.subheader("Preprocessing Configuration")

            # -----------------------------
            # Missing Value Strategy
            # -----------------------------

            numeric_strategy = st.selectbox(
                "Numerical Missing Value Strategy",
                ["mean", "median"]
            )

            # -----------------------------
            # Duplicate Strategy
            # -----------------------------

            duplicate_strategy = st.radio(
                "Duplicate Handling",
                ["keep", "remove"]
            )

            # -----------------------------
            # Encoding Section
            # -----------------------------

            st.subheader("Encoding Configuration")

            cat_cols = analyzer.categorical_columns()

            if len(cat_cols) > 0:

                selected_column = st.selectbox(
                    "Select Column For Encoding",
                    cat_cols
                )

                encoding_strategy = st.selectbox(
                    "Encoding Strategy",
                    [
                        "keep_Same",
                        "OneHotEncoder",
                        "LabelEncoder",
                        "OrdinalEncoder"
                    ]
                )

            else:

                st.info(
                    "No categorical columns found."
                )

                selected_column = None

                encoding_strategy = "keep_Same"

            st.info(
                "Categorical missing values are "
                "filled using Mode."
            )

            # -----------------------------
            # APPLY BUTTON
            # -----------------------------

            if st.button(
                "🚀 Apply Preprocessing",
                use_container_width=True
            ):

                preprocessor = Preprocessor(
                    df=df,
                    numeric_strategy=numeric_strategy,
                    cat_strategy="mode",
                    duplicate_strategy=duplicate_strategy,
                    encoding_strategy=encoding_strategy
                )

                # ==================================
                # Missing Values
                # ==================================

                clean_df, high_missing_columns = (
                    preprocessor.handle_missing_values()
                )

                # ==================================
                # Duplicates
                # ==================================

                preprocessor.df = clean_df

                clean_df = (
                    preprocessor.handle_duplicate()
                )

                # ==================================
                # Encoding
                # ==================================

                preprocessor.df = clean_df

                if selected_column:

                    clean_df = (
                        preprocessor
                        .encode_categorical_column(
                            selected_column
                        )
                    )

                # ==================================
                # OUTPUT
                # ==================================

                st.success(
                    "Preprocessing Completed Successfully!"
                )

                st.subheader(
                    "Processed Dataset Preview"
                )

                st.dataframe(
                    clean_df.head(),
                    use_container_width=True
                )

                st.subheader(
                    "Processed Dataset Shape"
                )

                r, c = clean_df.shape

                cc1, cc2 = st.columns(2)

                with cc1:
                    st.metric(
                        "Rows After Processing",
                        r
                    )

                with cc2:
                    st.metric(
                        "Columns After Processing",
                        c
                    )

                if high_missing_columns:

                    st.warning(
                        f"High Missing Columns Found: "
                        f"{high_missing_columns}"
                    )

                st.download_button(
                    label="📥 Download Processed Dataset",
                    data=clean_df.to_csv(index=False),
                    file_name="processed_dataset.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(f"Error: {e}")