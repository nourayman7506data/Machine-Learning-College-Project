import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats.mstats import winsorize

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, RocCurveDisplay,
    silhouette_score
)

# ---------------- 1. Data Upload ----------------
def upload_data():
    st.subheader("Upload Data")

    if st.session_state.data is not None:
        st.info(f"✅ Data already loaded. Shape: {st.session_state.data.shape}")
        if st.button("🔄 Upload New File"):
            st.session_state.data = None
            st.session_state.file_uploaded = False
            st.rerun()
        return

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        key="file_uploader_main"
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.data = df.copy()
        st.session_state.original_data = df.copy()

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.dataframe(df.head(10))
        st.rerun()

# ---------------- 2. Visualization ----------------
def data_visualization():
    st.subheader("Data Visualization")
    df = st.session_state.data
    st.write(f"Current data shape: {df.shape}")

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Choose X Axis", df.columns)
    with col2:
        y_axis = st.selectbox("Choose Y Axis", df.columns)

    plot_type = st.radio("Plot Type", ["Scatter Plot", "Line Plot", "Box Plot"])

    if st.button("Apply Plot"):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))

        if plot_type == "Scatter Plot":
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        elif plot_type == "Line Plot":
            sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
        else:
            sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)

        st.pyplot(fig)

# ---------------- 3. Missing Values ----------------
def handle_missing_values():
    st.subheader("Missing Values Handling")
    df = st.session_state.data

    st.write("### Missing Values Count (Before):")
    missing_before = df.isnull().sum()
    st.dataframe(
        missing_before[missing_before > 0]
        if missing_before.sum() > 0
        else pd.DataFrame({"No missing values": [0]})
    )

    selected_columns = st.multiselect(
        "Select Columns to handle",
        df.columns[df.isnull().any()].tolist()
    )

    if len(selected_columns) > 0:
        methods = {}
        strategies = {}
        k_values = {}

        for col in selected_columns:
            methods[col] = st.selectbox(
                f"Method for '{col}'",
                ["Simple Imputer", "KNN Imputer", "Iterative Imputer"],
                key=f"method_{col}"
            )

            if methods[col] == "Simple Imputer":
                strategies[col] = st.selectbox(
                    f"Strategy for '{col}'",
                    ["mean", "median", "most_frequent"],
                    key=f"strategy_{col}"
                )
            elif methods[col] == "KNN Imputer":
                k_values[col] = st.number_input(
                    f"Number of Neighbors (k) for '{col}'",
                    min_value=1,
                    value=5,
                    key=f"k_{col}"
                )

        if st.button("✅ Apply Imputation", type="primary"):
            df_copy = df.copy()

            # 1. Simple Imputer
            simple_cols = [c for c in selected_columns if methods[c] == "Simple Imputer"]
            for col in simple_cols:
                imputer = SimpleImputer(strategy=strategies[col])
                df_copy[[col]] = imputer.fit_transform(df_copy[[col]])

            # 2. KNN Imputer (Needs encoded numeric data)
            knn_cols = [c for c in selected_columns if methods[c] == "KNN Imputer"]
            if len(knn_cols) > 0:
                non_num_knn = df_copy[knn_cols].select_dtypes(exclude=[np.number]).columns.tolist()
                if len(non_num_knn) > 0:
                    st.error(f"❌ Columns {non_num_knn} must be Encoded before using KNN Imputer.")
                else:
                    k = list(k_values.values())[0] if len(k_values) > 0 else 5
                    imputer = KNNImputer(n_neighbors=k)
                    numeric_df = df_copy.select_dtypes(include=[np.number])
                    df_copy[numeric_df.columns] = imputer.fit_transform(numeric_df)

            # 3. Iterative Imputer
            iter_cols = [c for c in selected_columns if methods[c] == "Iterative Imputer"]
            if len(iter_cols) > 0:
                non_num_iter = df_copy[iter_cols].select_dtypes(exclude=[np.number]).columns.tolist()
                if len(non_num_iter) > 0:
                    st.error(f"❌ Columns {non_num_iter} must be Encoded before using Iterative Imputer.")
                else:
                    imputer = IterativeImputer(max_iter=10, random_state=0)
                    numeric_df = df_copy.select_dtypes(include=[np.number])
                    df_copy[numeric_df.columns] = imputer.fit_transform(numeric_df)

            st.session_state.data = df_copy
            st.success("✅ Missing values handled successfully!")
            st.rerun()

# ---------------- 4. Encoding (MODIFIED) ----------------
def handle_encoding():
    st.subheader("Encoding Options")
    df = st.session_state.data

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if len(categorical_cols) == 0:
        st.info("No categorical columns found.")
        return

    selected_cols = st.multiselect("Select Columns to Encode", categorical_cols)

    if selected_cols:
        encoding_methods = {}
        for col in selected_cols:
            encoding_methods[col] = st.selectbox(
                f"Method for '{col}'",
                ["Label Encoding", "One-Hot Encoding"],
                key=f"enc_{col}"
            )

        if st.button("✅ Apply Encoding", type="primary"):
            df_copy = df.copy()
            for col in selected_cols:
                if encoding_methods[col] == "Label Encoding":
                    # يحافظ على الـ NaN ولا يحولها لرقم
                    non_null_mask = df_copy[col].notnull()
                    le = LabelEncoder()
                    df_copy.loc[non_null_mask, col] = le.fit_transform(df_copy.loc[non_null_mask, col].astype(str))
                    # تحويل العمود لنوع رقمي لضمان قبول KNN Imputer له لاحقاً
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce') 
                else:
                    # One-Hot Encoding مع الاحتفاظ بالفراغات في أعمدة منفصلة
                    df_copy = pd.get_dummies(df_copy, columns=[col], prefix=col, dummy_na=True)
            
            st.session_state.data = df_copy
            st.success("✅ Encoding Applied! Missing values preserved as NaN.")
            st.dataframe(st.session_state.data.head(10))

# ---------------- 5. Normalization ----------------
def handle_normalization():
    st.subheader("Normalization / Scaling")
    df = st.session_state.data

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) == 0:
        st.info("No numeric columns found.")
        return

    selected_cols = st.multiselect("Select Columns to Normalize", numeric_cols)

    if selected_cols:
        norm_methods = {}
        for col in selected_cols:
            norm_methods[col] = st.selectbox(f"Method for '{col}'", ["Min-Max Scaling", "Standard Scaling"], key=f"norm_{col}")

        if st.button("✅ Apply Normalization", type="primary"):
            df_copy = df.copy()
            for col in selected_cols:
                scaler = MinMaxScaler() if norm_methods[col] == "Min-Max Scaling" else StandardScaler()
                df_copy[[col]] = scaler.fit_transform(df_copy[[col]])

            st.session_state.data = df_copy
            st.success("✅ Normalization Applied!")
            st.dataframe(st.session_state.data.head(10))

# ---------------- 6. Reset Data ----------------
def reset_data():
    st.subheader("Reset Data")
    if st.button("🔄 Reset to Original Data"):
        if st.session_state.original_data is not None:
            st.session_state.data = st.session_state.original_data.copy()
            st.success("Data reset to original!")
            st.rerun()

# ---------------- 7. Outliers ----------------
def handle_outliers():
    st.header("📦 Outliers Analysis")
    df = st.session_state.data
    if df is None: return

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols = st.multiselect("Select Columns", num_cols)
    method = st.selectbox("Choose Method", ["IQR", "Z-Score", "Winsorization"])

    if st.button("Apply Outlier Treatment"):
        df_copy = df.copy()
        if method == "IQR":
            for col in cols:
                q1, q3 = df_copy[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                df_copy = df_copy[(df_copy[col] >= q1 - 1.5*iqr) & (df_copy[col] <= q3 + 1.5*iqr)]
        elif method == "Z-Score":
            for col in cols:
                z = np.abs(stats.zscore(df_copy[col], nan_policy="omit"))
                df_copy = df_copy[z <= 3]
        else:
            for col in cols:
                df_copy[col] = winsorize(df_copy[col], limits=(0.05, 0.05))
        
        st.session_state.data = df_copy
        st.success(f"Outliers treated using {method}!")

# ---------------- Remaining Sections (Simplified for brevity) ----------------
def handle_transformation():
    st.subheader("🪄 Feature Transformation")
    # ... نفس كود التحويل السابق ...

def feature_selection():
    st.subheader("🎯 Feature Selection (RFE)")
    # ... نفس كود RFE السابق ...

def imbalancing():
    st.subheader("⚖️ Handle Class Imbalance")
    # ... نفس كود SMOTE السابق ...

def model_selection():
    st.subheader("🦾 Model Selection")
    # ... نفس كود التدريب السابق ...

def model_evaluation():
    st.subheader("📈 Model Evaluation")
    # ... نفس كود التقييم السابق ...

# ---------------- Main App ----------------
def main():
    st.title("📊 Machine Learning Data Preprocessing Tool")

    if "data" not in st.session_state:
        st.session_state.data = None
    if "original_data" not in st.session_state:
        st.session_state.original_data = None

    upload_data()

    if st.session_state.data is not None:
        page = st.sidebar.radio(
            "📋 Go to",
            ["📊 Visualization", "🏷️ Encoding", "📏 Normalization",  "🔧 Missing Values",
             "🔄 Reset Data", "📦 Outliers", "🪄 Transformation", "📍Feature Selection", 
             "⚖️ Handling Imbalancing", "🚀 Model Selection", "📈 Model Evaluation"]
        )

        if page == "📊 Visualization": data_visualization()
        elif page == "🏷️ Encoding": handle_encoding()
        elif page == "📏 Normalization": handle_normalization()
        elif page == "🔧 Missing Values": handle_missing_values()
        elif page == "🔄 Reset Data": reset_data()
        elif page == "📦 Outliers": handle_outliers()
        elif page == "🪄 Transformation": handle_transformation()
        elif page == "📍Feature Selection": feature_selection()
        elif page == "⚖️ Handling Imbalancing": imbalancing()
        elif page == "🚀 Model Selection": model_selection()
        elif page == "📈 Model Evaluation": model_evaluation()

if __name__ == "__main__":
    main()