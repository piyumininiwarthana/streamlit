# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------
# Config
# --------------------
st.set_page_config(page_title="Sales ML App", layout="wide")
MODEL_FILE = "best_model.joblib"   # saved model + metadata
DEFAULT_DATA_RELATIVE = "data/Sales-Data-Analysis.csv"  # optional in-repo dataset

# --------------------
# Helpers
# --------------------
@st.cache_data
def read_csv(uploaded_file):
    """Read uploaded csv or fallback to repo-relative default if exists."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists(DEFAULT_DATA_RELATIVE):
        return pd.read_csv(DEFAULT_DATA_RELATIVE)
    return None

def detect_feature_types(df, features):
    """Return dict mapping feature -> type: 'numeric'|'datetime'|'categorical'"""
    types = {}
    for c in features:
        ser = df[c]
        if pd.api.types.is_numeric_dtype(ser):
            types[c] = 'numeric'
            continue
        # try convert to datetime
        try:
            ser_dt = pd.to_datetime(ser, errors='coerce')
            non_na = ser_dt.notna().sum()
            if non_na >= (0.5 * len(ser)):  # mostly datetime-like
                types[c] = 'datetime'
                continue
        except Exception:
            pass
        types[c] = 'categorical'
    return types

def preprocess_for_training(df, features, feature_types):
    """
    Input: original df and selected raw features + types.
    Output: X_encoded (DataFrame ready for modeling), and X_raw_transformed (used to save for reference)
    - datetime -> ordinal integer
    - categorical left as strings (we'll one-hot via get_dummies)
    - numeric kept
    """
    X = pd.DataFrame()
    for c in features:
        if feature_types[c] == 'datetime':
            X[c] = pd.to_datetime(df[c], errors='coerce').map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
        elif feature_types[c] == 'numeric':
            X[c] = pd.to_numeric(df[c], errors='coerce')
        else:  # categorical
            X[c] = df[c].astype(str).fillna("nan")
    # fill numeric NaN with median
    for col in X.select_dtypes(include=[np.number]).columns:
        med = X[col].median()
        X[col] = X[col].fillna(med)
    # get_dummies for categorical (will also keep numeric columns)
    X_enc = pd.get_dummies(X, drop_first=False)
    return X_enc, X

def preprocess_for_prediction(input_raw: dict, feature_types, saved_feature_columns):
    """
    input_raw: dict of raw input values keyed by original feature names
    feature_types: dict from detect_feature_types
    saved_feature_columns: list of columns used by the trained model (after get_dummies)
    Returns: one-row dataframe aligned to saved_feature_columns
    """
    df = pd.DataFrame([input_raw])
    # convert datetimes and numerics analogous to training
    for c in list(df.columns):
        if feature_types[c] == 'datetime':
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce').map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
            except:
                df[c] = np.nan
        elif feature_types[c] == 'numeric':
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                df[c] = np.nan
        else:
            df[c] = df[c].astype(str).fillna("nan")
    # fill numeric NaNs with 0 (or could use training medians if stored)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(0)
    # one-hot encode and align
    df_enc = pd.get_dummies(df, drop_first=False)
    df_enc = df_enc.reindex(columns=saved_feature_columns, fill_value=0)
    return df_enc

def train_models_and_save(X_enc, y, raw_features, feature_types):
    """Train several models, evaluate, save best + metadata."""
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100)
    }
    results = {}
    best_name = None
    best_score = -np.inf
    best_model = None

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"MAE": float(mae), "MSE": float(mse), "R2": float(r2)}
        if r2 > best_score:
            best_score = r2
            best_name = name
            best_model = model

    # Save model and metadata
    model_data = {
        "model": best_model,
        "model_name": best_name,
        "metric_results": results,
        "feature_columns": X_enc.columns.tolist(),
        "raw_features": raw_features,
        "feature_types": feature_types
    }
    joblib.dump(model_data, MODEL_FILE)
    return model_data

def load_model_data():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

# --------------------
# UI: Navigation
# --------------------
st.title("Sales Data — Overview, Train, Evaluate, Predict")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload / Overview", "Visualization", "Model Training", "Model Performance", "Prediction"])

# --------------------
# Load Data (uploader OR default repo data)
# --------------------
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
df = read_csv(uploaded)

if df is None:
    st.info("Please upload a CSV file via the sidebar, or add a dataset at `data/Sales-Data-Analysis.csv` in the repo.")
    st.stop()

# Show basic preview on all pages
if page == "Upload / Overview":
    st.header("Dataset Preview & Info")
    st.write("Rows:", df.shape[0], "Columns:", df.shape[1])
    st.dataframe(df.head(200))
    st.write("Column types (sample):")
    st.write(df.dtypes.astype(str))

# --------------------
# Visualization
# --------------------
elif page == "Visualization":
    st.header("Simple Visualizations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    viz_type = st.selectbox("Plot type", ["Histogram (numeric)", "Bar / value counts (categorical)"])
    if viz_type == "Histogram (numeric)":
        if not numeric_cols:
            st.warning("No numeric columns to plot.")
        else:
            col = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=30)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            st.pyplot(fig)
    else:
        col = st.selectbox("Select column", all_cols)
        vc = df[col].value_counts().head(50)
        fig, ax = plt.subplots(figsize=(8, min(6, 0.15*len(vc))))
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_xticklabels(vc.index.astype(str), rotation=45, ha='right')
        st.pyplot(fig)

# --------------------
# Model Training
# --------------------
elif page == "Model Training":
    st.header("Train model")
    st.write("Choose target and features (dates are supported).")
    all_columns = df.columns.tolist()
    target = st.selectbox("Select target column (numeric target is expected)", all_columns)
    candidate_features = [c for c in all_columns if c != target]
    selected = st.multiselect("Select feature columns", candidate_features, default=candidate_features[:3])

    if st.button("Train"):
        if not selected:
            st.error("Please select at least one feature.")
        else:
            # detect feature types
            feat_types = detect_feature_types(df, selected)
            # preprocess
            X_enc, X_raw = preprocess_for_training(df, selected, feat_types)
            # prepare target
            y = pd.to_numeric(df[target], errors='coerce').fillna(0)
            # train and save
            with st.spinner("Training models..."):
                model_data = train_models_and_save(X_enc, y, selected, feat_types)
            st.success(f"Trained. Best model: {model_data['model_name']} (R² = {model_data['metric_results'][model_data['model_name']]['R2']:.3f})")
            st.write("All model results:")
            st.table(pd.DataFrame(model_data['metric_results']).T)

# --------------------
# Model Performance
# --------------------
elif page == "Model Performance":
    st.header("Model Performance / Evaluation")
    model_data = load_model_data()
    if model_data is None:
        st.warning("No trained model found. Please train a model first.")
    else:
        st.write("Best model:", model_data["model_name"])
        st.write("Trained on features (encoded):", len(model_data["feature_columns"]), "columns")
        st.write("Raw features used:", model_data["raw_features"])
        st.write("Feature types:", model_data["feature_types"])
        res_df = pd.DataFrame(model_data["metric_results"]).T
        st.table(res_df)

# --------------------
# Prediction
# --------------------
elif page == "Prediction":
    st.header("Make a prediction with the saved model")
    model_data = load_model_data()
    if model_data is None:
        st.warning("No trained model found. Please train a model first.")
        st.stop()

    raw_features = model_data["raw_features"]
    feat_types = model_data["feature_types"]
    saved_cols = model_data["feature_columns"]

    st.write("Enter values for each feature (match training units):")
    input_vals = {}
    for f in raw_features:
        t = feat_types.get(f, 'categorical')
        if t == 'numeric':
            # use mean as default if present
            default = None
            try:
                default = float(df[f].dropna().astype(float).mean())
            except:
                default = 0.0
            input_vals[f] = st.number_input(f, value=float(default))
        elif t == 'datetime':
            # provide date_input; default to first non-null value or today
            try:
                first = pd.to_datetime(df[f], errors='coerce').dropna().iloc[0]
                default_date = first.date()
            except:
                default_date = pd.Timestamp.now().date()
            dt = st.date_input(f, value=default_date)
            input_vals[f] = pd.to_datetime(dt)
        else:  # categorical
            # default to most common category
            try:
                default_cat = str(df[f].dropna().mode().iloc[0])
            except:
                default_cat = ""
            input_vals[f] = st.text_input(f, value=default_cat)

    if st.button("Predict"):
        # preprocess single row
        # convert datetimes to ordinal, leave numeric, categorical as string
        raw_for_pred = {}
        for k, v in input_vals.items():
            if feat_types[k] == 'datetime':
                if pd.isna(v):
                    raw_for_pred[k] = np.nan
                else:
                    raw_for_pred[k] = pd.to_datetime(v)
            else:
                raw_for_pred[k] = v

        X_pred_enc = preprocess_for_prediction(raw_for_pred, feat_types, saved_cols)
        model = model_data["model"]
        try:
            pred = model.predict(X_pred_enc)[0]
            st.success(f"Predicted value: {pred:.4f}")
        except Exception as e:
            st.error("Prediction failed — check inputs. Error: " + str(e))
