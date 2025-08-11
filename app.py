"""
app.py

Streamlit app for:
- EDA (data preview, summary, visualisations)
- Train & compare regression models (LinearRegression, RandomForest)
- Save/load models as model.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# ---------------------
# Config & Helpers
# ---------------------
st.set_page_config(page_title="Sales ML App", layout="wide")
st.title("📈 Sales ML Pipeline with Streamlit")

DATA_DIR = "data"
DEFAULT_DATA_FILENAME = "dataset.csv"
DATA_PATH = os.path.join(DATA_DIR, DEFAULT_DATA_FILENAME)
MODEL_PATH = "model.pkl"

@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day

    for col in ['Price', 'Quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Price' in df.columns and 'Quantity' in df.columns:
        df['Revenue'] = df['Price'] * df['Quantity']

    return df

def train_and_save_models(X, y, model_path=MODEL_PATH):
    numeric_feats = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_feats = [c for c in X.columns if c not in numeric_feats]

    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ], remainder='drop')

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    }

    trained = {}
    results = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, mdl in models.items():
        pipe = Pipeline([('pre', preprocessor), ('model', mdl)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        cv_rmse = -scores.mean()
        pipe.fit(X, y)
        trained[name] = pipe
        results[name] = {'cv_rmse': float(cv_rmse)}

    best_name = min(results.keys(), key=lambda n: results[n]['cv_rmse'])
    to_save = {
        'models': trained,
        'results': results,
        'best_model_name': best_name,
        'feature_cols': list(X.columns)
    }
    joblib.dump(to_save, model_path)
    return to_save

# ---------------------
# App UI / Pages
# ---------------------
# Sidebar navigation (Removed Model Performance & Predict)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Visualisations", "Model Training"])

# Data load
if os.path.exists(DATA_PATH):
    try:
        df = load_csv(DATA_PATH)
        df = clean_data(df)
        st.sidebar.success(f"Loaded data from `{DATA_PATH}`")
    except Exception as e:
        st.sidebar.error(f"Failed to load {DATA_PATH}: {e}")
        df = None
else:
    st.sidebar.info("No default dataset found. Upload a CSV or place `dataset.csv` in the data/ folder.")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df = clean_data(df)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        st.sidebar.success(f"Uploaded and saved to `{DATA_PATH}`")
    else:
        df = None

if df is None:
    st.warning("No dataset loaded yet.")
    st.stop()

# Page: Data Overview
if page == "Data Overview":
    st.header("Data Overview")
    st.write("Shape:", df.shape)
    st.subheader("Columns and dtypes")
    st.write(df.dtypes)
    st.subheader("Sample rows")
    st.dataframe(df.head(10))
    st.subheader("Missing values")
    miss = df.isnull().sum().reset_index()
    miss.columns = ['column', 'missing_count']
    st.dataframe(miss)

# Page: Visualisations
elif page == "Visualisations":
    st.header("Visualisations")
    with st.sidebar.expander("Visualization filters"):
        if 'City' in df.columns:
            city_choices = st.multiselect("Filter City", options=df['City'].dropna().unique().tolist())
        else:
            city_choices = None
        if 'Product' in df.columns:
            product_choices = st.multiselect("Filter Product", options=df['Product'].dropna().unique().tolist())
        else:
            product_choices = None

    df_vis = df.copy()
    if city_choices:
        df_vis = df_vis[df_vis['City'].isin(city_choices)]
    if product_choices:
        df_vis = df_vis[df_vis['Product'].isin(product_choices)]

    if 'Revenue' in df_vis.columns and 'City' in df_vis.columns:
        city_rev = df_vis.groupby('City', as_index=False)['Revenue'].sum().sort_values('Revenue', ascending=False)
        st.subheader("Revenue by City")
        st.plotly_chart(px.bar(city_rev, x='City', y='Revenue'), use_container_width=True)

# Page: Model Training
elif page == "Model Training":
    st.header("Model Training")
    if 'Revenue' not in df.columns:
        st.error("Target column `Revenue` not found.")
        st.stop()

    chosen = st.multiselect("Select features for training:", options=[col for col in df.columns if col != 'Revenue'])

    if chosen:
        X = df[chosen].copy()
        y = df['Revenue'].copy()
        mask = X.notnull().all(axis=1) & y.notnull()
        X = X[mask]
        y = y[mask]

        if st.button("Start training"):
            with st.spinner("Training models..."):
                model_info = train_and_save_models(X, y, model_path=MODEL_PATH)
            st.success(f"Best model: **{model_info['best_model_name']}**")
            for mname, res in model_info['results'].items():
                st.write(f"- {mname}: CV RMSE = {res['cv_rmse']:.4f}")
            with open(MODEL_PATH, "rb") as f:
                st.download_button("Download model.pkl", data=f, file_name="model.pkl")
