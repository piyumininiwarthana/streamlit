"""
app.py

Streamlit app for:
- EDA (data preview, summary, visualisations)
- Train & compare regression models (LinearRegression, RandomForest)
- Save/load models as model.pkl
- Interactive single-row predictions
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
from sklearn.metrics import mean_squared_error, r2_score
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

# Utility: load CSV (from default path)
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # strip column names
    df.columns = [c.strip() for c in df.columns]

    # trim string columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # parse Date if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day

    # ensure numeric
    for col in ['Price', 'Quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # create Revenue target if possible
    if 'Price' in df.columns and 'Quantity' in df.columns:
        df['Revenue'] = df['Price'] * df['Quantity']

    return df

def auto_select_features(df, max_unique_categorical=200):
    """Choose reasonable feature columns: numeric + categorical with not too many unique values."""
    # drop these automatic-exclusions
    excluded = {'Revenue', 'Date', 'Order ID', 'OrderID', 'Timestamp'}
    candidates = [c for c in df.columns if c not in excluded]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical = []
    for c in candidates:
        if c in numeric:
            continue
        nuniq = df[c].nunique(dropna=True)
        if 0 < nuniq <= max_unique_categorical:
            categorical.append(c)
    # final order: prefer Price, Quantity, Month first if present
    preferred = [c for c in ['Price','Quantity','Month','Year'] if c in numeric or c in categorical]
    others = [c for c in preferred + numeric + categorical if c not in preferred]
    # ensure unique and return
    final = []
    for c in (preferred + numeric + categorical):
        if c not in final and c in candidates:
            final.append(c)
    return final

def train_and_save_models(X, y, model_path=MODEL_PATH):
    # Identify numeric and categorical features
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
        # Use neg_root_mean_squared_error if available; fallback to neg_mean_squared_error
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
            cv_rmse = -scores.mean()
        except Exception:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            cv_rmse = np.sqrt(-scores.mean())
        pipe.fit(X, y)
        trained[name] = pipe
        results[name] = {'cv_rmse': float(cv_rmse), 'cv_scores': (-scores).tolist()}

    best_name = min(results.keys(), key=lambda n: results[n]['cv_rmse'])
    to_save = {
        'models': trained,
        'results': results,
        'best_model_name': best_name,
        'feature_cols': list(X.columns)
    }
    joblib.dump(to_save, model_path)
    return to_save

def load_model_file(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def predict_with_confidence(pipe, X_input):
    """Return prediction and, if model is RandomForest, estimate std across trees."""
    pred = pipe.predict(X_input)
    # Try to estimate uncertainty for RandomForest
    try:
        model = pipe.named_steps['model']
        if hasattr(model, 'estimators_'):
            # get predictions of individual trees on processed input
            # If pipeline preprocessor exists, transform X_input before feeding to trees
            try:
                transformed = pipe.named_steps['pre'].transform(X_input)
                preds = np.vstack([est.predict(transformed) for est in model.estimators_])
            except Exception:
                preds = np.vstack([est.predict(X_input) for est in model.estimators_])
            mean = preds.mean(axis=0)
            std = preds.std(axis=0)
            return mean, std
    except Exception:
        pass
    return pred, None

# ---------------------
# App UI / Pages
# ---------------------
# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Visualisations", "Model Training", "Model Performance", "Predict"])

# Data load: prefer dataset.csv in data/; otherwise allow upload
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
        # Save uploaded to data/dataset.csv for convenience
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        st.sidebar.success(f"Uploaded and saved to `{DATA_PATH}`")
    else:
        df = None

if df is None:
    st.warning("No dataset loaded yet. Upload a CSV in the sidebar or place `dataset.csv` in the `data/` folder.")
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

    st.subheader("Interactive filters")
    cols = st.multiselect("Choose columns to display", df.columns.tolist(), default=df.columns.tolist()[:6])
    if cols:
        st.dataframe(df[cols].head(200))

# Page: Visualisations
elif page == "Visualisations":
    st.header("Visualisations")
    st.markdown("Interactive charts. Use sidebar filters below to slice data.")

    # Optional filters
    with st.sidebar.expander("Visualization filters"):
        if 'City' in df.columns:
            city_choices = st.multiselect("Filter City", options=df['City'].dropna().unique().tolist(), default=None)
        else:
            city_choices = None
        if 'Product' in df.columns:
            product_choices = st.multiselect("Filter Product", options=df['Product'].dropna().unique().tolist(), default=None)
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
        st.plotly_chart(px.bar(city_rev, x='City', y='Revenue', title='Revenue by City'), use_container_width=True)

    if 'Date' in df_vis.columns and 'Revenue' in df_vis.columns:
        time_rev = df_vis.dropna(subset=['Date']).groupby('Date', as_index=False)['Revenue'].sum().sort_values('Date')
        st.subheader("Revenue over Time")
        st.plotly_chart(px.line(time_rev, x='Date', y='Revenue', title='Revenue over Time'), use_container_width=True)

    if 'Product' in df_vis.columns and 'Revenue' in df_vis.columns:
        top_prod = df_vis.groupby('Product', as_index=False)['Revenue'].sum().sort_values('Revenue', ascending=False).head(15)
        st.subheader("Top Products by Revenue")
        st.plotly_chart(px.bar(top_prod, x='Product', y='Revenue', title='Top Products by Revenue'), use_container_width=True)

# Page: Model Training
elif page == "Model Training":
    st.header("Model Training")
    if 'Revenue' not in df.columns:
        st.error("Target column `Revenue` not found. Ensure dataset has Price and Quantity columns or create Revenue.")
        st.stop()

    # Let the user pick features
    chosen = st.multiselect(
        "Select features for training:",
        options=[col for col in df.columns if col != 'Revenue']
    )

    # Remove rows with missing features or target
    if chosen:
        X = df[chosen].copy()
        y = df['Revenue'].copy()
        mask = X.notnull().all(axis=1) & y.notnull()
        X = X[mask]
        y = y[mask]

        st.write("Training set shape:", X.shape)

        if st.button("Start training"):
            with st.spinner("Training models (5-fold CV). This may take a few minutes..."):
                model_info = train_and_save_models(X, y, model_path=MODEL_PATH)
            st.success(f"Training finished. Best model: **{model_info['best_model_name']}**")
            st.write("Cross-validated RMSE results:")
            for mname, res in model_info['results'].items():
                st.write(f"- {mname}: CV RMSE = {res['cv_rmse']:.4f}")
            # Offer download
            with open(MODEL_PATH, "rb") as f:
                st.download_button("Download model.pkl", data=f, file_name="model.pkl")
    else:
        st.info("Pick at least one feature to train models.")



# End of app
st.sidebar.markdown("---")
st.sidebar.caption("Created for ML deployment assignment. Place dataset at data/dataset.csv or upload one.")