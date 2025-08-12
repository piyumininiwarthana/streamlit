# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# -------- Load CSV --------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\stream_lit\data\Sales-Data-Analysis.csv")
    # Convert date-like columns to datetime
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    return df

# -------- Encode + Train Model --------
def train_and_save_model(df, features, target):
    df = df.copy()

    # Convert datetimes to numeric (ordinal)
    for col in features:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].map(pd.Timestamp.toordinal)

    # Encode categorical strings
    le = LabelEncoder()
    for col in features:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col].astype(str))

    X = df[features]
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)
    joblib.dump((model, features), "model.pkl")
    return model

# -------- Streamlit Pages --------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Visualization", "Model Training", "Model Performance", "Predict"])

df = load_data()

# Overview
if page == "Overview":
    st.title("Data Overview")
    st.write(df.head())
    st.write(df.describe())

# Visualization
elif page == "Visualization":
    st.title("Data Visualization")
    col = st.selectbox("Select column", df.columns)
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True)
    st.pyplot(plt)

# Model Training
elif page == "Model Training":
    st.title("Train Model")
    target_col = st.selectbox("Select Target Column", df.columns)
    feature_cols = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target_col])

    if st.button("Train Model"):
        if not feature_cols:
            st.error("Please select at least one feature")
        else:
            train_and_save_model(df, feature_cols, target_col)
            st.success("Model trained and saved successfully!")

# Model Performance
elif page == "Model Performance":
    st.title("Model Performance")
    try:
        model, features = joblib.load("model.pkl")
        X = df[features].copy()

        # Handle datetime in performance check
        for col in features:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].map(pd.Timestamp.toordinal)
        for col in features:
            if X[col].dtype == "object":
                X[col] = pd.factorize(X[col])[0]

        y = df[st.selectbox("Select Target Column", df.columns)]
        score = model.score(X, y)
        st.write(f"R² Score: {score:.2f}")
    except:
        st.error("Please train the model first.")

# Predict
elif page == "Predict":
    st.title("Make Predictions")
    try:
        model, features = joblib.load("model.pkl")
        inputs = {}
        for f in features:
            val = st.text_input(f"Enter value for {f}")
            try:
                val = float(val)
            except:
                try:
                    val = pd.to_datetime(val).toordinal()
                except:
                    pass
            inputs[f] = val

        if st.button("Predict"):
            input_df = pd.DataFrame([inputs])
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Value: {prediction}")
    except:
        st.error("Please train the model first.")
