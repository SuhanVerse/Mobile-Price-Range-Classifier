# Streamlit web app for Mobile Price Classification 

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

# ------------------------------
# Page config and CSS
# ------------------------------
st.set_page_config(
    page_title="📱 Mobile Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #EAEAEA;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #FF4B4B;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B, #FF6B6B);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #e23e3e, #ff7b7b);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .card {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Paths
# ------------------------------
repo_root = Path(__file__).resolve().parent
data_dir = repo_root / "data"
results_dir = repo_root / "results"
results_dir.mkdir(exist_ok=True)

train_path = Path(os.getenv("TRAIN_PATH", data_dir / "train.csv"))
test_path = Path(os.getenv("TEST_PATH", data_dir / "test.csv"))

model_path = results_dir / "svm_pipeline.joblib"
feature_order_path = results_dir / "feature_order.csv"

# ------------------------------
# Load data
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data(train_csv: Path, test_csv: Path):
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing data files. Expected:\n- {train_csv}\n- {test_csv}")
    return pd.read_csv(train_csv), pd.read_csv(test_csv)

# ------------------------------
# Build or load model
# ------------------------------
@st.cache_resource
def build_or_load_model(df_train: pd.DataFrame):
    if model_path.exists() and feature_order_path.exists():
        pipeline = joblib.load(model_path)
        feature_order = pd.read_csv(feature_order_path, header=None)[0].tolist()
        return pipeline, feature_order

    X = df_train.drop(columns=["price_range"])
    y = df_train["price_range"]

    feature_order = list(X.columns)
    pd.Series(feature_order).to_csv(feature_order_path, index=False, header=False)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ])

    param_grid = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__kernel": ["linear", "rbf", "poly"],
        "clf__gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    pipeline = grid.best_estimator_
    joblib.dump(pipeline, model_path)

    return pipeline, feature_order

# ------------------------------
# Helpers
# ------------------------------
def align_features(df: pd.DataFrame, feature_order: list):
    missing = [f for f in feature_order if f not in df.columns]
    for m in missing:
        df[m] = np.nan
    return df[feature_order]

def predict_single(pipeline, feature_order, inputs_dict):
    arr = np.array([[inputs_dict[name] for name in feature_order]], dtype=float)
    pred = pipeline.predict(arr)[0]
    proba = pipeline.predict_proba(arr)[0]
    return pred, proba

def class_label_map():
    return {
        0: ("Low 💰", "#28a745"),
        1: ("Mid-Range 💎", "#ffc107"),
        2: ("High-End ✨", "#17a2b8"),
        3: ("Premium 👑", "#dc3545"),
    }

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("⚙️ Configuration")
st.sidebar.write(f"Train CSV: {train_path}")
st.sidebar.write(f"Test CSV: {test_path}")

# ------------------------------
# Load data and model
# ------------------------------
st.title("📱 Mobile Price Predictor")

with st.spinner("Loading data..."):
    try:
        df_train, df_test = load_data(train_path, test_path)
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

with st.spinner("Building/Loading model..."):
    pipeline, feature_order = build_or_load_model(df_train)

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs(["Single Prediction 📱", "Batch Prediction 📊", "Model Performance 📈", "About ℹ️"])

# ------------------------------
# Single Prediction
# ------------------------------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Enter Mobile Specifications")
    cols = st.columns(4)
    inputs = {}
    bool_feats = {'blue', 'dual_sim', 'four_g', 'five_g', 'touch_screen', 'wifi'}

    for i, feature in enumerate(feature_order):
        with cols[i % 4]:
            if feature in bool_feats:
                inputs[feature] = int(st.toggle(feature.replace('_', ' ').title(), value=False))
            else:
                col_data = df_train[feature]
                min_val = np.nanmin(col_data)
                max_val = np.nanmax(col_data)
                default = np.nanmedian(col_data)

                if feature == "battery_power":
                    max_val = max(max_val, 5000)
                elif feature == "ram":
                    max_val = max(max_val, 8000)
                elif feature == "n_cores":
                    max_val = max(max_val, 12)
                elif feature == "m_dep":
                    min_val = 0.0
                    max_val = max(max_val, 1.0)
                    inputs[feature] = st.slider(
                        feature.replace('_', ' ').title(),
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default),
                        step=0.1
                    )
                    continue

                inputs[feature] = st.slider(
                    feature.replace('_', ' ').title(),
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(default),
                    step=1
                )

    if st.button("🚀 Predict Price Range", use_container_width=True):
        pred, proba = predict_single(pipeline, feature_order, inputs)
        labels = class_label_map()
        label, color = labels[pred]

        st.success(f"🎯 Predicted Category: {label}")
        fig = go.Figure(data=[
            go.Bar(
                x=['Low', 'Mid-Range', 'High-End', 'Premium'],
                y=proba * 100,
                marker_color=['#28a745', '#ffc107', '#17a2b8', '#dc3545']
            )
        ])
        fig.update_layout(title="Prediction Confidence", yaxis_title="Confidence (%)", height=320)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Batch Prediction
# ------------------------------
with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📂 Upload CSV for Batch Prediction")
    st.write("Your CSV must contain these columns:")
    st.code(", ".join(feature_order))

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            X_test = align_features(test_df.copy(), feature_order)
            preds = pipeline.predict(X_test)
            test_df["predicted_price_range"] = preds

            st.success("✅ Predictions completed successfully!")
            st.dataframe(test_df, use_container_width=True)

            # Download buttons
            csv_bytes = test_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV",
                               data=csv_bytes,
                               file_name="mobile_price_predictions.csv",
                               mime="text/csv",
                               use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Model Performance
# ------------------------------
with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Validation Performance")

    # Split again for validation metrics
    X = df_train.drop(columns=["price_range"])
    y = df_train["price_range"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    # Accuracy metric
    st.metric("Validation Accuracy", f"{acc:.2%}")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=['Low', 'Mid-Range', 'High-End', 'Premium'],
        y=['Low', 'Mid-Range', 'High-End', 'Premium'],
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )
    fig_cm.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report
    st.text("Detailed Classification Report")
    st.code(classification_report(y_val, y_pred))
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# About
# ------------------------------
with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ About This App")
    st.markdown("""
This Streamlit app predicts mobile phone price ranges using a Support Vector Machine (SVM) classifier.  
It employs a robust **Pipeline** (imputation + scaling + SVM), cross-validated **GridSearchCV** for hyperparameter tuning,  
and reproducible artifacts (saved model and feature order).

### ✨ Features
- **Single Prediction**: Enter phone specs interactively and get instant predictions with confidence scores.
- **Batch Prediction**: Upload a CSV of multiple phones and download predictions.
- **Model Performance**: View validation accuracy, confusion matrix, and classification report.
- **Reproducibility**: Model and feature order are saved in the `results/` folder for consistent predictions.

### 📊 Dataset
The model is trained on the Mobile Price Classification dataset, which includes features like:
- 🔋 Battery capacity
- 💾 RAM and internal memory
- 📱 Screen size and resolution
- 📷 Camera specs
- 📡 Connectivity (4G, 5G, WiFi, Bluetooth)

---
""")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("<div class='footer'>Made with ❤️ by Trinity | © 2025 | Empowering Mobile Price Predictions</div>", unsafe_allow_html=True)
