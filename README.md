# 📱 Mobile Price Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Deployed](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-brightgreen?logo=streamlit)](https://suhanversempc.streamlit.app/)

This project leverages a **Support Vector Machine (SVM)** classifier with optimized hyperparameters to predict mobile phone price ranges based on their specifications. It also includes a full-featured **Streamlit web app** for interactive predictions.

🔗 **Live Demo**: [Mobile Price Predictor App](https://suhanversempc.streamlit.app/)

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [Model Evaluation](#-model-evaluation)
- [Results](#-results)
- [Deployment](#-deployment)
- [License](#-license)

---

## 🚀 Project Overview

The **Mobile Price Classification** project aims to classify mobile phones into four price ranges (Low, Mid‑Range, High‑End, Premium) using their technical specifications.  
The model is built with **SVM** and optimized via **GridSearchCV** for improved accuracy.

The Streamlit app provides:

- **Single prediction** with sliders and toggles for features.
- **Batch prediction** via CSV upload.
- **Model performance dashboard** with accuracy, confusion matrix, and classification report.

---

## ✨ Features

- Predicts mobile phone price ranges based on specifications.
- Interactive **Streamlit web app** with modern UI.
- **Single and batch predictions** supported.
- Optimized **SVM classifier** with hyperparameter tuning.
- Downloadable results (CSV).
- Validation metrics and visualizations.

---

## 📊 Dataset

The dataset includes features such as:

- 🔋 Battery capacity
- 💾 RAM and internal memory
- 📱 Screen size and resolution
- 📷 Camera specs (front and primary)
- ⚖️ Mobile weight
- 📡 Connectivity (Bluetooth, Wi‑Fi, 4G, 5G, Dual SIM, Touchscreen)

---

## 📂 Project Structure

```bash
Mobile-Price-Range-Classifier/
│
├── app.py                # 🎨 Streamlit web app (main entry point)
├── requirements.txt      # 📦 Python dependencies
├── README.md             # 📖 Project documentation
├── LICENSE
│
├── data/                 # 📊 Datasets
│   ├── train.csv         # Training dataset
│   └── test.csv          # Test dataset
│
├── results/              # 📁 Saved models & outputs
│   ├── svm_pipeline.joblib        # Trained SVM pipeline
│   ├── feature_order.csv          # Saved feature order for reproducibility
│   └── test_predictions_*.csv     # Batch prediction outputs
│
└── src/                  # 📓 Jupyter notebooks for EDA & experiments
    └── MPC.ipynb         # Model training & evaluation
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SuhanVerse/Mobile-Price-Range-Classifier.git
cd Mobile-Price-Range-Classifier
pip install -r requirements.txt
```

**To run the Jupyter Notebook:**  
If you want to explore the notebook (`src/MPC.ipynb`), install Jupyter as well:

```bash
pip install notebook jupyterlab
```

---

## ▶️ Usage

### Run the Streamlit App locally

```bash
streamlit run app.py
```

Or use the hosted app:

👉 [Mobile Price Predictor App](https://suhanversempc.streamlit.app/)

### Run the Jupyter Notebook

Navigate to the `src/` folder and open the notebook:

```bash
cd src
jupyter notebook MPC.ipynb
```

---

## 📈 Model Evaluation

The model's performance is evaluated using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- **Classification Report**

Hyperparameter tuning is performed with **GridSearchCV** to optimize the SVM classifier for better results.

---

## 🏆 Results

- The SVM classifier achieved **high accuracy** in predicting mobile phone price ranges.
- Validation results include **confusion matrices** and **classification reports**.
- The Streamlit app provides **real-time predictions** with confidence scores.

---

## 🌐 Deployment

The app is deployed on **Streamlit Cloud** and accessible here:  
👉 [https://suhanversempc.streamlit.app/](https://suhanversempc.streamlit.app/)

---

## 📜 License

This project is licensed under the **Apache License 2.0** – see the [LICENSE](LICENSE) file for details.

---
