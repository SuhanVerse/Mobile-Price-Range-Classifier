# 📱 Mobile Price Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Deployed](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-brightgreen?logo=streamlit)](https://suhanversempc.streamlit.app/)

This project leverages a **Support Vector Machine (SVM)** classifier with optimized hyperparameters to predict mobile phone price ranges based on their specifications. It also includes a full‑fledged **Streamlit web app** for interactive predictions.

🔗 **Live Demo**: [Mobile Price Predictor App](https://suhanversempc.streamlit.app/)

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Changes Made](#changes-made)
- [Deployment](#deployment)
- [License](#license)

---

## 🚀 Project Overview
The **Mobile Price Classification** project aims to classify mobile phones into four price ranges (Low, Mid‑Range, High‑End, Premium) using their technical specifications.  
The model is built with **SVM** and optimized via **GridSearchCV** for improved accuracy.  

The Streamlit app provides:
- **Single prediction** with sliders/toggles for features.
- **Batch prediction** via CSV upload.
- **Model performance dashboard** with accuracy, confusion matrix, and classification report.

---

## ✨ Features
- Predicts mobile phone price ranges based on specifications.
- Interactive **Streamlit web app** with modern UI.
- **Single & batch predictions** supported.
- Optimized **SVM classifier** with hyperparameter tuning.
- Downloadable results (CSV).
- Validation metrics and visualizations.

---

## 📊 Dataset
The dataset includes features such as:
- 🔋 Battery capacity  
- 💾 RAM & internal memory  
- 📱 Screen size & resolution  
- 📷 Camera specs (front & primary)  
- ⚖️ Mobile weight  
- 📡 Connectivity (Bluetooth, Wi‑Fi, 4G, 5G, Dual SIM, Touchscreen)  

---

## 📂 Project Structure

Mobile-Price-Range-Classifier/
│
├── app.py                  # 🎨 Streamlit web app (main entry point)
├── requirements.txt        # 📦 Python dependencies
├── README.md               # 📖 Project documentation
├── LICENSE
├── data/                   # 📊 Datasets
│   ├── train.csv           # Training dataset
│   ├── test.csv            # Test dataset
├── results/                # 📁 Saved models & outputs
│   ├── svm_pipeline.joblib # Trained SVM pipeline
│   ├── feature_order.csv   # Saved feature order for reproducibility
│   └── test_predictions_*.csv # Batch prediction outputs
├── src/                    # 📓 Jupyter notebooks for EDA & experiments
└   └── MPC.ipynb # Model training & evaluation

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SuhanVerse/Mobile-Price-Range-Classifier.git
cd Mobile-Price-Range-Classifier
pip install -r requirements.txt
```

## Usage

To use the application, run the main script or Jupyter Notebook provided in the repository. The steps include:

## Load the dataset

Preprocess the data (handle missing values, normalize features).
Train the SVM model with the training data.
Evaluate the model on the test data.

## Model Evaluation

The model's performance is evaluated using metrics such as:

Accuracy
Precision
Recall
F1-score
Hyperparameter tuning is performed to optimize the SVM classifier for better results.

## Results

The SVM classifier achieved a high accuracy rate in predicting mobile phone price ranges. Detailed results, including confusion matrices and classification reports, are available in the results section of the notebook.
