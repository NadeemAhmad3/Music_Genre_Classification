# 🎵 TuneFinder AI: A Deep Learning Music Genre Classifier

![python-shield](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![tensorflow-shield](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![sklearn-shield](https://img.shields.io/badge/scikit--learn-1.2%2B-blue)
![xgboost-shield](https://img.shields.io/badge/XGBoost-1.7%2B-blue)
![librosa-shield](https://img.shields.io/badge/Librosa-0.9%2B-ff69b4)
![streamlit-shield](https://img.shields.io/badge/Streamlit-1.25%2B-red)

An **end-to-end machine learning project** that builds, evaluates, and compares multiple deep learning and classical models to classify music genres from raw audio. This repository contains the complete workflow—from in-depth exploratory data analysis (EDA) and model implementation in a Jupyter Notebook to a polished, interactive web application built with Streamlit.

> 💡 The project's key insight is a direct comparison between **Tabular Feature Engineering** and **Image-Based Deep Learning**, revealing which approach is superior for the GTZAN dataset and why.

---

## 🌟 Key Features

- ✨ **Multi-Model Prediction**: Upload an audio file and get simultaneous genre predictions from our two best-performing models: XGBoost and a Dense Neural Network.
- 🖼️ **Dual Prediction Modes**: Test the AI with either raw audio (`.wav`) or pre-generated spectrogram images to see how different models handle different data types.
- 📊 **In-Depth EDA Gallery**: An entire section of the web app dedicated to showcasing 12+ visualizations of the GTZAN dataset, from feature distributions to single-file audio analysis.
- 🏆 **Performance Dashboard**: Interactively view and compare the final accuracy and confusion matrices for all four trained models to understand their strengths and weaknesses.
- 🎨 **Modern UI**: A clean, intuitive, and visually appealing dark-themed user interface with custom styling and a professional layout.

---

## 🛠️ Tech Stack & Libraries

| Category                | Tools & Libraries                                        |
|-------------------------|----------------------------------------------------------|
| **Data Processing**     | Pandas, NumPy, Librosa (for Audio)                       |
| **Visualization**       | Matplotlib, Seaborn                                      |
| **Machine Learning**    | Scikit-learn (Preprocessing, Metrics), XGBoost           |
| **Deep Learning**       | TensorFlow, Keras                                        |
| **Web Application**     | Streamlit                                                |
| **Development**         | Jupyter Notebook, Python 3.10+, Joblib                   |

---

## 📁 Project Structure

```bash
.
├── input/
│   └── Data/
│       ├── features_3_sec.csv
│       ├── genres_original/
│       └── images_original/
├── outputs/
│   ├── models/
│   │   ├── custom_cnn_model.keras
│   │   ├── dnn_model.keras
│   │   ├── label_encoder.joblib
│   │   ├── tabular_scaler.joblib
│   │   ├── training_columns.joblib
│   │   ├── transfer_learning_vgg16_model.keras
│   │   └── xgboost_model.joblib
│   └── visualizations/
│       └── (16+ saved plots and diagrams)
├── your_notebook.ipynb     # Jupyter Notebook for analysis and model training
├── app.py                  # Main Streamlit web app script
├── style.css               # Custom CSS for styling the app
└── README.md
