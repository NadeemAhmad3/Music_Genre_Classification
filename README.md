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
```
## ⚙️ Installation & Setup
**1. Clone the Repository** 
```bash
git clone https://github.com/NadeemAhmad3/Music_Genre_Classification.git
cd Music_Genre_Classification
```
**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/Mac
source venv/bin/activate
```
**3. Install Dependencies**

```bash
pip install pandas numpy librosa scikit-learn xgboost tensorflow matplotlib seaborn streamlit
```
**4. Download the Dataset**
The dataset used is the GTZAN Genre Collection. Download it from Kaggle:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
Unzip the file and place the Data folder into an input directory at the root of the project, as shown in the project structure.
## ▶️ How to Run the Project
⚠️ **Important:** Due to the large size of the dataset (input/) and the trained models (outputs/models/), these directories are typically not uploaded to GitHub. You must first run the notebook to generate the necessary files before launching the web app.
**1. Run the Jupyter Notebook** 
```bash
jupyter lab
```
Then open .ipynb to walk through data analysis and model training.
Launch Jupyter and run all cells in the notebook from top to bottom.
This will perform the EDA, train all four models, and save the required model files, preprocessors, and visualization plots into the outputs directory.
**2. Launch the Streamlit App** 
```bash
streamlit run streamlit_app.py
```
Your browser will automatically open the app at http://localhost:8501.
## 🧠 Modeling & Results

Three primary recommendation models were built and evaluated, which are then combined into a hybrid system.

| Model                      | Methodology              | Test Accuracy                      | Key Insight                                                             | 
| ---------------------------| ------------------------ | ---------------------------------- | ------------------------------------------------------------------------| 
| **Dense Neural Network**   | Tabular                  | **~91.1%**                         | **Best overall performer**.find complex patterns in engineered features.| 
| **XGBoost**                | Tabular                  | **~90.5%**                         | Highly accurate and reliable. A top contender.                          | 
| **(VGG16)**                | Image-Based              | **~58.0%**                         | Drastically better than a custom CNN                                    |
| **Custom CNN**             | Image-Based              | **~6.3%**                          | Failed to learn,                                                        |

✅ The final **Streamlit application** ntegrates the two top-performing models—**XGBoost** and the **Dense Neural Network**—for the live audio prediction tool, as they provide the most accurate and trustworthy results. The image-based models are also included to demonstrate the full scope of the project.

## 🤝 Contributing
We welcome contributions!

**1.** Fork the repo

**2.** Create your feature branch
```bash
git checkout -b feature/AmazingFeature
```
**3.** Commit your changes
```bash
git commit -m "Add some AmazingFeature"
```
**4.** Push to your branch
```bash
git push origin feature/AmazingFeature
```
**5.** Open a Pull Request

## 📧 Contact
**Nadeem Ahmad**

📫 **onedaysuccussfull@gmail.com**



