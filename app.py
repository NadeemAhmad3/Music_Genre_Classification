import streamlit as st
import os
import pandas as pd
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import xgboost as xgb

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TuneFinder - Music Genre AI",
    page_icon="🎵",
    layout="wide"
)

# --- 2. LOAD STYLESHEET ---
def local_css(file_name):
    """Function to load a local CSS file."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found. Please ensure it is in the same directory as the app.")

local_css("style.css")

# --- 3. MODEL & PREPROCESSOR LOADING ---
@st.cache_resource
def load_all_models():
    """Load all models (tabular and image-based) and preprocessors."""
    models_dir = os.path.join("outputs", "models")
    
    models = {
        'dnn': load_model(os.path.join(models_dir, "dnn_model.keras")),
        'xgb': joblib.load(os.path.join(models_dir, "xgboost_model.joblib")),
        # --- NEW: Load image models ---
        'cnn': load_model(os.path.join(models_dir, "custom_cnn_model.keras")),
        'vgg16': load_model(os.path.join(models_dir, "transfer_learning_vgg16_model.keras"))
    }
    
    preprocessors = {
        'scaler': joblib.load(os.path.join(models_dir, "tabular_scaler.joblib")),
        'label_encoder': joblib.load(os.path.join(models_dir, "label_encoder.joblib")),
        'training_columns': joblib.load(os.path.join(models_dir, 'training_columns.joblib'))
    }
    # Class names for image models (assuming they are the same as tabular)
    preprocessors['class_names'] = preprocessors['label_encoder'].classes_.tolist()
    
    return models, preprocessors
try:
    models, preprocessors = load_all_models()
except FileNotFoundError as e:
    st.error(f"A required file was not found: {e}. Please ensure all model and preprocessor files, including 'training_columns.joblib', exist in the 'outputs/models/' directory.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading models: {e}")
    st.stop()

# --- 4. HELPER FUNCTIONS ---
def extract_features_from_audio(audio_file):
    """Extracts tabular features and aligns them with training columns."""
    y, sr = librosa.load(audio_file, sr=22050, duration=30)
    features = {}
    
    # Extract all required features
    features['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    features['rms'] = np.mean(librosa.feature.rms(y=y))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc{i+1}'] = np.mean(mfcc)
    
    features_df = pd.DataFrame([features])
    
    # Align columns with the training data to prevent feature mismatch errors
    training_cols = preprocessors['training_columns']
    for col in training_cols:
        if col not in features_df.columns:
            features_df[col] = 0
    return features_df[training_cols]

# --- NEW: Image preprocessing function ---
def preprocess_image(image_file):
    """Loads and preprocesses an image for the CNN models."""
    img = load_img(image_file, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    return img_array
# --- 5. UI LAYOUT ---

# --- NAVIGATION BAR (FIXED) ---
st.markdown("""
<div class="navbar">
    <a class="navbar-logo" href="#home">TuneFinder</a>
    <div class="navbar-links">
        <a href="#about-the-project">About</a>
        <a href="#predict-a-genre">Predict</a>
        <a href="#eda-gallery">EDA Gallery</a>
        <a href="#model-insights">Model Insights</a>
    </div>
</div>
""", unsafe_allow_html=True)


# --- HERO SECTION (HOME) ---
st.markdown('<div id="home"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero-container">
    <img class="hero-bg" src="https://images.unsplash.com/photo-1507874457470-272b3c8d8ee2?q=80&w=1880&auto=format&fit=crop">
    <div class="hero-content">
        <h1 class="hero-title">TuneFinder AI</h1>
        <p class="hero-text">
            Harnessing Machine Learning to decipher the genre of any song. 
            This project analyzes audio's very fabric to distinguish between Blues, Rock, Classical, and more.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- ABOUT SECTION (UPDATED FOR FIXED LAYOUT) ---
st.markdown('<div class="about-section" id="about-the-project"></div>', unsafe_allow_html=True)

with st.container():
    # Modern header with gradient title
    st.markdown("""
        <div class="about-header">
            <h2 class="about-title">🎵 About The Project</h2>
            <p class="about-subtitle">
                From Raw Audio to Insightful Predictions — A showcase of AI-powered music genre classification using cutting-edge machine learning techniques.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Description text (now centered)
    st.markdown("""
    <p class="about-description">
        This project demonstrates a complete machine learning pipeline to classify music genres with exceptional precision. 
        We explored multiple innovative approaches to tackle the complex challenge of audio pattern recognition.
    </p>
    """, unsafe_allow_html=True)
    
    # Feature cards in a single row using Streamlit columns but with custom CSS
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">Tabular Feature Engineering</div>
            <div class="feature-description">
                Leveraging sophisticated audio features like MFCCs, Spectral Centroid, and Chroma Frequencies 
                to train high-performance models including XGBoost and custom Neural Networks.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">👁️</span>
            <div class="feature-title">Deep Learning Vision</div>
            <div class="feature-description">
                Transforming audio signals into visual spectrograms and applying state-of-the-art CNNs 
                with Transfer Learning techniques for advanced genre detection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tech stack section (centered)
    st.markdown("""
    <div class="tech-stack">
        <div class="tech-title">Technologies Used</div>
        <div class="tech-badges">
            <span class="tech-badge">Python</span>
            <span class="tech-badge">TensorFlow</span>
            <span class="tech-badge">XGBoost</span>
            <span class="tech-badge">Librosa</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Scikit-learn</span>
        </div>
    </div>
    """, unsafe_allow_html=True)



# --- PREDICTION TOOL SECTION ---
st.markdown('<div class="prediction-section" id="predict-a-genre"></div>', unsafe_allow_html=True)

# Modern header with gradient title (matching About/EDA/Model sections)
st.markdown("""
    <div class="prediction-header">
        <h2 class="prediction-title">🎤 Predict a Genre</h2>
        <p class="prediction-subtitle">
            Choose your prediction method: upload raw audio files for tabular feature analysis or pre-generated spectrograms for deep learning vision models.
        </p>
    </div>
""", unsafe_allow_html=True)

# Enhanced tabs styling
tab1, tab2 = st.tabs(["🎵 Audio File Analysis", "🖼️ Spectrogram Analysis"])

# --- AUDIO PREDICTION TAB ---
with tab1:
    st.markdown("""
    <div class="prediction-tab-container">
        <div class="prediction-tab-header">
            <h3 class="prediction-tab-title">Upload Audio File (.wav format)</h3>
            <p class="prediction-tab-description">
                Upload a .wav audio file to extract sophisticated features like MFCCs, Spectral Centroid, and Chroma for tabular model predictions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    audio_file = st.file_uploader("", type="wav", key="audio_uploader", label_visibility="collapsed")

    if audio_file:
        # Audio player with enhanced container
        st.markdown('<div class="audio-player-container">', unsafe_allow_html=True)
        st.audio(audio_file, format='audio/wav')
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.spinner('🧠 Analyzing audio features and generating predictions...'):
            try:
                features_df = extract_features_from_audio(audio_file)
                scaled_features = preprocessors['scaler'].transform(features_df)
                
                # Predictions
                pred_xgb = models['xgb'].predict(scaled_features)[0]
                prob_xgb = models['xgb'].predict_proba(scaled_features).max() * 100
                genre_xgb = preprocessors['label_encoder'].inverse_transform([pred_xgb])[0]
                
                pred_dnn = np.argmax(models['dnn'].predict(scaled_features, verbose=0), axis=1)[0]
                prob_dnn = models['dnn'].predict(scaled_features, verbose=0).max() * 100
                genre_dnn = preprocessors['label_encoder'].inverse_transform([pred_dnn])[0]

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.stop()

        # Success message with modern styling
        st.markdown("""
        <div class="prediction-success">
            <span class="success-icon">✅</span>
            <span class="success-text">Audio Analysis Complete!</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced results section
        st.markdown("""
        <div class="prediction-results-header">
            <h3 class="results-title">Tabular Model Predictions</h3>
            <p class="results-subtitle">Feature-engineered models analyzing audio characteristics</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown(f"""
            <div class="prediction-result-card xgboost">
                <div class="result-card-header">
                    <span class="result-icon">🚀</span>
                    <span class="result-model">XGBoost</span>
                </div>
                <div class="result-prediction">{genre_xgb.capitalize()}</div>
                <div class="result-confidence">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {prob_xgb:.1f}%"></div>
                    </div>
                    <span class="confidence-text">{prob_xgb:.1f}% Confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="prediction-result-card dnn">
                <div class="result-card-header">
                    <span class="result-icon">🧠</span>
                    <span class="result-model">Deep Neural Network</span>
                </div>
                <div class="result-prediction">{genre_dnn.capitalize()}</div>
                <div class="result-confidence">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {prob_dnn:.1f}%"></div>
                    </div>
                    <span class="confidence-text">{prob_dnn:.1f}% Confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- IMAGE PREDICTION TAB ---
with tab2:
    st.markdown("""
    <div class="prediction-tab-container">
        <div class="prediction-tab-header">
            <h3 class="prediction-tab-title">Upload Spectrogram Image</h3>
            <p class="prediction-tab-description">
                Upload a spectrogram image (PNG, JPG, JPEG) to analyze visual patterns using CNN and Transfer Learning models.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    image_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="image_uploader", label_visibility="collapsed")

    if image_file:
        # Image display with enhanced container
        st.markdown('<div class="image-display-container">', unsafe_allow_html=True)
        st.image(image_file, caption="Uploaded Spectrogram", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.spinner('👁️ Analyzing spectrogram patterns and generating predictions...'):
            try:
                # Preprocess the image
                preprocessed_image = preprocess_image(image_file)
                class_names = preprocessors['class_names']
                
                # CNN Prediction
                pred_cnn_probs = models['cnn'].predict(preprocessed_image)[0]
                pred_cnn_idx = np.argmax(pred_cnn_probs)
                prob_cnn = pred_cnn_probs[pred_cnn_idx] * 100
                genre_cnn = class_names[pred_cnn_idx]
                
                # VGG16 Prediction
                pred_vgg16_probs = models['vgg16'].predict(preprocessed_image)[0]
                pred_vgg16_idx = np.argmax(pred_vgg16_probs)
                prob_vgg16 = pred_vgg16_probs[pred_vgg16_idx] * 100
                genre_vgg16 = class_names[pred_vgg16_idx]

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.stop()
        
        # Success message with modern styling
        st.markdown("""
        <div class="prediction-success">
            <span class="success-icon">✅</span>
            <span class="success-text">Image Analysis Complete!</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced results section
        st.markdown("""
        <div class="prediction-results-header">
            <h3 class="results-title">Image-Based Model Predictions</h3>
            <p class="results-subtitle">Deep learning models analyzing visual spectrogram patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown(f"""
            <div class="prediction-result-card cnn">
                <div class="result-card-header">
                    <span class="result-icon">🏗️</span>
                    <span class="result-model">Custom CNN</span>
                </div>
                <div class="result-prediction">{genre_cnn.capitalize()}</div>
                <div class="result-confidence">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {prob_cnn:.1f}%"></div>
                    </div>
                    <span class="confidence-text">{prob_cnn:.1f}% Confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="prediction-result-card vgg16">
                <div class="result-card-header">
                    <span class="result-icon">🎯</span>
                    <span class="result-model">Transfer Learning (VGG16)</span>
                </div>
                <div class="result-prediction">{genre_vgg16.capitalize()}</div>
                <div class="result-confidence">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {prob_vgg16:.1f}%"></div>
                    </div>
                    <span class="confidence-text">{prob_vgg16:.1f}% Confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
   # --- EDA GALLERY SECTION ---
st.markdown('<div class="eda-section" id="eda-gallery"></div>', unsafe_allow_html=True)

# Modern header with gradient title (matching About section)
st.markdown("""
    <div class="eda-header">
        <h2 class="eda-title">📊 EDA Gallery</h2>
        <p class="eda-subtitle">
            Visualizing the unique signatures of each music genre from the GTZAN dataset.
        </p>
    </div>
""", unsafe_allow_html=True)

plots_dir = os.path.join("outputs", "visualizations")

# This dictionary now correctly contains ALL 12 EDA plots, logically grouped.
plot_groups = {
    "Overall Dataset Structure": ["1_class_distribution.png"],
    "Single Audio File Deep Dive": [
        "2_waveform.png", "3_mel_spectrogram.png", "4_zcr.png", 
        "5_spectral_centroid.png", "6_mfcc.png", "7_chroma.png"
    ],
    "Feature Distributions Across All Genres": [
        "10_centroid_distribution_by_genre.png", 
        "11_rolloff_distribution_by_genre.png", 
        "12_zcr_distribution_by_genre.png"
    ]
}

# Dictionary of insights for each plot
insights = {
    "1_class_distribution.png": "Confirms the dataset is perfectly balanced with 100 samples per genre, ensuring fair model training.",
    "2_waveform.png": "Shows the amplitude of a classical audio signal over time, revealing its dynamic range.",
    "3_mel_spectrogram.png": "Visualizes the frequency content on a perceptual scale, a key input for our image-based models.",
    "4_zcr.png": "Highlights the rate of signal changes, often correlating with percussive or noisy sounds.",
    "5_spectral_centroid.png": "Indicates the 'brightness' of the sound. Classical music typically has a lower centroid.",
    "6_mfcc.png": "Represents the timbral quality of the audio, one of the most powerful features for classification.",
    "7_chroma.png": "Maps the harmonic content to the 12 musical pitch classes, useful for identifying melody and chords.",
    "10_centroid_distribution_by_genre.png": "Compares the 'brightness' across genres. Metal is bright, classical is not.",
    "11_rolloff_distribution_by_genre.png": "Shows the spectral shape. Genres like Hip-hop have a higher rolloff frequency.",
    "12_zcr_distribution_by_genre.png": "Shows that genres like Metal have a consistently higher Zero-Crossing Rate."
}

try:
    for i, (group_title, plot_files) in enumerate(plot_groups.items()):
        # Group title with modern styling
        st.markdown(f'<h3 class="eda-group-title">{group_title}</h3>', unsafe_allow_html=True)
        
        # Use the 2-column layout with better styling
        cols = st.columns(2, gap="large")
        col_idx = 0
        
        for plot_file in plot_files:
            plot_path = os.path.join(plots_dir, plot_file)
            if os.path.exists(plot_path):
                with cols[col_idx % 2]:
                    # Create a clean, descriptive caption
                    caption = plot_file.split('_', 1)[1].replace('.png', '').replace('_', ' ').capitalize()
                    st.markdown(f'<div class="plot-caption">{caption}</div>', unsafe_allow_html=True)
                    
                    # Display the image
                    st.image(plot_path, use_container_width=True)
                    
                    # Add the corresponding insight text with enhanced styling
                    insight_text = insights.get(plot_file, "")
                    st.markdown(f'<p class="eda-insight">{insight_text}</p>', unsafe_allow_html=True)

                col_idx += 1
        
        # Add separator between groups (except for the last one)
        if i < len(plot_groups) - 1:
            st.markdown('<hr class="eda-group-separator">', unsafe_allow_html=True)
        
except FileNotFoundError:
    st.warning(f"Visualizations directory not found. Please ensure EDA plots are saved correctly.")
# --- MODEL INSIGHTS SECTION (MATCHING EDA STRUCTURE) ---
st.markdown('<div class="model-section" id="model-insights"></div>', unsafe_allow_html=True)

# Modern header with gradient title (matching EDA section)
st.markdown("""
    <div class="model-header">
        <h2 class="model-title">🚀 Model Insights & Performance</h2>
        <p class="model-subtitle">
            Comparing the final performance and behavior of all trained models through comprehensive evaluation metrics.
        </p>
    </div>
""", unsafe_allow_html=True)

# Display the main comparison chart first
st.markdown('<h3 class="model-group-title">Final Performance Comparison</h3>', unsafe_allow_html=True)

comparison_chart_path = os.path.join("outputs", "visualizations", "18_final_model_comparison.png")
if os.path.exists(comparison_chart_path):
    st.image(comparison_chart_path, use_container_width=True)
    st.markdown("""
    <p class="model-insight comparison-insight">
        This comprehensive comparison reveals the effectiveness of different approaches: tabular models (XGBoost & DNN) 
        achieve superior accuracy through engineered features, while transfer learning dramatically outperforms 
        custom CNN training from scratch.
    </p>
    """, unsafe_allow_html=True)
else:
    st.warning("Final performance comparison chart not found.")

# Add separator
st.markdown('<hr class="model-group-separator">', unsafe_allow_html=True)

st.markdown('<h3 class="model-group-title">Model-Specific Confusion Matrices</h3>', unsafe_allow_html=True)

# Model-specific evaluation plots
model_plots = [
    "13_confusion_matrix_xgb.png", 
    "14_confusion_matrix_dnn.png",
    "16_confusion_matrix_cnn.png", 
    "17_confusion_matrix_transfer_learning.png"
]

# Dictionary of insights for model plots
model_insights = {
    "13_confusion_matrix_xgb.png": "XGBoost demonstrates robust performance with minimal confusion between genres, showcasing the power of gradient boosting for tabular feature classification.",
    "14_confusion_matrix_dnn.png": "The Deep Neural Network exhibits excellent accuracy with slightly different confusion patterns, particularly excelling at distinguishing complex genre boundaries.",
    "16_confusion_matrix_cnn.png": "The custom CNN struggles significantly, highlighting the challenge of training deep networks from scratch with limited data and the importance of proper architecture design.",
    "17_confusion_matrix_transfer_learning.png": "Transfer Learning with VGG16 shows remarkable improvement over custom CNN, demonstrating how pre-trained features can be effectively adapted for music genre classification."
}

# Plot display names
plot_display_names = {
    "13_confusion_matrix_xgb.png": "XGBoost Confusion Matrix",
    "14_confusion_matrix_dnn.png": "Deep Neural Network Confusion Matrix", 
    "16_confusion_matrix_cnn.png": "Custom CNN Confusion Matrix",
    "17_confusion_matrix_transfer_learning.png": "Transfer Learning (VGG16) Confusion Matrix"
}

# Display model plots in 2-column grid (exactly like EDA section)
cols = st.columns(2, gap="large")
col_idx = 0

for plot_file in model_plots:
    plot_path = os.path.join("outputs", "visualizations", plot_file)
    if os.path.exists(plot_path):
        with cols[col_idx % 2]:
            # Create a clean, descriptive caption
            caption = plot_display_names.get(plot_file, plot_file.replace('.png', '').replace('_', ' ').title())
            st.markdown(f'<div class="plot-caption">{caption}</div>', unsafe_allow_html=True)
            
            # Display the image
            st.image(plot_path, use_container_width=True)
            
            # Add the corresponding insight text
            insight_text = model_insights.get(plot_file, "")
            st.markdown(f'<p class="model-insight">{insight_text}</p>', unsafe_allow_html=True)

        col_idx += 1
# --- FINAL CONCLUSION & KEY INSIGHTS SECTION ---
st.markdown('<div class="conclusion-section" id="final-conclusion"></div>', unsafe_allow_html=True)

# Modern header with gradient title (matching other sections)
st.markdown("""
    <div class="conclusion-header">
        <h2 class="conclusion-title">💡 Final Conclusion & Key Insights</h2>
        <p class="conclusion-subtitle">
            Comparing methodologies and identifying the optimal approach for music genre classification through comprehensive analysis.
        </p>
    </div>
""", unsafe_allow_html=True)

# Key insights in modern card layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="conclusion-card">
        <span class="conclusion-icon">🧠</span>
        <div class="conclusion-card-title">The Power of Feature Engineering</div>
        <div class="conclusion-card-description">
            The tabular models (XGBoost and DNN) achieved exceptional performance with over 90% accuracy, 
            proving that manually extracted audio features like MFCCs, Chroma, and Spectral Centroid 
            create a highly effective classification strategy.
        </div>
        <div class="conclusion-highlight">
            <strong>Key Finding:</strong> Both models demonstrate high confidence and often agree on predictions, 
            confirming the reliability and computational efficiency of this approach.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="conclusion-card">
        <span class="conclusion-icon">👁️</span>
        <div class="conclusion-card-title">The Challenge of Raw Visual Data</div>
        <div class="conclusion-card-description">
            The custom CNN trained from scratch failed to learn effectively, while Transfer Learning 
            with VGG16 showed dramatic improvement, demonstrating the immense value of pre-trained models 
            for limited datasets.
        </div>
        <div class="conclusion-highlight">
            <strong>Key Finding:</strong> Pre-trained networks eliminate the need to learn basic visual patterns, 
            making Transfer Learning the only trustworthy choice for spectrogram-based predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add separator
st.markdown('<hr class="conclusion-separator">', unsafe_allow_html=True)

# Final recommendation section
st.markdown("""
<div class="recommendation-section">
    <div class="recommendation-title">🎯 Recommended Approach</div>
    <div class="recommendation-content">
        <div class="recommendation-item">
            <span class="recommendation-badge primary">For Production</span>
            <span class="recommendation-text">Use <strong>XGBoost or DNN models</strong> with engineered audio features for maximum accuracy and efficiency</span>
        </div>
        <div class="recommendation-item">
            <span class="recommendation-badge secondary">For Research</span>
            <span class="recommendation-text">Explore <strong>Transfer Learning approaches</strong> to leverage pre-trained visual models on spectrograms</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)