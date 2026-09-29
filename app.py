import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #FF6B6B;
        text-align: center;
        font-size: 3rem !important;
        font-weight: bold !important;
    }
    .stSubheader {
        color: #4ECDC4;
        text-align: center;
    }
    .fake-box {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .real-box {
        background: linear-gradient(135deg, #4ECDC4 0%, #6EE7DE 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        border-left: 4px solid #FF6B6B;
    }
    .confidence-text {
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .probability-text {
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Paths
MODEL_DIR = "models"

LR_PATH = os.path.join(MODEL_DIR, "model_lr.pkl")
RF_PATH = os.path.join(MODEL_DIR, "model_rf.pkl")
NB_PATH = os.path.join(MODEL_DIR, "model_nb.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

@st.cache_resource
def load_models():
    required_files = [LR_PATH, RF_PATH, NB_PATH, VECTORIZER_PATH, ENCODER_PATH]
    missing_files = [file for file in required_files if not os.path.exists(file)]
    if missing_files:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing_files))

    with open(LR_PATH, "rb") as file:
        lr_model = pickle.load(file)
    with open(RF_PATH, "rb") as file:
        rf_model = pickle.load(file)
    with open(NB_PATH, "rb") as file:
        nb_model = pickle.load(file)
    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)
    with open(ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)

    return lr_model, rf_model, nb_model, vectorizer, label_encoder

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

def convert_label_to_text(prediction, label_encoder):
    try:
        prediction_value = int(prediction)
        return str(label_encoder.inverse_transform([prediction_value])[0]).lower()
    except Exception:
        return str(prediction).lower()

def get_probabilities(model, text_vector, label_encoder):
    probabilities = model.predict_proba(text_vector)[0]
    classes = list(model.classes_)

    fake_probability = 0.0
    real_probability = 0.0

    for class_value, probability in zip(classes, probabilities):
        label = convert_label_to_text(class_value, label_encoder)
        if label == "fake":
            fake_probability = float(probability)
        elif label == "real":
            real_probability = float(probability)

    return fake_probability, real_probability

def predict_article(title, article_text, model_name):
    combined_text = (preprocess_text(title) + " " + preprocess_text(article_text)).strip()
    if not combined_text:
        return None

    text_vector = vectorizer.transform([combined_text])

    if model_name == "Logistic Regression":
        model = lr_model
    elif model_name == "Random Forest":
        model = rf_model
    else:
        model = nb_model

    prediction = model.predict(text_vector)[0]
    prediction_label = convert_label_to_text(prediction, label_encoder)
    fake_probability, real_probability = get_probabilities(model, text_vector, label_encoder)

    return {
        "label": prediction_label,
        "fake_probability": fake_probability,
        "real_probability": real_probability
    }

# load models
try:
    lr_model, rf_model, nb_model, vectorizer, label_encoder = load_models()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("Copy model_lr.pkl, model_rf.pkl, model_nb.pkl, vectorizer.pkl, and label_encoder.pkl into the models folder.")
    st.stop()

# Sidebar
st.sidebar.title("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Classification Model:",
    ["Logistic Regression", "Random Forest", "Naive Bayes"],
    help="Choose model for fake/real classification"
)
st.sidebar.markdown("---")
st.sidebar.info("""
**About this app:**
- Detects fake vs real news
- Shows confidence score
- Shows probability for both fake and real
""")

# Main header
st.title("📰 Fake News Detector")
st.markdown("<h3 style='text-align: center; color: #666;'>AI-Powered News Classification System</h3>", unsafe_allow_html=True)
st.markdown("---")

# Input area
st.subheader("📝 Enter News Article")
title_input = st.text_input("Article Title:", placeholder="Enter the article title here...")
article_input = st.text_area("Article Content:", placeholder="Paste the article text here...", height=220)

if st.button("🔍 Analyze Article", use_container_width=True, type="primary"):
    if not title_input.strip() and not article_input.strip():
        st.warning("⚠️ Please enter a title or article text.")
    else:
        with st.spinner("🔄 Analyzing article..."):
            result = predict_article(title_input, article_input, model_choice)

            if result is None:
                st.error("❌ Could not process the article.")
            else:
                label = result["label"].upper()
                fake_probability = result["fake_probability"]
                real_probability = result["real_probability"]
                confidence = max(fake_probability, real_probability)

                st.markdown("---")
                st.subheader("🎯 Analysis Results")

                col1, col2 = st.columns(2)

                with col1:
                    if label == "FAKE":
                        st.markdown("""
                            <div class='fake-box'>
                                <h2 style='margin: 0;'>🔴 FAKE NEWS</h2>
                                <p style='margin: 1rem 0 0 0; font-size: 1rem;'>Article is likely fabricated or misleading</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class='real-box'>
                                <h2 style='margin: 0;'>🟢 REAL NEWS</h2>
                                <p style='margin: 1rem 0 0 0; font-size: 1rem;'>Article appears genuine</p>
                            </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.metric("Confidence Score", f"{confidence:.2%}")

                st.markdown("---")
                st.subheader("📊 Detailed Probabilities")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size: 1.5rem; color: #FF6B6B; font-weight: bold;'>FAKE</div>
                            <div class='confidence-text'>{fake_probability:.2%}</div>
                            <div class='probability-text'>Probability that the article is fake</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                        <div class='metric-card' style='border-left-color: #4ECDC4;'>
                            <div style='font-size: 1.5rem; color: #4ECDC4; font-weight: bold;'>REAL</div>
                            <div class='confidence-text'>{real_probability:.2%}</div>
                            <div class='probability-text'>Probability that the article is real</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.info(f"**Model Used:** {model_choice} | **Prediction:** {label} | **Confidence:** {confidence:.4f}")