import streamlit as st
import joblib
import numpy as np
import io
from sklearn.linear_model import LogisticRegression

# Load models and vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")
models = {
    "Logistic Regression": joblib.load("models/logistic_model.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl")
}

# Page Config
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Initialize session state counters
if 'real_count' not in st.session_state:
    st.session_state['real_count'] = 0
if 'fake_count' not in st.session_state:
    st.session_state['fake_count'] = 0

# Custom CSS
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #0f1116;
            color: #f5f5f5;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .title-text {
            font-size: 3rem;
            font-weight: 700;
            color: #f56565;
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #f56565;
            margin-top: 2rem;
        }
        .confidence-bar {
            height: 30px;
            background-color: #2d3748;
            border-radius: 5px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            text-align: center;
            color: white;
            font-weight: bold;
            background-color: #48bb78;
        }
        .result-box {
            font-size: 1.2rem;
            font-weight: 600;
            padding: 1rem;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Instructions
st.sidebar.markdown("""
## 🛠️ Model Highlights
- ✅ **Bias Mitigation**
- 📊 **8,000 TF-IDF Features**
- 🤖 **Ensemble Learning Ready**

## 💡 How to Use
1. Paste news article in the text area
2. Choose model and click **Analyze**
3. Check **confidence level**
4. Review **model decision**
""")

# Title
st.markdown("<div class='title-text'>📰 AI-Powered Fake News Detector</div>", unsafe_allow_html=True)

# Layout Columns
col1, col2 = st.columns([2, 3])

# Input Section
with col1:
    st.markdown("<div class='section-title'>📄 Enter News Article</div>", unsafe_allow_html=True)
    default_text = """Russian energy giant Rosneft’s plan to sell its stake in India-based Nayara Energy Ltd. may be imperiled by fresh restrictions from the European Union.

The refinery, in which Rosneft has a 49.13% stake, will be targetted in the bloc’s 18th sanctions package over Moscow’s invasion of Ukraine, an EU official said in an X post on Friday."""
    if st.button("📥 Load Example Article"):
        st.session_state['text'] = default_text
    input_text = st.text_area("Paste the news article text here:", value=st.session_state.get('text', ''), height=200)
    st.success(f"✅ Text length: {len(input_text)} characters")

# Prediction and Analysis Section
with col2:
    st.markdown("<div class='section-title'>📊 Analysis Results</div>", unsafe_allow_html=True)
    model_name = st.selectbox("Choose Model:", list(models.keys()), index=0)

    if st.button("🔍 Analyze") and input_text.strip():
        vec_input = vectorizer.transform([input_text])

        try:
            model = models[model_name]
            prediction = model.predict(vec_input)[0]
            proba = model.predict_proba(vec_input)[0][prediction]
        except Exception:
            model = LogisticRegression()
            model.fit(vec_input, [0])
            prediction = 0
            proba = 0.5
            model_name = "Fallback Model"

        result_label = "✅ REAL NEWS" if prediction == 1 else "🚫 FAKE NEWS"
        result_color = "#48bb78" if prediction == 1 else "#f56565"

        # Update counters
        if prediction == 1:
            st.session_state['real_count'] += 1
        else:
            st.session_state['fake_count'] += 1

        st.markdown(f"""
        <div class='result-box' style='background-color:{result_color}; color:white;'>
            {result_label}
        </div>
        <p><strong>Confidence:</strong> {proba * 100:.2f}%</p>
        <p><strong>Model Used:</strong> {model_name}</p>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>🎯 Confidence Level</div>", unsafe_allow_html=True)
        st.markdown("""
            <div class='confidence-bar'>
                <div class='confidence-fill' style='width: {:.2f}%;'>{:.2f}%</div>
            </div>
        """.format(proba * 100, proba * 100), unsafe_allow_html=True)

        if proba < 0.6:
            st.info("🧐 This prediction has low confidence. Please cross-check the source.")

        report = f"Prediction: {result_label}\nConfidence: {proba * 100:.2f}%\nModel: {model_name}\n\nInput Text:\n{input_text}"
        st.download_button("Download Report", io.BytesIO(report.encode("utf-8")), file_name="prediction_report.txt")

    elif not input_text.strip():
        st.warning("⚠️ Please enter or load an article to analyze.")

# Display summary stats
st.sidebar.markdown(f"""
---
### 🔢 Prediction Stats (Session)
- 📰 Real News: {st.session_state['real_count']}
- ⛔ Fake News: {st.session_state['fake_count']}
""")
