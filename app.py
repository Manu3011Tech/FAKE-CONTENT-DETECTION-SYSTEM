"""
Fake News Detection System - Complete with Combined Analysis
"""

import streamlit as st
import pickle
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import torch
# import torch ke baad ye check add karo
print(f"PyTorch version: {torch.__version__}")
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fake News Detection System")
st.markdown("*AI-powered tool to detect misinformation in news articles and images*")

# ==================== LOAD TEXT MODEL ====================
@st.cache_resource
def load_text_model():
    """Load the trained text model"""
    model_path = 'models/text_model.pkl'
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ Text model not found")
        return None, None
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        st.sidebar.success("✅ Text Model: Loaded")
        return data['vectorizer'], data['classifier']
    except Exception as e:
        st.sidebar.error(f"Error loading text model")
        return None, None

# ==================== LOAD IMAGE MODEL ====================
@st.cache_resource
def load_image_model():
    """Load image model - Auto download if not exists"""
    try:
        # Pehle check karo model hai ya nahi
        model_path = 'models/image_model'
        processor_path = 'models/image_processor'
        
        if os.path.exists(model_path):
            # Local hai toh load karo
            processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True)
            model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
            st.sidebar.success("✅ Image Model: Loaded from local")
        else:
            # Nahi hai toh download karo (pehli baar time lagega)
            with st.spinner("Downloading image model (343MB)... First time takes 2-3 minutes"):
                processor = AutoImageProcessor.from_pretrained('dima806/deepfake_vs_real_image_detection')
                model = AutoModelForImageClassification.from_pretrained('dima806/deepfake_vs_real_image_detection')
                
                # Save locally for next time
                os.makedirs(model_path, exist_ok=True)
                os.makedirs(processor_path, exist_ok=True)
                processor.save_pretrained(processor_path)
                model.save_pretrained(model_path)
                st.sidebar.success("✅ Image Model: Downloaded and saved")
        
        model.eval()
        return processor, model
    except Exception as e:
        st.sidebar.warning(f"⚠️ Image Model: {str(e)[:50]}")
        return None, None

# ==================== IMAGE ANALYSIS FUNCTION ====================
def analyze_image_deep(image_file, processor, model):
    """Deep learning based image analysis"""
    try:
        img = Image.open(image_file).convert('RGB')
        inputs = processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            fake_prob = probabilities[0][1].item()
            real_prob = probabilities[0][0].item()
            
            reasoning = []
            if fake_prob > 0.7:
                reasoning.append("Deep learning model detected strong manipulation patterns")
            elif fake_prob > 0.5:
                reasoning.append("Model detected suspicious patterns in the image")
            else:
                reasoning.append("No significant manipulation detected by AI model")
            
            return {
                'fake_score': fake_prob,
                'real_score': real_prob,
                'class': 'Fake' if fake_prob > 0.5 else 'Real',
                'confidence': max(fake_prob, real_prob),
                'reasoning': " | ".join(reasoning)
            }
    except Exception as e:
        st.error(f"Image analysis error: {e}")
        return None

# ==================== TEXT REASONING ====================
def generate_text_reasoning(text, fake_score):
    """Generate detailed reasoning for text - Both Fake and Real"""
    text_lower = text.lower()
    reasoning = []
    
    # ===== FAKE NEWS INDICATORS =====
    sensational = ['breaking', 'urgent', 'shocking', 'viral', 'alert', 'warning', 'breaking news', 'exclusive', 'secret']
    found_sensational = [w for w in sensational if w in text_lower]
    if found_sensational:
        reasoning.append(f"⚠️ Sensational language detected: {', '.join(found_sensational[:3])}")
    
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / max(len(text), 1)
    if caps_ratio > 0.15:
        reasoning.append(f"⚠️ Excessive capitalization ({caps_ratio:.0%} of text is uppercase)")
    
    exclamation_count = text.count('!')
    if exclamation_count > 2:
        reasoning.append(f"⚠️ Multiple exclamations found ({exclamation_count} ! marks)")
    
    question_count = text.count('?')
    if question_count > 2:
        reasoning.append(f"⚠️ Multiple rhetorical questions detected")
    
    urgent_words = ['urgent', 'immediately', 'asap', 'now', 'breaking', 'alert']
    found_urgent = [w for w in urgent_words if w in text_lower]
    if found_urgent:
        reasoning.append(f"⚠️ Urgency language: {', '.join(found_urgent[:2])}")
    
    clickbait_patterns = ['you won\'t believe', 'doctors hate', 'this one trick', 'click here', 'share this', 'before deleted']
    found_clickbait = [p for p in clickbait_patterns if p in text_lower]
    if found_clickbait:
        reasoning.append(f"⚠️ Clickbait pattern detected: {found_clickbait[0]}")
    
    # ===== REAL NEWS INDICATORS =====
    formal_words = ['announced', 'statement', 'official', 'government', 'president', 'minister', 'department', 'commission', 'report', 'study', 'research', 'published']
    found_formal = [w for w in formal_words if w in text_lower]
    if len(found_formal) >= 2:
        reasoning.append(f"✅ Formal/official language detected (mentions: {', '.join(found_formal[:2])})")
    
    # Check for source attribution
    source_words = ['according to', 'reuters', 'ap', 'associated press', 'bbc', 'cnn', 'times', 'post', 'official', 'spokesperson']
    found_sources = [s for s in source_words if s in text_lower]
    if found_sources:
        reasoning.append(f"✅ Source attribution found: {', '.join(found_sources[:2])}")
    
    # Check for balanced language
    balanced_indicators = ['however', 'although', 'while', 'according to', 'said', 'reported', 'stated']
    found_balanced = [b for b in balanced_indicators if b in text_lower]
    if len(found_balanced) >= 2:
        reasoning.append(f"✅ Balanced reporting indicators detected")
    
    # ===== OVERALL ASSESSMENT =====
    if fake_score > 0.7:
        reasoning.append("🔴 VERDICT: HIGH PROBABILITY OF FAKE NEWS - Multiple red flags detected")
    elif fake_score > 0.5:
        reasoning.append("🟠 VERDICT: SUSPICIOUS - Some indicators of potential misinformation")
    elif fake_score > 0.3:
        reasoning.append("🟡 VERDICT: UNCERTAIN - Mixed signals, verify with trusted sources")
    else:
        reasoning.append("🟢 VERDICT: LIKELY REAL - Text shows patterns consistent with legitimate news")
    
    # ===== ADDITIONAL NOTES =====
    if len(text.split()) < 30:
        reasoning.append("📝 Note: Short text may affect accuracy")
    elif len(text.split()) > 500:
        reasoning.append("📝 Note: Long article analyzed completely")
    
    # Confidence note
    confidence = 1 - abs(fake_score - 0.5) * 2
    if confidence > 0.8:
        reasoning.append(f"🎯 High confidence prediction ({confidence*100:.0f}%)")
    
    return " | ".join(reasoning) if reasoning else "Analysis complete - No significant patterns detected"
# ==================== GAUGE CHART ====================
def create_gauge_chart(score, title="Fake Score"):
    """Create a gauge chart"""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    if score > 0.7:
        color = '#e74c3c'
        status = "⚠️ High Risk"
    elif score > 0.5:
        color = '#f39c12'
        status = "⚠️ Medium Risk"
    else:
        color = '#2ecc71'
        status = "Low Risk"
    
    ax.barh([0], [score], color=color, height=0.3)
    ax.barh([0], [1], alpha=0.2, color='gray', height=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(f"{title}: {score*100:.1f}%", fontsize=14, fontweight='bold')
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.text(score, 0.4, f"{score*100:.0f}%", ha='center', fontsize=12, fontweight='bold')
    ax.text(0.95, -0.3, status, ha='right', fontsize=10, color=color, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Load models
vectorizer, classifier = load_text_model()
image_processor, image_model = load_image_model()

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    if vectorizer and classifier:
        st.success("✅ Text Model: Ready")
    else:
        st.error("❌ Text Model: Missing")
    
    if image_processor and image_model:
        st.success("✅ Image Model: Ready")
    else:
        st.warning("⚠️ Image Model: Basic Mode")
    
    st.markdown("---")
    st.header("📌 How Combined Analysis Works")
    st.markdown("""
    **Weighted Scoring:**
    - Text Analysis: **60%**
    - Image Analysis: **40%**
    
    **Final Verdict:**
    - Combined Score > 0.5 → FAKE
    - Combined Score < 0.5 → REAL
    """)
    
    st.markdown("---")
    st.header("💡 Best Practice")
    st.info("For most accurate results, use Combined Analysis with both text and image from the same news article.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "🔗 Combined Analysis"])

# ==================== TAB 1: TEXT ANALYSIS ====================
with tab1:
    st.header("📝 Analyze News Article")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Load Fake Example", use_container_width=True):
            st.session_state['news_text'] = "URGENT! Breaking news! You won't believe what happened! Click here now! Share before deleted!"
    with col2:
        if st.button("📋 Load Real Example", use_container_width=True):
            st.session_state['news_text'] = "The president announced new economic policies today at the White House to help small businesses grow."
    
    default_text = st.session_state.get('news_text', '')
    news_text = st.text_area(
        "Enter or paste the news article:",
        height=150,
        value=default_text,
        placeholder="Paste your news article here..."
    )
    
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if news_text and vectorizer and classifier:
            with st.spinner("Analyzing text..."):
                processed = news_text.lower()
                processed = re.sub(r'[^a-zA-Z\s]', '', processed)
                features = vectorizer.transform([processed])
                proba = classifier.predict_proba(features)[0]
                
                fake_score = proba[0]
                real_score = proba[1]
                
                col1, col2 = st.columns(2)
                with col1:
                    if fake_score > 0.5:
                        st.error(f"## ⚠️ FAKE NEWS DETECTED")
                    else:
                        st.success(f"## ✅ REAL NEWS")
                
                with col2:
                    st.metric("Fake Score", f"{fake_score*100:.1f}%")
                    st.metric("Confidence", f"{max(proba)*100:.1f}%")
                
                st.pyplot(create_gauge_chart(fake_score, "Fake News Probability"))
                plt.close()
                
                st.subheader("🔍 Detailed Analysis")
                reasoning = generate_text_reasoning(news_text, fake_score)
                st.info(reasoning)
        else:
            st.warning("Please enter some text to analyze")

# ==================== TAB 2: IMAGE ANALYSIS ====================
with tab2:
    st.header("🖼️ Analyze Image for Manipulation")
    
    uploaded_image = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'webp'],
        key="image_upload"
    )
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            img = Image.open(uploaded_image)
            st.caption(f"📐 Dimensions: {img.size[0]} x {img.size[1]} pixels")
        
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image with Deep Learning AI..."):
                if image_processor and image_model:
                    result = analyze_image_deep(uploaded_image, image_processor, image_model)
                else:
                    st.error("Image model not loaded")
                    result = None
                
                if result:
                    with col2:
                        if result['class'] == 'Fake':
                            st.error(f"## ⚠️ FAKE IMAGE DETECTED")
                        else:
                            st.success(f"## ✅ REAL IMAGE")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fake Score", f"{result['fake_score']*100:.1f}%")
                    with col2:
                        st.metric("Real Score", f"{result['real_score']*100:.1f}%")
                    with col3:
                        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    
                    st.pyplot(create_gauge_chart(result['fake_score'], "Image Fake Score"))
                    plt.close()
                    
                    st.subheader("🔍 Analysis Details")
                    st.info(result['reasoning'])

# ==================== TAB 3: COMBINED ANALYSIS (TEXT + IMAGE) ====================
# ==================== TAB 3: COMBINED ANALYSIS (FIXED) ====================
with tab3:
    st.header("Combined Text + Image Analysis")
    st.caption("Upload both text and image from the same news article for the most accurate prediction")
    
    # Initialize session state
    if 'combined_text_input' not in st.session_state:
        st.session_state.combined_text_input = ""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("News Text")
        
        # Quick test buttons - FIXED
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Fake Example", use_container_width=True):
                st.session_state.combined_text_input = "URGENT! Breaking news! You won't believe what happened! This is the biggest secret they don't want you to know! Share before deleted!"
                st.rerun()
        with col_b:
            if st.button("Real Example", use_container_width=True):
                st.session_state.combined_text_input = "The government announced new economic policies today aimed at helping small businesses. The plan includes tax incentives and infrastructure funding."
                st.rerun()
        
        combined_text = st.text_area(
            "Paste the news article text here:",
            height=200,
            key="combined_text_input_widget",
            value=st.session_state.combined_text_input,
            placeholder="Enter the complete news article text..."
        )
    
    with col2:
        st.subheader("Associated Image")
        combined_image = st.file_uploader(
            "Upload the image associated with this news:",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="combined_image"
        )
        if combined_image:
            st.image(combined_image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Both (Text + Image)", type="primary", use_container_width=True):
        if combined_text and combined_image:
            if vectorizer and classifier and image_processor and image_model:
                with st.spinner("Analyzing both text and image together..."):
                    
                    # Text Analysis
                    processed = combined_text.lower()
                    processed = re.sub(r'[^a-zA-Z\s]', '', processed)
                    features = vectorizer.transform([processed])
                    proba = classifier.predict_proba(features)[0]
                    text_fake_score = proba[0]
                    text_confidence = max(proba)
                    
                    # Image Analysis
                    img_result = analyze_image_deep(combined_image, image_processor, image_model)
                    
                    if img_result:
                        image_fake_score = img_result['fake_score']
                        image_confidence = img_result['confidence']
                        
                        # Combined Score (Text 60%, Image 40%)
                        combined_fake_score = (text_fake_score * 0.6) + (image_fake_score * 0.4)
                        combined_confidence = (text_confidence * 0.6) + (image_confidence * 0.4)
                        
                        # Display Results
                        st.subheader("Combined Analysis Results")
                        
                        if combined_fake_score > 0.5:
                            st.error(f"## OVERALL VERDICT: FAKE NEWS")
                        else:
                            st.success(f"## OVERALL VERDICT: REAL NEWS")
                        
                        st.pyplot(create_gauge_chart(combined_fake_score, "Overall Fake Score"))
                        plt.close()
                        
                        st.subheader("Individual Analysis Breakdown")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Text Analysis", f"{text_fake_score*100:.1f}% fake")
                        with col2:
                            st.metric("Image Analysis", f"{image_fake_score*100:.1f}% fake")
                        with col3:
                            st.metric("Combined Score", f"{combined_fake_score*100:.1f}% fake")
                        
                        st.progress(combined_confidence)
                        st.caption(f"Overall Confidence: {combined_confidence*100:.1f}%")
                        
                        st.subheader("Detailed Analysis")
                        st.info(f"Text Analysis: {generate_text_reasoning(combined_text, text_fake_score)}")
                        st.info(f"Image Analysis: {img_result['reasoning']}")
                        
                        # Final Recommendation
                        st.subheader("Final Recommendations")
                        if combined_fake_score > 0.6:
                            st.warning("This content is likely FAKE. Do NOT share without verification.")
                        elif combined_fake_score > 0.4:
                            st.warning("This content is SUSPICIOUS. Verify before sharing.")
                        else:
                            st.success("This content appears REAL. Still verify critical claims.")
                    else:
                        st.error("Image analysis failed. Please try again.")
            else:
                st.error("Models not loaded properly. Please check system status in sidebar.")
        elif not combined_text:
            st.warning("Please enter some text to analyze")
        elif not combined_image:
            st.warning("Please upload an image to analyze")

# Footer
st.markdown("---")
st.markdown("🛡️ **Fake News Detection System** | Text + Image + Combined Analysis | Powered by AI")
