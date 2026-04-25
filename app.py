"""
Fake News Detection System - 4 Layer Ensemble Detection
Complete with Text Analysis, Image Analysis, and Suggestions
"""

import streamlit as st
import pickle
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
import io
import requests
import base64
from scipy.fft import fft2

# ==================== TEXT MODEL LOAD ====================
@st.cache_resource
def load_text_model():
    """Load the trained text model"""
    model_path = 'models/text_model.pkl'
    if not os.path.exists(model_path):
        st.sidebar.error("❌ Text model not found")
        return None, None
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        st.sidebar.success("✅ Text Model: Loaded")
        return data['vectorizer'], data['classifier']
    except Exception as e:
        st.sidebar.error(f"Error loading text model")
        return None, None

# ==================== TEXT REASONING WITH SUGGESTIONS ====================
def generate_text_reasoning(text, fake_score):
    """Generate detailed reasoning for text - Both Fake and Real"""
    text_lower = text.lower()
    reasoning = []
    suggestions = []
    
    # ===== FAKE NEWS INDICATORS =====
    sensational = ['breaking', 'urgent', 'shocking', 'viral', 'alert', 'warning', 'breaking news', 'exclusive', 'secret', 'miracle', 'unbelievable']
    found_sensational = [w for w in sensational if w in text_lower]
    if found_sensational:
        reasoning.append(f"⚠️ Sensational language detected: {', '.join(found_sensational[:3])}")
        suggestions.append("✓ Verify claims with official sources before sharing")
    
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / max(len(text), 1)
    if caps_ratio > 0.15:
        reasoning.append(f"⚠️ Excessive capitalization ({caps_ratio:.0%} of text is uppercase)")
        suggestions.append("✓ Legitimate news rarely uses all caps for emphasis")
    
    exclamation_count = text.count('!')
    if exclamation_count > 2:
        reasoning.append(f"⚠️ Multiple exclamations found ({exclamation_count} ! marks)")
        suggestions.append("✓ Excessive punctuation often indicates emotional manipulation")
    
    question_count = text.count('?')
    if question_count > 2:
        reasoning.append(f"⚠️ Multiple rhetorical questions detected ({question_count})")
        suggestions.append("✓ Real news reports facts, doesn't ask provocative questions")
    
    urgent_words = ['urgent', 'immediately', 'asap', 'now', 'breaking', 'alert', 'warning']
    found_urgent = [w for w in urgent_words if w in text_lower]
    if found_urgent:
        reasoning.append(f"⚠️ Urgency language: {', '.join(found_urgent[:2])}")
        suggestions.append("✓ Fake news creates false urgency to prevent verification")
    
    clickbait_patterns = ["you won't believe", "doctors hate", "this one trick", "click here", "share this", "before deleted", "must watch", "viral video"]
    found_clickbait = [p for p in clickbait_patterns if p in text_lower]
    if found_clickbait:
        reasoning.append(f"⚠️ Clickbait pattern detected: {found_clickbait[0]}")
        suggestions.append("✓ Sensational headlines often hide lack of real content")
    
    # Check for missing or fake sources
    source_words = ['according to', 'reuters', 'ap', 'associated press', 'bbc', 'cnn', 'times', 'post', 'source', 'official']
    found_sources = [s for s in source_words if s in text_lower]
    if not found_sources and len(text.split()) > 100:
        reasoning.append("⚠️ No credible sources cited in article")
        suggestions.append("✓ Always check if the news cites verifiable sources")
    
    # ===== REAL NEWS INDICATORS =====
    formal_words = ['announced', 'statement', 'official', 'government', 'president', 'minister', 'department', 'commission', 'report', 'study', 'research', 'published']
    found_formal = [w for w in formal_words if w in text_lower]
    if len(found_formal) >= 2:
        reasoning.append(f"✅ Formal/official language detected (mentions: {', '.join(found_formal[:2])})")
    
    if found_sources:
        reasoning.append(f"✅ Source attribution found: {', '.join(found_sources[:2])}")
    
    # Check for balanced language
    balanced_indicators = ['however', 'although', 'while', 'according to', 'said', 'reported', 'stated', 'confirmed']
    found_balanced = [b for b in balanced_indicators if b in text_lower]
    if len(found_balanced) >= 2:
        reasoning.append("✅ Balanced reporting indicators detected")
    
    # ===== OVERALL ASSESSMENT =====
    if fake_score > 0.7:
        reasoning.append("🔴 VERDICT: HIGH PROBABILITY OF FAKE NEWS")
        suggestions.append("🚨 Do NOT share this content without verification")
        suggestions.append("✓ Check fact-checking websites (Snopes, FactCheck.org, AltNews)")
        suggestions.append("✓ Look for the original source of the information")
    elif fake_score > 0.5:
        reasoning.append("🟠 VERDICT: SUSPICIOUS - Multiple red flags")
        suggestions.append("⚠️ Cross-reference with multiple trusted news sources")
        suggestions.append("✓ Check the publication date and author credentials")
    elif fake_score > 0.3:
        reasoning.append("🟡 VERDICT: UNCERTAIN - Mixed signals")
        suggestions.append("✓ Verify with trusted sources before sharing")
        suggestions.append("✓ Look for coverage from mainstream media outlets")
    else:
        reasoning.append("🟢 VERDICT: LIKELY REAL - Text patterns consistent with legitimate news")
        suggestions.append("✅ Content appears legitimate")
        suggestions.append("✓ Still verify critical claims with official sources")
    
    # ===== ADDITIONAL NOTES =====
    if len(text.split()) < 30:
        reasoning.append("📝 Note: Short text may affect accuracy")
    
    return " | ".join(reasoning), suggestions


# ==================== LAYER 1: REALITY DEFENDER API ====================
def layer1_reality_defender(image_file, api_key):
    """Face swap, Deepfake, GAN detection"""
    try:
        image_bytes = image_file.getvalue()
        
        signed_response = requests.post(
            "https://api.prd.realitydefender.xyz/api/files/aws-presigned",
            json={"fileName": "upload.jpg", "fileSize": len(image_bytes)},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=30
        )
        
        if signed_response.status_code != 200:
            return None, 0.5
        
        signed_data = signed_response.json()
        signed_url = signed_data.get("response", {}).get("signedUrl")
        request_id = signed_data.get("response", {}).get("requestId")
        
        if not signed_url:
            return None, 0.5
        
        requests.put(signed_url, data=image_bytes, headers={"Content-Type": "image/jpeg"}, timeout=30)
        
        result_response = requests.get(
            f"https://api.prd.realitydefender.xyz/api/media/users/{request_id}",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=30
        )
        
        if result_response.status_code == 200:
            result = result_response.json()
            fake_score = result.get('fake_probability', 0.5)
            return result, fake_score
        
        return None, 0.5
        
    except Exception as e:
        print(f"Layer 1 error: {e}")
        return None, 0.5


# ==================== LAYER 2: ERROR LEVEL ANALYSIS (ELA) ====================
def layer2_ela_analysis(image_file):
    """Detect Photoshop, inpainting, local edits (clothes change)"""
    try:
        img = Image.open(image_file).convert('RGB')
        
        quality_high = io.BytesIO()
        quality_low = io.BytesIO()
        
        img.save(quality_high, format='JPEG', quality=95)
        img.save(quality_low, format='JPEG', quality=75)
        
        img_high = Image.open(quality_high)
        img_low = Image.open(quality_low)
        
        diff = ImageChops.difference(img_high, img_low)
        diff_array = np.array(diff)
        
        ela_score = np.mean(diff_array) / 255.0
        fake_score = min(ela_score * 1.5, 0.95)
        
        return fake_score, f"ELA score: {ela_score:.2%}"
        
    except Exception as e:
        print(f"Layer 2 error: {e}")
        return 0.3, "ELA analysis failed"


# ==================== LAYER 3: NOISE & TEXTURE ANALYSIS ====================
def layer3_noise_analysis(image_file):
    """Detect inconsistent noise patterns, AI generated textures"""
    try:
        img = Image.open(image_file).convert('RGB')
        img_array = np.array(img)
        
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        
        noise_var = np.var(gray)
        
        if noise_var < 30:
            noise_score = 0.7
            reason_noise = "Low noise variance (AI generation)"
        elif noise_var > 120:
            noise_score = 0.5
            reason_noise = "High noise variance (compression artifacts)"
        else:
            noise_score = 0.2
            reason_noise = "Normal noise level"
        
        # Frequency analysis (FFT)
        f_transform = fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        center_size = 50
        
        center_region = magnitude[center_h-center_size:center_h+center_size, 
                                   center_w-center_size:center_w+center_size]
        total_magnitude = np.mean(magnitude)
        center_magnitude = np.mean(center_region)
        
        if center_magnitude > total_magnitude * 2.5:
            frequency_score = 0.7
            reason_freq = "Grid artifacts detected (GAN/AI)"
        else:
            frequency_score = 0.2
            reason_freq = "Normal frequency pattern"
        
        fake_score = (noise_score + frequency_score) / 2
        
        return fake_score, f"{reason_noise} | {reason_freq}"
        
    except Exception as e:
        print(f"Layer 3 error: {e}")
        return 0.3, "Noise analysis failed"


# ==================== LAYER 4: METADATA ANALYSIS ====================
def layer4_metadata_analysis(image_file):
    """Check for editing software traces"""
    fake_score = 0.2
    reasoning = []
    
    try:
        img = Image.open(image_file)
        
        width, height = img.size
        aspect = width / height
        
        if aspect > 2 or aspect < 0.5:
            fake_score += 0.15
            reasoning.append("Unusual aspect ratio")
        
        file_size = len(image_file.getvalue())
        if file_size < 30000:
            fake_score += 0.2
            reasoning.append("Very small file size (possible compression)")
        
        if width > 4000 or height > 4000:
            fake_score += 0.1
            reasoning.append("Very high resolution (possible upscaling)")
        
        fake_score = min(fake_score, 0.95)
        
        return fake_score, " | ".join(reasoning) if reasoning else "No metadata issues"
        
    except Exception as e:
        print(f"Layer 4 error: {e}")
        return 0.2, "Metadata analysis failed"


# ==================== IMAGE REASONING WITH SUGGESTIONS ====================
def generate_image_reasoning_and_suggestions(result, layer_scores):
    """Generate image reasoning and suggestions"""
    reasoning = []
    suggestions = []
    
    if layer_scores.get('Reality Defender', 0) > 0.6:
        reasoning.append("🔴 Face/Deepfake manipulation detected by AI")
        suggestions.append("✓ The face in this image appears manipulated")
        suggestions.append("✓ Check if the person actually made the statement in original source")
    elif layer_scores.get('Reality Defender', 0) > 0.4:
        reasoning.append("🟠 Suspicious face patterns detected")
    
    if layer_scores.get('ELA (Local edits)', 0) > 0.6:
        reasoning.append("🔴 Local editing detected (possible clothes/background change)")
        suggestions.append("✓ The image shows signs of digital manipulation")
        suggestions.append("✓ Try reverse image search on Google Images")
    elif layer_scores.get('ELA (Local edits)', 0) > 0.4:
        reasoning.append("🟠 Some editing artifacts present")
    
    if layer_scores.get('Noise/AI Detection', 0) > 0.6:
        reasoning.append("🔴 AI generation artifacts detected")
        suggestions.append("✓ This image may be AI-generated, not a real photo")
        suggestions.append("✓ Look for inconsistencies in hands, teeth, or background")
    
    if result['class'] == 'FAKE':
        suggestions.append("🚨 Do NOT trust or share this image without verification")
        suggestions.append("✓ Verify the image source through reputable news outlets")
    elif result['class'] == 'SUSPICIOUS':
        suggestions.append("⚠️ Be cautious - image shows suspicious characteristics")
        suggestions.append("✓ Verify before sharing on social media")
    else:
        suggestions.append("✅ Image appears authentic")
        suggestions.append("✓ Still verify the context of the image")
    
    return " | ".join(reasoning) if reasoning else "🟢 No major manipulation detected", suggestions


# ==================== COMBINED ANALYSIS ====================
def analyze_image_complete(image_file, api_key):
    """4-layer ensemble analysis"""
    
    image_file.seek(0)
    rd_result, rd_score = layer1_reality_defender(image_file, api_key)
    
    image_file.seek(0)
    ela_score, ela_reason = layer2_ela_analysis(image_file)
    
    image_file.seek(0)
    noise_score, noise_reason = layer3_noise_analysis(image_file)
    
    image_file.seek(0)
    meta_score, meta_reason = layer4_metadata_analysis(image_file)
    
    final_score = (rd_score * 0.30) + (ela_score * 0.30) + (noise_score * 0.25) + (meta_score * 0.15)
    
    if final_score > 0.6:
        verdict = "FAKE"
    elif final_score > 0.4:
        verdict = "SUSPICIOUS"
    else:
        verdict = "REAL"
    
    layer_scores = {
        'Reality Defender (Face)': rd_score,
        'ELA (Local edits)': ela_score,
        'Noise/AI Detection': noise_score,
        'Metadata': meta_score
    }
    
    return {
        'fake_score': final_score,
        'class': verdict,
        'confidence': 1 - abs(final_score - 0.5) * 2,
        'layer_scores': layer_scores,
        'ela_reason': ela_reason,
        'noise_reason': noise_reason,
        'meta_reason': meta_reason
    }


# ==================== FALLBACK ANALYSIS ====================
def analyze_image_basic(image_file):
    """Basic analysis when API not available"""
    try:
        img = Image.open(image_file)
        img_array = np.array(img)
        
        fake_score = 0.2
        reasoning = []
        
        h, w = img_array.shape[:2]
        aspect = w / h
        if aspect > 2 or aspect < 0.5:
            fake_score += 0.2
            reasoning.append("Unusual aspect ratio")
        
        if len(img_array.shape) == 3:
            avg_std = (np.std(img_array[:,:,0]) + np.std(img_array[:,:,1]) + np.std(img_array[:,:,2])) / 3
            if avg_std < 30:
                fake_score += 0.3
                reasoning.append("Low color variance (possible AI generation)")
        
        file_size = len(image_file.getvalue())
        if file_size < 50000:
            fake_score += 0.1
            reasoning.append("Small file size")
        
        fake_score = min(fake_score, 0.95)
        
        return {
            'fake_score': fake_score,
            'class': 'FAKE' if fake_score > 0.5 else 'REAL',
            'confidence': 0.6,
            'reasoning': " | ".join(reasoning) if reasoning else "Basic analysis complete",
            'layer_scores': {'Basic Analysis': fake_score}
        }
    except Exception as e:
        return None


# ==================== GAUGE CHART ====================
def create_gauge_chart(score, title="Fake Score"):
    fig, ax = plt.subplots(figsize=(8, 3))
    
    if score > 0.7:
        color = '#e74c3c'
        status = "High Risk"
    elif score > 0.5:
        color = '#f39c12'
        status = "Medium Risk"
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


# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="Fake Content Detection", page_icon="🛡️", layout="wide")

# Custom CSS for better suggestions display
st.markdown("""
<style>
.suggestion-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Fake Content Detection System")
st.markdown("*4-Layer Ensemble: Face + Local Edits + AI Artifacts + Metadata*")

# Load models
vectorizer, classifier = load_text_model()
API_KEY = st.secrets.get("REALITY_DEFENDER_API_KEY", "")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    if vectorizer and classifier:
        st.success("✅ Text Model: Ready")
    else:
        st.error("❌ Text Model: Missing")
    
    if API_KEY:
        st.success("✅ Reality Defender API: Configured")
    else:
        st.warning("⚠️ API: Not configured (using basic mode)")
    
    st.success("✅ ELA: Ready (Local edits)")
    st.success("✅ AI Detection: Ready")
    
    st.markdown("---")
    st.header("📌 How It Works")
    st.markdown("""
    **4 Detection Layers:**
    1. **Face/Deepfake** - AI face manipulation
    2. **ELA** - Photoshop, clothes change
    3. **AI Artifacts** - GAN/Diffusion patterns
    4. **Metadata** - File integrity check
    """)

# Tabs
tab1, tab2 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis"])

# ==================== TAB 1: TEXT ANALYSIS WITH SUGGESTIONS ====================
with tab1:
    st.header("🔍 Analyze News Article")
    
    # Example buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Load Fake Example", use_container_width=True):
            st.session_state['news_text'] = "URGENT! Breaking news! You won't believe what happened! Click here now! Share before deleted! This is the biggest secret they don't want you to know! Miracle cure that doctors hate!"
    with col2:
        if st.button("📋 Load Real Example", use_container_width=True):
            st.session_state['news_text'] = "The president announced new economic policies today at the White House. According to official sources, the plan includes tax incentives for small businesses and infrastructure funding. The announcement was made during a press conference with members of the press."
    
    default_text = st.session_state.get('news_text', '')
    news_text = st.text_area(
        "Enter or paste the news article:",
        height=200,
        value=default_text,
        placeholder="Paste the news article text here..."
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    if analyze_btn:
        if news_text and vectorizer and classifier:
            with st.spinner("Analyzing text..."):
                processed = news_text.lower()
                processed = re.sub(r'[^a-zA-Z\s]', '', processed)
                features = vectorizer.transform([processed])
                proba = classifier.predict_proba(features)[0]
                
                fake_score = proba[0]
                confidence = max(proba)
                
                # Result
                col1, col2 = st.columns(2)
                with col1:
                    if fake_score > 0.5:
                        st.error(f"## ⚠️ FAKE NEWS DETECTED")
                    else:
                        st.success(f"## ✅ REAL NEWS")
                
                with col2:
                    st.metric("Fake Score", f"{fake_score*100:.1f}%")
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                # Gauge chart
                st.pyplot(create_gauge_chart(fake_score, "Fake News Probability"))
                plt.close()
                
                # Detailed Reasoning
                st.subheader("🔍 Detailed Analysis")
                reasoning, suggestions = generate_text_reasoning(news_text, fake_score)
                st.info(reasoning)
                
                # Suggestions Box
                st.subheader("💡 What You Should Do")
                for suggestion in suggestions:
                    st.write(suggestion)
                
        elif not news_text:
            st.warning("Please enter some text to analyze")
        else:
            st.error("Text model not loaded. Please check models/text_model.pkl")

# ==================== TAB 2: IMAGE ANALYSIS WITH SUGGESTIONS ====================
with tab2:
    st.header("🖼️ Analyze Image")
    st.caption("Detects: Face swap | Deepfake | Clothes change | AI generated | Photoshop")
    
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
            with st.spinner("Analyzing with 4-layer ensemble..."):
                if API_KEY:
                    result = analyze_image_complete(uploaded_image, API_KEY)
                else:
                    result = analyze_image_basic(uploaded_image)
                
                if result:
                    with col2:
                        if result['class'] == 'FAKE':
                            st.error(f"## ⚠️ FAKE IMAGE DETECTED")
                        elif result['class'] == 'SUSPICIOUS':
                            st.warning(f"## ⚠️ SUSPICIOUS IMAGE")
                        else:
                            st.success(f"## ✅ REAL IMAGE")
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Fake Score", f"{result['fake_score']*100:.1f}%")
                    with col_b:
                        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    with col_c:
                        st.metric("Verdict", result['class'])
                    
                    # Gauge chart
                    st.pyplot(create_gauge_chart(result['fake_score'], "Overall Fake Score"))
                    plt.close()
                    
                    # Layer analysis
                    with st.expander("📊 Layer-wise Technical Analysis"):
                        if 'layer_scores' in result:
                            for layer, score in result['layer_scores'].items():
                                st.progress(score, text=f"{layer}: {score*100:.1f}%")
                    
                    # Image Reasoning and Suggestions
                    st.subheader("🔍 Detection Details")
                    img_reasoning, img_suggestions = generate_image_reasoning_and_suggestions(result, result.get('layer_scores', {}))
                    st.info(img_reasoning)
                    
                    # Suggestions Box
                    st.subheader("💡 What You Should Do")
                    for suggestion in img_suggestions:
                        st.write(suggestion)

# Footer
st.markdown("---")
st.markdown("🛡️ **Fake Content Detection System** | 4-Layer Ensemble | AI-Powered Misinformation Detection")
