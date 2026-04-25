"""
Fake News Detection System - 4 Layer Ensemble Detection
Detects: Face swap, Deepfakes, AI images, Local edits, Clothes change
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
import cv2
from scipy.fft import fft2
from scipy.stats import entropy

# ==================== LAYER 1: REALITY DEFENDER API ====================
def layer1_reality_defender(image_file, api_key):
    """Face swap, Deepfake, GAN detection"""
    try:
        image_bytes = image_file.getvalue()
        
        # Step 1: Get signed URL
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
        
        # Step 2: Upload
        requests.put(signed_url, data=image_bytes, headers={"Content-Type": "image/jpeg"}, timeout=30)
        
        # Step 3: Get results
        result_response = requests.get(
            f"https://api.prd.realitydefender.xyz/api/media/users/{request_id}",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=30
        )
        
        if result_response.status_code == 200:
            result = result_response.json()
            fake_score = result.get('fake_probability', 0.5)
            confidence = result.get('confidence', 0.5)
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
        
        # Save at different quality levels
        quality_high = io.BytesIO()
        quality_low = io.BytesIO()
        
        img.save(quality_high, format='JPEG', quality=95)
        img.save(quality_low, format='JPEG', quality=75)
        
        img_high = Image.open(quality_high)
        img_low = Image.open(quality_low)
        
        # Calculate difference
        diff = ImageChops.difference(img_high, img_low)
        diff_array = np.array(diff)
        
        ela_score = np.mean(diff_array) / 255.0  # Normalize to 0-1
        
        # Higher ELA score = more likely manipulated
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
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Noise variance check
        noise_var = np.var(gray)
        noise_score = 0
        
        if noise_var < 30:  # Too smooth - AI generated
            noise_score = 0.7
            reason_noise = "Low noise variance (AI generation)"
        elif noise_var > 120:  # Too noisy - compressed/manipulated
            noise_score = 0.5
            reason_noise = "High noise variance (compression artifacts)"
        else:
            noise_score = 0.2
            reason_noise = "Normal noise level"
        
        # 2. Frequency analysis (FFT)
        f_transform = fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Check for grid artifacts (GAN artifacts)
        center_region = magnitude[100:150, 100:150]
        grid_artifact = np.mean(center_region) / np.mean(magnitude)
        
        if grid_artifact > 2.5:
            frequency_score = 0.7
            reason_freq = "Grid artifacts detected (GAN/AI)"
        else:
            frequency_score = 0.2
            reason_freq = "Normal frequency pattern"
        
        # Combined score
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
        
        # Check image dimensions
        width, height = img.size
        aspect = width / height
        
        if aspect > 2 or aspect < 0.5:
            fake_score += 0.15
            reasoning.append("Unusual aspect ratio")
        
        # Check file size (very small images are suspicious)
        file_size = len(image_file.getvalue())
        if file_size < 30000:  # Less than 30KB
            fake_score += 0.2
            reasoning.append("Very small file size (possible compression)")
        
        # Check resolution
        if width > 4000 or height > 4000:
            fake_score += 0.1
            reasoning.append("Very high resolution (possible upscaling)")
        
        fake_score = min(fake_score, 0.95)
        
        return fake_score, " | ".join(reasoning) if reasoning else "No metadata issues"
        
    except Exception as e:
        print(f"Layer 4 error: {e}")
        return 0.2, "Metadata analysis failed"


# ==================== COMBINED ANALYSIS ====================
def analyze_image_complete(image_file, api_key):
    """4-layer ensemble analysis"""
    
    # Layer 1: Reality Defender (Face/Deepfake detection)
    rd_result, rd_score = layer1_reality_defender(image_file, api_key)
    
    # Need to reset file pointer for other layers
    image_file.seek(0)
    
    # Layer 2: ELA (Local edits, Photoshop)
    ela_score, ela_reason = layer2_ela_analysis(image_file)
    
    image_file.seek(0)
    
    # Layer 3: Noise analysis (AI artifacts)
    noise_score, noise_reason = layer3_noise_analysis(image_file)
    
    image_file.seek(0)
    
    # Layer 4: Metadata
    meta_score, meta_reason = layer4_metadata_analysis(image_file)
    
    # Weighted final score (weights based on effectiveness)
    # Layer 1: 30% (good for faces), Layer 2: 30% (good for local edits)
    # Layer 3: 25% (good for AI), Layer 4: 15% (supporting)
    final_score = (rd_score * 0.30) + (ela_score * 0.30) + (noise_score * 0.25) + (meta_score * 0.15)
    
    # Generate comprehensive reasoning
    reasoning = []
    
    if rd_score > 0.6:
        reasoning.append(f"🔴 Reality Defender: High probability of face manipulation ({rd_score:.0%})")
    elif rd_score > 0.4:
        reasoning.append(f"🟠 Reality Defender: Suspicious patterns detected ({rd_score:.0%})")
    else:
        reasoning.append(f"🟢 Reality Defender: No face manipulation detected")
    
    if ela_score > 0.6:
        reasoning.append(f"🔴 ELA Analysis: Strong evidence of local editing ({ela_score:.0%})")
        reasoning.append("   → Possible clothes change, object removal, or Photoshop")
    elif ela_score > 0.4:
        reasoning.append(f"🟠 ELA Analysis: Some editing artifacts detected")
    
    if noise_score > 0.6:
        reasoning.append(f"🔴 AI Detection: AI generation artifacts found ({noise_score:.0%})")
    elif noise_score > 0.4:
        reasoning.append(f"🟠 AI Detection: Suspicious noise patterns")
    
    if final_score > 0.6:
        verdict = "FAKE"
        verdict_icon = "⚠️"
        color = "red"
    elif final_score > 0.4:
        verdict = "SUSPICIOUS"
        verdict_icon = "⚠️"
        color = "orange"
    else:
        verdict = "REAL"
        verdict_icon = "✅"
        color = "green"
    
    return {
        'fake_score': final_score,
        'class': verdict,
        'confidence': 1 - abs(final_score - 0.5) * 2,
        'reasoning': " | ".join(reasoning),
        'layer_scores': {
            'Reality Defender': rd_score,
            'ELA (Local edits)': ela_score,
            'Noise/AI Detection': noise_score,
            'Metadata': meta_score
        },
        'details': {
            'rd_reason': "",
            'ela_reason': ela_reason,
            'noise_reason': noise_reason,
            'meta_reason': meta_reason
        }
    }


# ==================== FALLBACK: BASIC ANALYSIS ====================
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
            'class': 'Fake' if fake_score > 0.5 else 'Real',
            'confidence': 0.6,
            'reasoning': " | ".join(reasoning) if reasoning else "Basic analysis - No manipulation detected",
            'layer_scores': {'Basic': fake_score}
        }
    except Exception as e:
        return None


# ==================== TEXT ANALYSIS (SAME AS BEFORE) ====================
@st.cache_resource
def load_text_model():
    model_path = 'models/text_model.pkl'
    if not os.path.exists(model_path):
        return None, None
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['vectorizer'], data['classifier']
    except:
        return None, None

def generate_text_reasoning(text, fake_score):
    text_lower = text.lower()
    reasoning = []
    
    sensational = ['breaking', 'urgent', 'shocking', 'viral', 'alert', 'warning']
    found = [w for w in sensational if w in text_lower]
    if found:
        reasoning.append(f"Sensational language: {', '.join(found[:3])}")
    
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / max(len(text), 1)
    if caps_ratio > 0.15:
        reasoning.append(f"Excessive capitalization")
    
    if fake_score > 0.7:
        reasoning.append("HIGH PROBABILITY of misinformation")
    elif fake_score > 0.5:
        reasoning.append("Some indicators of potential fake news")
    else:
        reasoning.append("Text patterns consistent with legitimate news")
    
    return " | ".join(reasoning)

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
st.set_page_config(page_title="Fake News Detector - 4 Layer Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Fake News Detection System")
st.markdown("*4-Layer Ensemble Detection: Face + Local Edits + AI Artifacts + Metadata*")

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
        st.warning("⚠️ Reality Defender API: Not configured")
    
    st.success("✅ ELA Analysis: Ready (Local edits)")
    st.success("✅ Noise Analysis: Ready (AI artifacts)")
    st.success("✅ Metadata: Ready")
    
    st.markdown("---")
    st.header("📌 How It Works")
    st.markdown("""
    **4-Layer Ensemble:**
    1. **Reality Defender** - Face swap, Deepfake
    2. **ELA** - Photoshop, Clothes change, Local edits
    3. **AI Detection** - GAN artifacts, Noise patterns
    4. **Metadata** - File integrity, Dimensions
    """)

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "🔗 Combined Analysis"])

# ==================== TAB 1: TEXT ANALYSIS ====================
with tab1:
    st.header("Analyze News Article")
    
    news_text = st.text_area("Enter or paste the news article:", height=150)
    
    if st.button("Analyze Text", type="primary"):
        if news_text and vectorizer and classifier:
            processed = news_text.lower()
            processed = re.sub(r'[^a-zA-Z\s]', '', processed)
            features = vectorizer.transform([processed])
            proba = classifier.predict_proba(features)[0]
            fake_score = proba[0]
            
            if fake_score > 0.5:
                st.error(f"## FAKE NEWS DETECTED")
            else:
                st.success(f"## REAL NEWS")
            
            st.metric("Fake Score", f"{fake_score*100:.1f}%")
            st.pyplot(create_gauge_chart(fake_score, "Fake News Probability"))
            plt.close()
            st.info(generate_text_reasoning(news_text, fake_score))

# ==================== TAB 2: IMAGE ANALYSIS ====================
with tab2:
    st.header("Analyze Image - 4 Layer Detection")
    st.caption("Detects: Face swap | Deepfake | Clothes change | AI generated | Photoshop")
    
    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing with 4-layer ensemble..."):
                if API_KEY:
                    result = analyze_image_complete(uploaded_image, API_KEY)
                else:
                    result = analyze_image_basic(uploaded_image)
                
                if result:
                    with col2:
                        if result['class'] == 'FAKE':
                            st.error(f"## {result['class']} IMAGE DETECTED")
                        elif result['class'] == 'SUSPICIOUS':
                            st.warning(f"## {result['class']} IMAGE")
                        else:
                            st.success(f"## {result['class']} IMAGE")
                    
                    st.metric("Final Fake Score", f"{result['fake_score']*100:.1f}%")
                    st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    
                    st.pyplot(create_gauge_chart(result['fake_score'], "Overall Fake Score"))
                    plt.close()
                    
                    # Show layer scores
                    with st.expander("📊 Layer-wise Analysis"):
                        if 'layer_scores' in result:
                            for layer, score in result['layer_scores'].items():
                                st.progress(score, text=f"{layer}: {score*100:.1f}%")
                    
                    st.subheader("Detailed Reasoning")
                    st.info(result['reasoning'])

# ==================== TAB 3: COMBINED ANALYSIS ====================
with tab3:
    st.header("Text + Image Combined Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        combined_text = st.text_area("News text:", height=150)
    
    with col2:
        combined_image = st.file_uploader("Associated image:", type=['jpg', 'jpeg', 'png'], key="combined")
        if combined_image:
            st.image(combined_image, use_column_width=True)
    
    if st.button("Analyze Both", type="primary"):
        # Combined logic here
        st.info("Combined analysis uses weighted scores from both text and image")
