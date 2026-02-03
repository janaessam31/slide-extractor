import streamlit as st
import cv2
import torch
import clip
import img2pdf
import numpy as np
import tempfile
import os
from PIL import Image
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Slide Extractor", layout="wide")
st.title("🎥 Smart Slide Extractor")
st.markdown("Upload a lecture video to automatically extract unique slides into a PDF.")

# --- SIDEBAR CONTROLS (Dynamic Thresholding) ---
st.sidebar.header("⚙️ Extraction Settings")

clip_thresh = st.sidebar.slider(
    "Visual Similarity (CLIP)", 0.70, 0.99, 0.92, 0.01,
    help="Higher = more sensitive to tiny visual changes. Lower = groups more frames together."
)

ink_sensitivity = st.sidebar.slider(
    "Ink Sensitivity (Adaptive)", 1, 50, 15,
    help="Higher = captures more subtle pen marks/text. Adjust if slides are too dark or too light."
)

sample_rate = st.sidebar.select_slider(
    "Sample Every (Seconds)", options=[0.5, 1.0, 2.0, 5.0], value=1.0
)

# --- CACHE MODEL ---
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_clip()

# --- CORE LOGIC ---
def get_features(frame, ink_c):
    # Adaptive Thresholding (Dynamic Ink Detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Block size 11, constant ink_c (from slider)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, ink_c
    )
    ink_count = np.sum(binary > 0)

    # CLIP Feature
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_input = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img_input)
        feat /= feat.norm(dim=-1, keepdim=True)
    
    return feat, ink_count

# --- UI LOGIC ---
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("🚀 Start Processing"):
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        slide_history = defaultdict(list)
        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_cols = st.columns(4)
        col_idx = 0
        
        last_gray = None
        
        for fno in range(0, total_frames, int(fps * sample_rate)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Fast SSIM check
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if last_gray is not None:
                score = ssim(last_gray, gray)
                if score > 0.99: continue
            last_gray = gray
            
            # 2. CLIP & Ink
            feat, ink = get_features(frame, ink_sensitivity)
            
            # 3. Match Logic
            is_new = True
            matched_sid = -1
            
            for sid, versions in slide_history.items():
                rep_feat = max(versions, key=lambda x: x['ink'])['feat']
                sim = torch.mm(feat, rep_feat.t()).item()
                if sim > clip_thresh:
                    is_new = False
                    matched_sid = sid
                    break
            
            if is_new:
                new_id = len(slide_history)
                slide_history[new_id].append({'feat': feat, 'ink': ink, 'frame': frame})
                with preview_cols[col_idx % 4]:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Slide {new_id}")
                col_idx += 1
            else:
                # Update if current frame has more 'content'
                if ink > max(v['ink'] for v in slide_history[matched_sid]):
                    slide_history[matched_sid].append({'feat': feat, 'ink': ink, 'frame': frame})
            
            progress_bar.progress(fno / total_frames)
            status_text.text(f"Processing Frame {fno}/{total_frames} | Found {len(slide_history)} slides")

        cap.release()
        
        # Save to PDF
        st.success("✅ Processing Complete!")
        pdf_paths = []
        for sid in sorted(slide_history.keys()):
            best = max(slide_history[sid], key=lambda x: x['ink'])
            img_path = f"slide_{sid}.png"
            cv2.imwrite(img_path, best['frame'])
            pdf_paths.append(img_path)
            
        with open("notes.pdf", "wb") as f:
            f.write(img2pdf.convert(pdf_paths))
            
        with open("notes.pdf", "rb") as f:
            st.download_button("📥 Download Final PDF", f, file_name="lecture_notes.pdf")
