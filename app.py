import os
import cv2
import torch
import clip
import img2pdf
import pytesseract
import numpy as np
import logging
import streamlit as st
import tempfile
import shutil
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from difflib import SequenceMatcher

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- CONFIGURATION ---
class SlideExtractorConfig:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.pdf_path = os.path.join(output_dir, f"Final_Notes_{self.timestamp}.pdf")
        self.sample_rate_seconds = 1.0  
        self.min_stable_frames = 2
        self.visual_threshold = 0.92    
        self.phash_threshold = 8        
        self.text_sim_threshold = 0.85  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.edge_mask_percent = 0.08   
        self.ocr_lang = 'eng' 

# --- ENGINE ---
class SlideExtractor:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.config.device)
        
    def get_features(self, frame):
        h, w = frame.shape[:2]
        m_h, m_w = int(h * self.config.edge_mask_percent), int(w * self.config.edge_mask_percent)
        roi = frame[m_h:h-m_h, m_w:w-m_w]
        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        img_input = self.preprocess(pil_img).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            visual_feat = self.model.encode_image(img_input)
            visual_feat /= visual_feat.norm(dim=-1, keepdim=True)
        
        raw_text = pytesseract.image_to_string(roi, lang=self.config.ocr_lang)
        clean_text = " ".join(raw_text.split()).lower()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        phash = (resized[:, 1:] > resized[:, :-1]).flatten()
        return {'visual': visual_feat, 'phash': phash, 'text': clean_text, 'path': None}

    def is_duplicate(self, current_feat, saved_slides):
        if not saved_slides: return -1
        curr_text_alnum = "".join(filter(str.isalnum, current_feat['text']))
        for i in range(len(saved_slides) - 1, -1, -1):
            old = saved_slides[i]
            old_text_alnum = "".join(filter(str.isalnum, old['text']))
            if len(old_text_alnum) > 10 and len(curr_text_alnum) > 10:
                if old_text_alnum in curr_text_alnum or SequenceMatcher(None, old_text_alnum, curr_text_alnum).ratio() > self.config.text_sim_threshold:
                    return i
            dist = np.count_nonzero(current_feat['phash'] != old['phash'])
            vis_sim = torch.mm(current_feat['visual'], old['visual'].t()).item()
            if dist <= self.config.phash_threshold or vis_sim > self.config.visual_threshold:
                return i
        return -1

    def handle_stable_block(self, frame, fno, saved_slides):
        feat = self.get_features(frame)
        if len("".join(filter(str.isalnum, feat['text']))) < 10: return 
        idx = self.is_duplicate(feat, saved_slides)
        if idx != -1:
            if len(feat['text']) >= len(saved_slides[idx]['text']):
                path = saved_slides[idx]['path']
                saved_slides[idx].update(feat)
                saved_slides[idx]['path'] = path 
                cv2.imwrite(path, frame)
        else:
            s_idx = len(saved_slides)
            path = os.path.join(self.config.output_dir, f"slide_{s_idx:03d}.png")
            cv2.imwrite(path, frame)
            feat['path'] = path
            saved_slides.append(feat)

    def process(self, progress_callback=None):
        cap = cv2.VideoCapture(self.config.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved_slides, stability_buffer = [], []
        curr_fno = 0
        normal_step = max(1, int(fps * self.config.sample_rate_seconds))
        
        while curr_fno < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_fno)
            ret, frame = cap.read()
            if not ret: break
            if stability_buffer:
                s1 = cv2.resize(cv2.cvtColor(stability_buffer[-1], cv2.COLOR_BGR2GRAY), (64, 64))
                s2 = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
                if np.mean(cv2.absdiff(s1, s2)) < 4.0:
                    stability_buffer.append(frame)
                else:
                    if len(stability_buffer) >= self.config.min_stable_frames:
                        self.handle_stable_block(stability_buffer[-1], curr_fno, saved_slides)
                    stability_buffer = [frame]
            else:
                stability_buffer.append(frame)
            
            curr_fno += normal_step
            if progress_callback:
                progress_callback(curr_fno / total_frames)
        
        if len(stability_buffer) >= self.config.min_stable_frames:
            self.handle_stable_block(stability_buffer[-1], curr_fno, saved_slides)
        cap.release()
        return self.export_pdf(saved_slides)

    def export_pdf(self, saved_slides):
        if not saved_slides: return None
        paths = sorted([s['path'] for s in saved_slides])
        with open(self.config.pdf_path, "wb") as f:
            f.write(img2pdf.convert(paths))
        return self.config.pdf_path

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Slide Extractor", page_icon="📝")
st.title("📝 Video to PDF Slide Extractor")
st.write("Upload a video and I will use AI (CLIP + OCR) to extract unique slides.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Use temporary directories to avoid filling up the server
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "input_video.mp4")
        slides_dir = os.path.join(temp_dir, "slides")
        
        # Save upload to disk
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Start Extraction"):
            config = SlideExtractorConfig(video_path, slides_dir)
            extractor = SlideExtractor(config)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(p):
                progress_bar.progress(p)
                status_text.text(f"Processing: {int(p*100)}%")

            pdf_result = extractor.process(progress_callback=update_progress)
            
            if pdf_result and os.path.exists(pdf_result):
                st.success("Finished!")
                with open(pdf_result, "rb") as f:
                    st.download_button("Download PDF", f, file_name="Extracted_Slides.pdf")
            else:
                st.error("No slides detected. Try a different video.")
