import streamlit as st
import os, cv2, torch, clip, img2pdf, numpy as np, pytesseract, re, tempfile
from PIL import Image
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# --- CONFIG ---
st.set_page_config(page_title="Slide Extractor AI", layout="wide")

class AppConfig:
    out_dir = "extracted_slides"
    sample_rate = 1           
    clip_thresh = 0.88            
    ocr_change_thresh = 0.25      
    ssim_thresh = 0.90            
    min_stable_frames = 3         
    duplicate_similarity = 0.95   
    device = "cuda" if torch.cuda.is_available() else "cpu"

# --- CORE LOGIC (Modified for Streamlit) ---
class SlideExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        if not os.path.exists(cfg.out_dir): os.makedirs(cfg.out_dir)
        self.model, self.pre = clip.load("ViT-B/32", device=cfg.device)
        self.current_anchor = None
        self.best_frame = None
        self.stable_count = 0
        self.captures = []

    def extract_text(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        txt = pytesseract.image_to_string(gray, config="--psm 3")
        return set(re.sub(r"\W+", " ", txt.lower()).split())

    def get_features(self, frame, frame_num=0):
        h, w = frame.shape[:2]
        roi = frame[int(h*0.05):int(h*0.90), int(w*0.05):int(w*0.95)]
        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        inp = self.pre(pil).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            vis = self.model.encode_image(inp)
            vis /= vis.norm(dim=-1, keepdim=True)
        words = self.extract_text(roi)
        gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (640, 480))
        return {"visual": vis, "words": words, "gray": gray, "frame": frame.copy(), "frame_number": frame_num}

    def _is_new_slide(self, feat):
        if self.current_anchor is None: return True
        vis_sim = torch.mm(feat["visual"], self.current_anchor["visual"].T).item()
        w1, w2 = self.current_anchor["words"], feat["words"]
        text_sim = len(w1 & w2) / max(len(w1 | w2), 1) if w1 or w2 else 1.0
        s_sim, _ = ssim(self.current_anchor["gray"], feat["gray"], full=True)
        
        if vis_sim < self.cfg.clip_thresh or text_sim < (1 - self.cfg.ocr_change_thresh) or s_sim < self.cfg.ssim_thresh:
            return True
        return False

    def _commit_slide(self):
        if self.best_frame is None or self.stable_count < self.cfg.min_stable_frames: return
        for idx, existing in enumerate(self.captures):
            if torch.mm(self.best_frame["visual"], existing["visual"].T).item() > self.cfg.duplicate_similarity:
                self.captures[idx] = self.best_frame
                return
        self.captures.append(self.best_frame)

# --- UI INTERFACE ---
st.sidebar.title("Settings")
sample_rate = st.sidebar.slider("Sample Rate (seconds)", 0.5, 5.0, 1.0)
AppConfig.sample_rate = sample_rate

uploaded_file = st.file_uploader("Upload Lecture Video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    extractor = SlideExtractor(AppConfig)
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.info(f"Processing video at {fps:.2f} FPS...")
    progress_bar = st.progress(0)
    col1, col2 = st.columns(2)
    live_img = col1.empty()
    last_slide_img = col2.empty()

    fno = 0
    step = max(1, int(fps * AppConfig.sample_rate))

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            extractor._commit_slide()
            break
        
        feat = extractor.get_features(frame, fno)
        if extractor._is_new_slide(feat):
            extractor._commit_slide()
            extractor.current_anchor = feat
            extractor.best_frame = feat
            extractor.stable_count = 1
        else:
            extractor.stable_count += 1
            extractor.best_frame = feat
        
        # Update UI
        live_img.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processing...")
        if extractor.captures:
            last_slide_img.image(cv2.cvtColor(extractor.captures[-1]["frame"], cv2.COLOR_BGR2RGB), caption=f"Captured Slide {len(extractor.captures)}")
        
        fno += step
        progress_bar.progress(min(fno / total_frames, 1.0))

    cap.release()

    # Create PDF
    if extractor.captures:
        extractor.captures.sort(key=lambda x: x['frame_number'])
        paths = []
        for i, s in enumerate(extractor.captures):
            p = f"slide_{i:03d}.png"
            cv2.imwrite(p, s["frame"])
            paths.append(p)
        
        pdf_name = "Lecture_Slides.pdf"
        with open(pdf_name, "wb") as f:
            f.write(img2pdf.convert(paths))
        
        st.success(f"Done! Extracted {len(extractor.captures)} slides.")
        with open(pdf_name, "rb") as f:
            st.download_button("Download PDF", f, file_name=pdf_name)
        
        # Cleanup
        for p in paths: os.remove(p)
