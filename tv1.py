import cv2
import numpy as np

MAX_IMG_W = 1280

def resize_keep_ratio(img, max_w=MAX_IMG_W):
    h, w = img.shape[:2]
    if w <= max_w: return img, 1.0
    scale = max_w / float(w)
    nh = int(h * scale)
    return cv2.resize(img, (max_w, nh), interpolation=cv2.INTER_AREA), scale

def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def build_text_mask(gray):

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))
    
    black = cv2.addWeighted(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k1), 0.55, 
                            cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k2), 0.45, 0)
    top = cv2.addWeighted(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k1), 0.55, 
                          cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k2), 0.45, 0)
    
    feat = cv2.max(black, top)
    feat = cv2.GaussianBlur(feat, (3, 3), 0)
    
    _, bw = cv2.threshold(feat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))
    bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), 1)
    
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(gray.shape[1]*0.18), 1)))
    bw = cv2.subtract(bw, horiz)
    
    return bw

def normalize_text(s):
    if not s: return ""
    s = s.lower()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_gt_file(path):
    if not path or not os.path.exists(path): return ""
    texts = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 9:
                text = ",".join(parts[8:]).strip()
                if text != "###": texts.append(text)
    return " ".join(texts).strip()


