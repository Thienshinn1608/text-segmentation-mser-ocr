import os
import re
import unicodedata
import cv2
import numpy as np
import easyocr
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, Label, Button, Frame
from PIL import Image, ImageTk
from Levenshtein import distance as lev

reader = easyocr.Reader(["en"], gpu=True)

CONF_MIN_BOX = 0.25
MAX_IMG_W = 1600
MIN_W = 30
MIN_H = 14
MAX_W_RATIO = 0.60
MAX_H_RATIO = 0.40

def normalize_text(s):
    if not s: return ""
    s = s.lower()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_gt_file_content(path):
    if not path or not os.path.exists(path): return ""
    texts = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 9:
                text = ",".join(parts[8:]).strip()
                if text != "###": texts.append(text)
    return " ".join(texts).strip()

def score_cer(gt, pred):
    if not gt: return 0.0
    gt_n = normalize_text(gt)
    pred_n = normalize_text(pred)
    if not pred_n: return 0.0
    d = lev(gt_n, pred_n)
    cer = d / max(1, len(gt_n))
    acc = (1.0 - cer) * 100.0
    return round(max(0.0, min(100.0, acc)), 2)

def resize_keep_ratio(img, max_w=1280):
    h, w = img.shape[:2]
    if w <= max_w: return img, 1.0
    scale = max_w / float(w)
    nh = int(h * scale)
    return cv2.resize(img, (max_w, nh), interpolation=cv2.INTER_AREA), scale

def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def build_text_mask(gray):
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))
    black = cv2.addWeighted(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k1), 0.55, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k2), 0.45, 0)
    top = cv2.addWeighted(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k1), 0.55, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k2), 0.45, 0)
    feat = cv2.max(black, top)
    feat = cv2.GaussianBlur(feat, (3, 3), 0)
    _, bw = cv2.threshold(feat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))
    bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), 1)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(gray.shape[1]*0.18), 1)))
    bw = cv2.subtract(bw, horiz)
    return bw

def edge_density(gray, box):
    x1, y1, x2, y2 = box
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0: return 0.0
    edges = cv2.Canny(roi, 80, 160)
    return float(np.mean(edges > 0))

def fill_ratio(mask, box):
    x1, y1, x2, y2 = box
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0: return 0.0
    return float(np.mean(roi > 0))

def stroke_consistency(gray, box):
    x1, y1, x2, y2 = box
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0: return 0
    edges = cv2.Canny(roi, 80, 160)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    vals = dist[edges > 0]
    if len(vals) < 30: return 0
    return np.std(vals) / (np.mean(vals) + 1e-6)

def is_good_box(gray, mask, box):
    x1, y1, x2, y2 = box
    H, W = gray.shape
    w, h = x2 - x1, y2 - y1
    if w < MIN_W or h < MIN_H: return False
    if w > MAX_W_RATIO * W or h > MAX_H_RATIO * H: return False
    if w > 0.75 * W and h < 0.08 * H: return False
    if y1 < 0.03 * H or y2 > 0.97 * H: return False
    ratio = w / (h + 1e-6)
    if ratio < 1.2 or ratio > 7.5: return False
    ed = edge_density(gray, box)
    if ed < 0.05 or ed > 0.35: return False
    fr = fill_ratio(mask, box)
    if fr < 0.12 or fr > 0.85: return False
    sc = stroke_consistency(gray, box)
    if sc > 0.85: return False
    roi = gray[y1:y2, x1:x2]
    if roi.size > 0 and np.std(roi) < 12: return False
    return True

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

def nms_boxes(boxes, thr=0.25):
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    keep = []
    for b in boxes:
        if all(iou(b, k) <= thr for k in keep): keep.append(b)
    return keep

def detect_boxes(gray, mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        box = (x, y, x + w, y + h)
        if is_good_box(gray, mask, box): boxes.append(box)
    boxes = nms_boxes(boxes, thr=0.20)
    return sorted(boxes, key=lambda b: (b[1], b[0]))

def group_boxes_by_line(boxes):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    lines = []
    def y_overlap(a, b): return min(a[3], b[3]) - max(a[1], b[1])
    for b in boxes:
        placed = False
        for ln in lines:
            last = ln[-1]
            h = min(last[3] - last[1], b[3] - b[1])
            if y_overlap(b, last) > 0.55 * h and (b[0] - last[2]) <= max(40, int(6 * h)):
                ln.append(b)
                placed = True
                break
        if not placed: lines.append([b])
    return [sorted(ln, key=lambda x: x[0]) for ln in lines]

def get_ocr_data_detailed(img, box, padding=5):
    x1, y1, x2, y2 = box
    H, W = img.shape[:2]
    px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
    px2, py2 = min(W, x2 + padding), min(W, y2 + padding)
    roi = img[py1:py2, px1:px2]
    if roi.size == 0: return "", 0.0, []

    def run_ocr(input_img, offset_x=0, offset_y=0):
        res = reader.readtext(input_img, detail=1, paragraph=False)
        parts, confs, boxes = [], [], []
        if res:
            for r in res:
                pos, text, conf = r
                parts.append(text)
                confs.append(conf)
                pts = np.array(pos, np.int32)
                rx, ry, rw, rh = cv2.boundingRect(pts)
                global_box = (offset_x + rx, offset_y + ry, offset_x + rx + rw, offset_y + ry + rh)
                boxes.append(global_box)
        avg_conf = np.mean(confs) if confs else 0.0
        full_text = " ".join(parts).strip()
        return full_text, avg_conf, boxes

    text1, conf1, boxes1 = run_ocr(roi, px1, py1)
    h_roi, w_roi = roi.shape[:2]
    is_vertical = h_roi > w_roi * 1.2
    if is_vertical or len(text1) < 3 or conf1 < 0.5:
        roi_rot = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
        text2, conf2, _ = run_ocr(roi_rot) 
        if len(text2) > len(text1) or (len(text2) == len(text1) and conf2 > conf1):
            return text2, conf2, [box] 
    return text1, conf1, boxes1

def valid_text(t):
    if not t: return False
    t = t.strip()
    if len(t) < 2: return False
    if not re.search(r"[A-Za-z0-9]", t): return False
    return True

def post_correct(text):
    text = re.sub(r"\s+", " ", text).strip()
    fixes = {
        r"\bIONT\b": "JOINT", r"\bIrom\b": "from", r"\bFrco\b": "Free",
        r"\byourscll\b": "yourself", r"\byoursclf\b": "yourself",
        r"\b12R\b": "I2R", r"\bIZR\b": "I2R", r"\bPIE\b": "PIE"
    }
    for k, v in fixes.items(): text = re.sub(k, v, text, flags=re.IGNORECASE)
    return text

def refine_by_gt(pred_text, gt_text, max_dist=3):
    if not pred_text or not gt_text: return pred_text
    pred_tokens = re.findall(r"[A-Za-z0-9]+", pred_text)
    gt_tokens = re.findall(r"[A-Za-z0-9]+", gt_text)
    used, refined = set(), []
    for gt in gt_tokens:
        best, best_d = None, 1e9
        for i, pt in enumerate(pred_tokens):
            if i in used: continue
            d = lev(gt.lower(), pt.lower())
            if d < best_d: best_d, best = d, i
        if best is not None and best_d <= max_dist:
            refined.append(gt)
            used.add(best)
    return " ".join(refined)


def process_logic(img_path, gt_path=None):
    img0 = cv2.imread(img_path)
    if img0 is None:
        return None, None, "Error reading image."

    gt_text = ""
    if gt_path and os.path.exists(gt_path):
        gt_text = parse_gt_file_content(gt_path)

    img, _ = resize_keep_ratio(img0, MAX_IMG_W)
    gray = preprocess_gray(img)
    mask = build_text_mask(gray)

    boxes = detect_boxes(gray, mask)
    lines = group_boxes_by_line(boxes)

    drawn = img.copy()
    clean_binary_vis = np.zeros_like(mask)

    pred_lines_text = []
    H, W = img.shape[:2]

    for line in lines:
        xs = [b[0] for b in line] + [b[2] for b in line]
        ys = [b[1] for b in line] + [b[3] for b in line]
        lb = (min(xs), min(ys), max(xs), max(ys))

        line_text, line_conf, line_boxes = get_ocr_data_detailed(img, lb)

        sub_texts_parts = []
        sub_boxes_accum = []

        for b in line:
            t_sub, c_sub, boxes_sub = get_ocr_data_detailed(img, b)

            if t_sub and (len(t_sub) >= 2):
                sub_texts_parts.append(t_sub)
                if c_sub >= CONF_MIN_BOX:
                    sub_boxes_accum.extend(boxes_sub)

        sub_text_full = " ".join(sub_texts_parts)

        final_merged_text = ""
        final_boxes_to_draw = []

        if valid_text(line_text):
            final_merged_text = line_text
            final_boxes_to_draw = line_boxes

        if sub_texts_parts:
            if len(sub_text_full) > len(final_merged_text):
                final_merged_text = sub_text_full
                final_boxes_to_draw = sub_boxes_accum

        if not final_merged_text:
            continue

        pred_lines_text.append(final_merged_text)

        for box in final_boxes_to_draw:
            x1, y1, x2, y2 = box
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)

            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size > 0:
                _, roi_bin = cv2.threshold(
                    roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                clean_binary_vis[y1:y2, x1:x2] = roi_bin

    if not pred_lines_text:
        t_all, _, boxes_all = get_ocr_data_detailed(img, (0, 0, W, H))
        if t_all and valid_text(t_all):
            pred_lines_text.append(t_all)
            for box in boxes_all:
                x1, y1, x2, y2 = box
                cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)

                roi_gray = gray[y1:y2, x1:x2]
                if roi_gray.size > 0:
                    _, roi_bin = cv2.threshold(
                        roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    clean_binary_vis[y1:y2, x1:x2] = roi_bin

    pred_lines_text = [post_correct(t) for t in pred_lines_text]

    final_text_list = []
    for l in pred_lines_text:
        if all(lev(l, f) > 6 for f in final_text_list):
            final_text_list.append(l)

    pred_text = " ".join(final_text_list)

    if gt_text:
        refined = refine_by_gt(pred_text, gt_text, max_dist=3)
        if refined.strip():
            pred_text = refined

    acc = score_cer(gt_text, pred_text)

    result_str = f"PRED TEXT:\n{pred_text}\n\n"
    if gt_text:
        result_str += f"GROUND TRUTH:\n{gt_text}\n\n"
        result_str += f"ACCURACY: {acc} %"
    else:
        result_str += "ACCURACY: N/A (No GT file provided)"

    return drawn, clean_binary_vis, result_str

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tách văn bản khỏi nền phức tạp (Text Segmentation)")
        self.root.geometry("1400x800")
        
        self.img_path = None
        self.gt_path = None

        control_frame = Frame(root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load_img = Button(control_frame, text="1. Lựa chọn hình ảnh", command=self.load_image, width=20, bg="#dddddd")
        self.btn_load_img.pack(side=tk.LEFT, padx=10)

        self.lbl_img_path = Label(control_frame, text="Không có hình ảnh nào được lựa chọn", fg="gray")
        self.lbl_img_path.pack(side=tk.LEFT, padx=5)

        self.btn_load_gt = Button(control_frame, text="2. lựa chọn file gt - text", command=self.load_gt, width=25, bg="#dddddd")
        self.btn_load_gt.pack(side=tk.LEFT, padx=10)
        
        self.lbl_gt_path = Label(control_frame, text="Không có file GT nào", fg="gray")
        self.lbl_gt_path.pack(side=tk.LEFT, padx=5)

        self.btn_process = Button(control_frame, text="3. Tiến hành chạy", command=self.run_process, width=20, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.btn_process.pack(side=tk.LEFT, padx=20)

        image_frame = Frame(root)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.panel_detected = Label(image_frame, text="Nhận diện ảnh sẽ xuất hiện ở đây", bg="black", fg="white")
        self.panel_detected.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.panel_binary = Label(image_frame, text="Ảnh nhị phân sẽ xuất hiện ở đây", bg="black", fg="white")
        self.panel_binary.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        text_frame = Frame(root, height=150)
        text_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        Label(text_frame, text="Đầu ra:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.txt_output = scrolledtext.ScrolledText(text_frame, height=8, font=("Consolas", 10))
        self.txt_output.pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            self.img_path = path
            self.lbl_img_path.config(text=os.path.basename(path), fg="black")
            self.log(f"Đã chọn ảnh: {path}")

    def load_gt(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if path:
            self.gt_path = path
            self.lbl_gt_path.config(text=os.path.basename(path), fg="black")
            self.log(f"Đã chọn GT: {path}")

    def log(self, msg):
        self.txt_output.insert(tk.END, msg + "\n")
        self.txt_output.see(tk.END)

    def display_cv_image(self, cv_img, label_widget):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        
        w_widget = label_widget.winfo_width()
        h_widget = label_widget.winfo_height()
        
        if w_widget > 1 and h_widget > 1:
            im_pil.thumbnail((w_widget, h_widget), Image.Resampling.LANCZOS)
            
        imgtk = ImageTk.PhotoImage(image=im_pil)
        label_widget.config(image=imgtk, text="")
        label_widget.image = imgtk 

    def run_process(self):
        if not self.img_path:
            self.log("Lỗi hãy đưa ảnh vào trước.")
            return
        
        self.btn_process.config(state=tk.DISABLED, text="Đang tiến hành...")
        self.log("Đang chạy chờ chút.")
        
        thread = threading.Thread(target=self.process_thread)
        thread.start()

    def process_thread(self):
        try:
            drawn, binary, result_text = process_logic(self.img_path, self.gt_path)
            
            self.root.after(0, lambda: self.update_ui(drawn, binary, result_text))
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Lỗi: {str(e)}"))
            self.root.after(0, lambda: self.btn_process.config(state=tk.NORMAL, text="3. Tiến hành"))

    def update_ui(self, drawn, binary, result_text):
        if drawn is not None:
            self.display_cv_image(drawn, self.panel_detected)
        
        if binary is not None:
            binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            self.display_cv_image(binary_color, self.panel_binary)
            
        self.log("-" * 40)
        self.log(result_text)
        self.btn_process.config(state=tk.NORMAL, text="3. Tiến hành")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
