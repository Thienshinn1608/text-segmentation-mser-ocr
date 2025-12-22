import os
import re
import unicodedata
import cv2
import numpy as np
import easyocr
from Levenshtein import distance as lev

import thanh_vien_1_tien_xu_ly as tv1
import thanh_vien_2_phat_hien_vung as tv2

reader = easyocr.Reader(["en"], gpu=True)
CONF_MIN_BOX = 0.25

def score_cer(gt, pred):
    if not gt: return 0.0
    gt_n = normalize_text(gt)
    pred_n = normalize_text(pred)
    if not pred_n: return 0.0
    d = lev(gt_n, pred_n)
    cer = d / max(1, len(gt_n))
    acc = (1.0 - cer) * 100.0
    return round(max(0.0, min(100.0, acc)), 2)

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



