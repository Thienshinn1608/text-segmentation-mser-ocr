import cv2
import numpy as np

MIN_W = 30
MIN_H = 14
MAX_W_RATIO = 0.60
MAX_H_RATIO = 0.40

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
            # Kiểm tra độ chồng lấp theo chiều dọc và khoảng cách ngang
            if y_overlap(b, last) > 0.55 * h and (b[0] - last[2]) <= max(40, int(6 * h)):
                ln.append(b)
                placed = True
                break
        if not placed: lines.append([b])
    return [sorted(ln, key=lambda x: x[0]) for ln in lines]