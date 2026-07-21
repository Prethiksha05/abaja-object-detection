"""
ABAJA Traffic Detector  –  Lag-free live detection for:
  Pedestrian | Car | Bicyclist | Two Wheeler | Bus | Truck | Cow
  Traffic Light (Red / Amber / Green)
  Sign Board / Speed Limit (OCR)
  Traffic Cone  (HSV orange colour detector)
  Steel Barricade (HSV orange+white stripe detector)

Controls:
  Q : Quit        H : Toggle HUD
"""

import re
import sys
import threading
import time

import cv2
import numpy as np
import torch
import easyocr
from ultralytics import YOLO


# ────────────────────────────────────────────────────────────────────────────
#  COCO class IDs  →  display labels + BGR colours
# ────────────────────────────────────────────────────────────────────────────
COCO_TARGETS = {
    0:  ("Pedestrian",  (30,  140, 255)),   # orange-ish
    1:  ("Bicyclist",   (255, 200,   0)),   # cyan-yellow
    2:  ("Car",         (255,  80,  80)),   # blue
    3:  ("Two Wheeler", (180,  50, 220)),   # purple
    5:  ("Bus",         (50,  160,  80)),   # green
    7:  ("Truck",       (80,  120, 180)),   # slate
    9:  ("Traffic Light", None),            # colour set per detection
    11: ("Sign Board",  (255,  40, 220)),   # magenta
    19: ("Cow",         (40,  130, 200)),   # brown-ish
}
ALLOWED_IDS = set(COCO_TARGETS.keys())

# Extra label colours
COLOR_RED    = (0,   30, 220)
COLOR_AMBER  = (0,  165, 255)
COLOR_GREEN  = (0,  200,  60)
COLOR_CONE   = (0,  140, 255)   # orange-ish in BGR
COLOR_BARRIER= (0,  215, 255)   # yellow-ish in BGR
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,     0,   0)

CONF_THRESH  = 0.35

# Speed limit values EasyOCR should look for
SPEED_NUMS = {10,15,20,25,30,40,50,60,70,80,90,100,110,120}


# ────────────────────────────────────────────────────────────────────────────
#  Threaded camera  –  always latest frame, no buffer lag
# ────────────────────────────────────────────────────────────────────────────
class ThreadedCamera:
    def __init__(self, src=0):
        print(f"Opening camera {src}…")
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened() and src != 0:
            print(f"  Source {src} unavailable → trying 0")
            self.cap = cv2.VideoCapture(0)
        self.ok = self.cap.isOpened()
        self.ret, self.frame = (False, None)
        if self.ok:
            self.ret, self.frame = self.cap.read()
        self.lock    = threading.Lock()
        self.stopped = False

    def start(self):
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True; break
            with self.lock:
                self.ret, self.frame = ret, frame
            time.sleep(0.008)

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.stopped = True
        self.cap.release()


# ────────────────────────────────────────────────────────────────────────────
#  Async YOLO inference thread
# ────────────────────────────────────────────────────────────────────────────
class InferenceThread:
    def __init__(self, model_path, device):
        print(f"Loading '{model_path}' on {device.upper()}…")
        self.model   = YOLO(model_path)
        self.device  = device
        self.pending = None
        self.results = None
        self.latency = 0.0
        self.inf_fps = 0.0
        self.busy    = False
        self.stopped = False
        self.lock    = threading.Lock()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        last = time.time()
        while not self.stopped:
            frame = None
            with self.lock:
                if not self.busy and self.pending is not None:
                    frame, self.pending = self.pending, None
            if frame is not None:
                self.busy = True
                t0 = time.time()
                try:
                    res = self.model(frame, conf=CONF_THRESH, verbose=False,
                                     device=self.device, classes=list(ALLOWED_IDS))
                    t1 = time.time()
                    with self.lock:
                        self.results = res[0]
                        self.latency = (t1 - t0) * 1000
                        self.inf_fps = 1.0 / (t1 - last) if (t1 - last) > 0 else 0
                        last = t1
                except Exception as e:
                    print(f"Inference error: {e}")
                finally:
                    self.busy = False
            else:
                time.sleep(0.005)

    def submit(self, frame):
        with self.lock:
            if not self.busy:
                self.pending = frame.copy()

    def get(self):
        with self.lock:
            return self.results, self.latency, self.inf_fps

    def stop(self):
        self.stopped = True


# ────────────────────────────────────────────────────────────────────────────
#  Traffic-light colour analysis  (top=Red, middle=Amber, bottom=Green)
# ────────────────────────────────────────────────────────────────────────────
def classify_traffic_light(frame, x1, y1, x2, y2):
    h = y2 - y1
    if h < 6:
        return "Unknown", COLOR_WHITE
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "Unknown", COLOR_WHITE
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    t3  = max(1, h // 3)
    sections = {
        "Red":   hsv[0:t3,    :],
        "Amber": hsv[t3:2*t3, :],
        "Green": hsv[2*t3:,   :],
    }
    scores = {n: float(np.mean(s[:,:,1]) * np.mean(s[:,:,2]))
              for n, s in sections.items() if s.size > 0}
    if not scores:
        return "Unknown", COLOR_WHITE
    state = max(scores, key=scores.get)
    col   = {"Red": COLOR_RED, "Amber": COLOR_AMBER, "Green": COLOR_GREEN}[state]
    return state, col


# ────────────────────────────────────────────────────────────────────────────
#  Traffic-cone detector  (HSV orange mask + contour shape filter)
# ────────────────────────────────────────────────────────────────────────────
def detect_cones(frame):
    """Returns list of (x1,y1,x2,y2) for probable traffic cones."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Orange range in HSV
    mask1 = cv2.inRange(hsv, (0,  150, 120), (15,  255, 255))
    mask2 = cv2.inRange(hsv, (160,150, 120), (180, 255, 255))
    mask  = cv2.bitwise_or(mask1, mask2)
    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:            # too small
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / max(w, 1)
        if 1.3 <= aspect <= 4.0:  # cone-like shape (taller than wide)
            boxes.append((x, y, x+w, y+h))
    return boxes


# ────────────────────────────────────────────────────────────────────────────
#  Speed Limit Sign detector (HSV red circle + contour shape filter)
# ────────────────────────────────────────────────────────────────────────────
def detect_speed_signs(frame):
    """Returns list of (x1,y1,x2,y2) for probable speed limit signs (red circles)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red/Orange range in HSV for the sign border
    mask1 = cv2.inRange(hsv, (0, 100, 100), (15, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    mask  = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / max(h, 1)
        # Circular shape check (aspect ratio near 1.0)
        if 0.7 <= aspect <= 1.3:
            boxes.append((x, y, x+w, y+h))
    return boxes


# ────────────────────────────────────────────────────────────────────────────
#  Steel-barricade detector  (orange + white alternating stripes / shape)
# ────────────────────────────────────────────────────────────────────────────
def detect_barricades(frame):
    """Returns list of (x1,y1,x2,y2) for probable barricades."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Orange parts of barricade
    mask_o1 = cv2.inRange(hsv, (0,  120, 100), (20,  255, 255))
    mask_o2 = cv2.inRange(hsv, (155,120, 100), (180, 255, 255))
    mask_o  = cv2.bitwise_or(mask_o1, mask_o2)
    # White parts
    mask_w  = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
    # Combined: must have both orange and white nearby
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    dil_o  = cv2.dilate(mask_o, kernel)
    dil_w  = cv2.dilate(mask_w, kernel)
    combined = cv2.bitwise_and(dil_o, dil_w)

    kernel2  = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if aspect >= 1.5:         # barricades are wider than tall
            boxes.append((x, y, x+w, y+h))
    return boxes


# ────────────────────────────────────────────────────────────────────────────
#  EasyOCR speed-limit reader  (runs async so it never stalls video)
# ────────────────────────────────────────────────────────────────────────────
_ocr_reader = None
_ocr_lock   = threading.Lock()

def _load_ocr():
    global _ocr_reader
    with _ocr_lock:
        if _ocr_reader is None:
            print("Loading EasyOCR…")
            _ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(),
                                          verbose=False)
            print("EasyOCR ready.")

class OCRWorker:
    def __init__(self):
        self.task    = None
        self.cache   = {}
        self.busy    = False
        self.stopped = False
        self.lock    = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        _load_ocr()   # pre-load
        while not self.stopped:
            task = None
            with self.lock:
                if not self.busy and self.task is not None:
                    task, self.task = self.task, None
            if task:
                frame, x1, y1, x2, y2, key = task
                self.busy = True
                result = self._read(frame, x1, y1, x2, y2)
                with self.lock:
                    self.cache[key] = result
                self.busy = False
            else:
                time.sleep(0.02)

    def _read(self, frame, x1, y1, x2, y2):
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            return None
        scale = max(1, 128 // max(crop.shape[:2]))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        try:
            res = _ocr_reader.readtext(crop, detail=0, allowlist='0123456789')
            for tok in res:
                for n in re.findall(r'\d+', tok.strip()):
                    if int(n) in SPEED_NUMS:
                        return n
        except Exception:
            pass
        return None

    def submit(self, frame, x1, y1, x2, y2):
        key = (x1//20, y1//20, x2//20, y2//20)
        with self.lock:
            cached = self.cache.get(key)
            if not self.busy:
                self.task = (frame, x1, y1, x2, y2, key)
        return cached

    def stop(self):
        self.stopped = True


# ────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ────────────────────────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, label, conf_str, color):
    """Draws a box with corner accents and a pill label."""
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    # Corner accents
    L = min(18, (x2-x1)//4, (y2-y1)//4)
    for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px,py), (px+dx*L, py), color, 4)
        cv2.line(frame, (px,py), (px, py+dy*L), color, 4)
    # Pill
    text = f"{label}  {conf_str}"
    sc, th = 0.52, 1
    (tw, txh), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
    py1 = max(0, y1 - txh - 8)
    ov  = frame.copy()
    cv2.rectangle(ov, (x1, py1), (x1+tw+10, y1), color, -1)
    cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, text, (x1+5, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, sc, COLOR_WHITE, th, cv2.LINE_AA)


def draw_tl_indicator(frame, cx, cy, r, col):
    """Glowing circle for traffic lights."""
    cv2.circle(frame, (cx,cy), r+4, COLOR_BLACK, -1)
    cv2.circle(frame, (cx,cy), r+2, col, -1)
    glow = frame.copy()
    cv2.circle(glow, (cx,cy), r+10, col, -1)
    cv2.addWeighted(glow, 0.2, frame, 0.8, 0, frame)


# ────────────────────────────────────────────────────────────────────────────
#  HUD overlay
# ────────────────────────────────────────────────────────────────────────────
LEGEND = [
    ("Pedestrian",   (30,  140, 255)),
    ("Bicyclist",    (255, 200,   0)),
    ("Car",          (255,  80,  80)),
    ("Two Wheeler",  (180,  50, 220)),
    ("Bus",          (50,  160,  80)),
    ("Truck",        (80,  120, 180)),
    ("Cow",          (40,  130, 200)),
    ("Traffic Cone", COLOR_CONE),
    ("Barricade",    COLOR_BARRIER),
    ("Sign Board",   (255,  40, 220)),
    ("🔴 Red",       COLOR_RED),
    ("🟡 Amber",     COLOR_AMBER),
    ("🟢 Green",     COLOR_GREEN),
]

def draw_hud(frame, cam_fps, inf_fps, latency, device, show_hud):
    if not show_hud:
        return
    h, w = frame.shape[:2]
    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,60), (10,10,10), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.line(frame, (0,60), (w,60), (55,55,55), 1)

    cv2.putText(frame, "ABAJA  Traffic Detector", (14,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,215,255), 2, cv2.LINE_AA)
    info = f"Device: {device.upper()}   Cam: {cam_fps:.0f} fps   Inf: {inf_fps:.0f} fps  ({latency:.0f} ms)"
    (iw,_),_ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(frame, info, (w-iw-10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,200), 1, cv2.LINE_AA)

    # Legend row
    x = 12; y = 56
    for txt, col in LEGEND:
        cv2.putText(frame, f"● {txt}", (x,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)
        (lw,_),_ = cv2.getTextSize(f"● {txt}", cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
        x += lw + 12
        if x > w - 80:
            x = 12; y += 14

    # Bottom bar
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h-26), (w, h), (10,10,10), -1)
    cv2.addWeighted(ov2, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, "  [Q] Quit    [H] Toggle HUD", (10, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1, cv2.LINE_AA)


# ────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--model",  default="yolo11n.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cam = ThreadedCamera(src=args.source).start()
    time.sleep(0.8)
    ret, frame = cam.read()
    if not ret or frame is None:
        print("ERROR: Cannot open camera."); cam.release(); sys.exit(1)

    inf = InferenceThread(args.model, device).start()
    ocr = OCRWorker()

    show_hud = True
    cam_fps  = 0.0
    fc       = 0
    fc_t     = time.time()

    print("\n" + "─"*60)
    print("  Detecting:")
    print("  Pedestrian | Bicyclist | Car | Two Wheeler | Bus | Truck")
    print("  Traffic Light (Red/Amber/Green) | Sign Board (OCR speed)")
    print("  Cow | Traffic Cone | Steel Barricade")
    print("─"*60)
    print("  [Q] Quit   [H] Toggle HUD\n")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.005); continue

            fc += 1
            t_now = time.time()
            if t_now - fc_t >= 1.0:
                cam_fps = fc / (t_now - fc_t)
                fc = 0; fc_t = t_now

            # ── Keep a clean copy for colour-based detectors ──────────────
            # IMPORTANT: run HSV detectors on the ORIGINAL frame (before any
            # annotation is drawn) so that drawn circles/boxes don't trigger
            # false positives in the cone / barricade colour detectors.
            orig = frame.copy()

            # ── YOLO inference (async) ───────────────────────────────────
            inf.submit(orig)
            results, latency, inf_fps = inf.get()

            # Collect YOLO bounding boxes so we can filter colour overlaps
            yolo_boxes = []

            if results is not None and results.boxes is not None:
                for box in results.boxes:
                    cls  = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    yolo_boxes.append((x1, y1, x2, y2))
                    label, color = COCO_TARGETS.get(cls, ("Object", COLOR_WHITE))

                    if cls == 9:  # Traffic Light – analyse colour on ORIG
                        state, color = classify_traffic_light(orig, x1, y1, x2, y2)
                        label = f"Traffic Light – {state}"

                    elif cls == 11:  # Sign Board → OCR for speed number
                        speed = ocr.submit(orig, x1, y1, x2, y2)
                        label = f"Speed Limit  {speed} km/h" if speed else "Sign Board"
                        color = (255, 40, 220)

                    draw_box(frame, x1, y1, x2, y2, label, f"{conf:.0%}", color)

            # ── Helper: check if a box heavily overlaps any YOLO box ──────
            def overlaps_yolo(bx1, by1, bx2, by2, threshold=0.4):
                """Return True if this box overlaps a YOLO box by > threshold."""
                for (ax1,ay1,ax2,ay2) in yolo_boxes:
                    ix1 = max(bx1, ax1); iy1 = max(by1, ay1)
                    ix2 = min(bx2, ax2); iy2 = min(by2, ay2)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    inter = (ix2-ix1) * (iy2-iy1)
                    area  = (bx2-bx1) * (by2-by1)
                    if area > 0 and inter / area > threshold:
                        return True
                return False

            # ── Speed Limit Sign (HSV on original frame + OCR) ───────────
            for (x1,y1,x2,y2) in detect_speed_signs(orig):
                if not overlaps_yolo(x1, y1, x2, y2):
                    speed = ocr.submit(orig, x1, y1, x2, y2)
                    label = f"Speed Limit  {speed} km/h" if speed else "Sign Board"
                    color = (255, 40, 220)
                    draw_box(frame, x1, y1, x2, y2, label, "", color)

            # ── Traffic Cone  (HSV on original frame) ────────────────────
            for (x1,y1,x2,y2) in detect_cones(orig):
                if not overlaps_yolo(x1, y1, x2, y2):
                    draw_box(frame, x1, y1, x2, y2, "Traffic Cone", "", COLOR_CONE)

            # ── Steel Barricade  (HSV on original frame) ─────────────────
            for (x1,y1,x2,y2) in detect_barricades(orig):
                if not overlaps_yolo(x1, y1, x2, y2):
                    draw_box(frame, x1, y1, x2, y2, "Steel Barricade", "", COLOR_BARRIER)

            # ── HUD ───────────────────────────────────────────────────────
            draw_hud(frame, cam_fps, inf_fps, latency, device, show_hud)

            cv2.imshow("ABAJA Traffic Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_hud = not show_hud

    finally:
        inf.stop(); ocr.stop(); cam.release()
        cv2.destroyAllWindows()
        print("Stopped.")

if __name__ == "__main__":
    main()
