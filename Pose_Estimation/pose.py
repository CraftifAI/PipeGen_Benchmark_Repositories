"""Standalone ONNX pose estimation. Everything lives in this folder.

Usage:
    python pose.py                        # input.gif  -> output.mp4
    python pose.py --input sample.png     # auto-writes sample_pose.png
    python pose.py --input my.mp4 --output annotated.mp4
    python pose.py --input 0 --show       # webcam

Defaults assume:
    model:  ./pose_estimation.onnx
    input:  ./input.gif
    output: ./output.mp4 (videos) or <name>_pose.<ext> (images)
"""

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models" , "pose_estimation.onnx")
DEFAULT_INPUT = os.path.join("0")
DEFAULT_OUTPUT = os.path.join("")

INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# 21 links: [paf_x_idx, paf_y_idx, kpt_a_idx, kpt_b_idx]  (verbatim from post_process.cpp)
TOPOLOGY = np.array([
    [0,1,15,13],[2,3,13,11],[4,5,16,14],[6,7,14,12],[8,9,11,12],
    [10,11,5,7],[12,13,6,8],[14,15,7,9],[16,17,8,10],[18,19,1,2],
    [20,21,0,1],[22,23,0,2],[24,25,1,3],[26,27,2,4],[28,29,3,5],
    [30,31,4,6],[32,33,17,0],[34,35,17,5],[36,37,17,6],[38,39,17,11],
    [40,41,17,12],
], dtype=np.int64)

LIMB_COLORS = [
    (0,255,0),(0,200,255),(255,0,0),(255,0,255),(0,255,255),
    (255,255,0),(128,0,255),(255,128,0),(0,128,255),(128,255,0),
    (255,0,128),(0,255,128),(200,100,50),(50,200,100),(100,50,200),
    (50,100,200),(200,50,100),(100,200,50),(255,200,200),(200,255,200),
    (200,200,255),
]


# ---------- ONNX inference ----------

class PoseSession:
    def __init__(self, onnx_path):
        avail = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
        self.s = ort.InferenceSession(onnx_path, providers=providers)
        self.in_name = self.s.get_inputs()[0].name
        outs = self.s.get_outputs()
        self.out_names = [o.name for o in outs]
        # Detect cmap (18 channels) vs paf (42 channels) by output shape.
        self.cmap_idx, self.paf_idx = 0, 1
        for i, o in enumerate(outs):
            if len(o.shape) >= 2 and o.shape[1] == 18:
                self.cmap_idx = i
            elif len(o.shape) >= 2 and o.shape[1] == 42:
                self.paf_idx = i

    def __call__(self, x_nchw):
        outs = self.s.run(self.out_names, {self.in_name: x_nchw})
        return outs[self.cmap_idx][0], outs[self.paf_idx][0]


def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)


# ---------- Post-processing (port of post_process.cpp) ----------

def find_peaks(cmap, threshold=0.1, window=5, max_count=20):
    C, H, W = cmap.shape
    pad = window // 2
    # Local-max NMS via dilation.
    kernel = np.ones((window, window), np.uint8)
    counts = np.zeros(C, np.int64)
    peaks = np.zeros((C, max_count, 2), np.int64)
    for c in range(C):
        m = cmap[c]
        dil = cv2.dilate(m, kernel)
        is_peak = (m == dil) & (m >= threshold)
        ys, xs = np.where(is_peak)
        if len(ys) == 0:
            continue
        order = np.lexsort((xs, ys))
        n = min(len(ys), max_count)
        peaks[c, :n, 0] = ys[order][:n]
        peaks[c, :n, 1] = xs[order][:n]
        counts[c] = n
    return counts, peaks


def refine_peaks(counts, peaks, cmap, window=5):
    C, H, W = cmap.shape
    w = window // 2
    refined = np.zeros((C, peaks.shape[1], 2), np.float32)
    for c in range(C):
        for p in range(counts[c]):
            i, j = peaks[c, p]
            wsum = ri = rj = 0.0
            for ii in range(i - w, i + w + 1):
                ii_idx = -ii if ii < 0 else (H - (ii - H) - 2 if ii >= H else ii)
                for jj in range(j - w, j + w + 1):
                    jj_idx = -jj if jj < 0 else (W - (jj - W) - 2 if jj >= W else jj)
                    weight = cmap[c, ii_idx, jj_idx]
                    ri += weight * ii
                    rj += weight * jj
                    wsum += weight
            if wsum > 0:
                ri /= wsum
                rj /= wsum
            refined[c, p, 0] = (ri + 0.5) / H
            refined[c, p, 1] = (rj + 0.5) / W
    return refined


def paf_score_graph(paf, counts, refined, n_samples=7, eps=1e-6):
    _, H, W = paf.shape
    K = TOPOLOGY.shape[0]
    max_count = refined.shape[1]
    sg = np.zeros((K, max_count, max_count), np.float32)
    for k in range(K):
        pi, pj, a_idx, b_idx = TOPOLOGY[k]
        paf_i, paf_j = paf[pi], paf[pj]
        for a in range(counts[a_idx]):
            pa_i = refined[a_idx, a, 0] * H
            pa_j = refined[a_idx, a, 1] * W
            for b in range(counts[b_idx]):
                pb_i = refined[b_idx, b, 0] * H
                pb_j = refined[b_idx, b, 1] * W
                di, dj = pb_i - pa_i, pb_j - pa_j
                norm = np.sqrt(di * di + dj * dj) + eps
                ui, uj = di / norm, dj / norm
                integral = 0.0
                for t in range(n_samples):
                    pr = t / n_samples
                    ti = int(pa_i + pr * di)
                    tj = int(pa_j + pr * dj)
                    if 0 <= ti < H and 0 <= tj < W:
                        integral += paf_i[ti, tj] * ui + paf_j[ti, tj] * uj
                sg[k, a, b] = integral / n_samples
    return sg


def assignment(score_graph, counts, threshold=0.1):
    K = TOPOLOGY.shape[0]
    max_count = score_graph.shape[1]
    conn = -np.ones((K, 2, max_count), np.int64)
    for k in range(K):
        a_idx, b_idx = TOPOLOGY[k, 2], TOPOLOGY[k, 3]
        nr, nc = int(counts[a_idx]), int(counts[b_idx])
        if nr == 0 or nc == 0:
            continue
        sub = score_graph[k, :nr, :nc]
        rows, cols = linear_sum_assignment(-sub)
        for i, j in zip(rows, cols):
            if sub[i, j] > threshold:
                conn[k, 0, i] = j
                conn[k, 1, j] = i
    return conn


def connect_parts(conn, counts, max_objects=100):
    K = TOPOLOGY.shape[0]
    C = len(counts)
    max_count = conn.shape[2]
    visited = np.zeros((C, max_count), bool)
    objects = -np.ones((max_objects, C), np.int64)
    n = 0
    for c in range(C):
        if n >= max_objects:
            break
        for i in range(counts[c]):
            if n >= max_objects:
                break
            queue, new_obj = [(c, i)], False
            while queue:
                cn, in_ = queue.pop(0)
                if visited[cn, in_]:
                    continue
                visited[cn, in_] = True
                new_obj = True
                objects[n, cn] = in_
                for k in range(K):
                    ca, cb = TOPOLOGY[k, 2], TOPOLOGY[k, 3]
                    if ca == cn:
                        ib = conn[k, 0, in_]
                        if ib >= 0:
                            queue.append((cb, int(ib)))
                    if cb == cn:
                        ia = conn[k, 1, in_]
                        if ia >= 0:
                            queue.append((ca, int(ia)))
            if new_obj:
                n += 1
    return objects[:n]


def parse(cmap, paf):
    counts, peaks = find_peaks(cmap)
    refined = refine_peaks(counts, peaks, cmap)
    sg = paf_score_graph(paf, counts, refined)
    conn = assignment(sg, counts)
    return connect_parts(conn, counts), refined


# ---------- Drawing ----------

def draw(frame, objects, peaks):
    h, w = frame.shape[:2]
    for obj in objects:
        for k in range(TOPOLOGY.shape[0]):
            a, b = TOPOLOGY[k, 2], TOPOLOGY[k, 3]
            if obj[a] >= 0 and obj[b] >= 0:
                pa, pb = peaks[a, obj[a]], peaks[b, obj[b]]
                p0 = (int(pa[1] * w), int(pa[0] * h))
                p1 = (int(pb[1] * w), int(pb[0] * h))
                cv2.line(frame, p0, p1, LIMB_COLORS[k], 3, cv2.LINE_AA)
        for c in range(len(obj)):
            if obj[c] >= 0:
                p = peaks[c, obj[c]]
                pt = (int(p[1] * w), int(p[0] * h))
                cv2.circle(frame, pt, 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)


# ---------- I/O ----------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def run_image(model, in_path, out_path):
    frame = cv2.imread(in_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")
    cmap, paf = model(preprocess(frame))
    objects, peaks = parse(cmap, paf)
    draw(frame, objects, peaks)
    cv2.imwrite(out_path, frame)
    print(f"Wrote {out_path} -- {len(objects)} people")


def run_video(model, source, out_path, show=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cmap, paf = model(preprocess(frame))
        objects, peaks = parse(cmap, paf)
        draw(frame, objects, peaks)
        cv2.putText(frame, f"Frame {n}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    dt = time.time() - t0
    if n:
        print(f"Processed {n} frames in {dt:.1f}s ({n/dt:.1f} FPS)")
    if out_path:
        print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to .onnx model.")
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Image / video path, or '0' for webcam.")
    p.add_argument("--output", default="",
                   help="Output file. Defaults to output.mp4 (video) or <name>_pose.<ext> (image).")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print(f"Loading {args.model} ...")
    model = PoseSession(args.model)
    print("Providers:", model.s.get_providers())

    src = args.input
    if src.isdigit():
        out = args.output  # optional for webcam
        run_video(model, int(src), out, show=args.show or not out)
        return

    if not os.path.exists(src):
        raise FileNotFoundError(f"Input not found: {src}")

    if src.lower().endswith(IMG_EXTS):
        out = args.output or os.path.splitext(src)[0] + "_pose.jpg"
        run_image(model, src, out)
    else:
        out = args.output or DEFAULT_OUTPUT
        run_video(model, src, out, show=args.show)


if __name__ == "__main__":
    main()
