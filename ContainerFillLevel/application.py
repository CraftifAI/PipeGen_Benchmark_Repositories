"""Single-image sanity test for monocular container fill-level estimation.

No GitHub repo, Hugging Face model, or dataset exists for "container fill
level" as a packaged task (searched both; nothing came up beyond unrelated
LoRA/image-gen models with "bin"/"dumpster" in the name). So this composes
a general building block instead of training anything: 77ukhtar/depth-
anything-v2-metric-onnx (same metric depth model used in the Parcel
Dimension Estimation use case), output predicted_depth in METRES.

Algorithm (this script only, not a trained fill-level model):
  1. Run metric depth on the whole frame.
  2. Take the median depth of each image row -> a depth profile from top to
     bottom of the frame.
  3. Smooth it, then find the largest jump across the smoothing window (the
     moving average spreads a real step over SMOOTH_WINDOW rows, so the jump
     has to be measured over that same span, not row-to-row). A real fill boundary
     shows up as a genuine discontinuity: the packed material occupies a
     comparatively narrow, continuous depth band (it's all near-ish and
     touching), while true empty space beyond the fill line jumps to a
     clearly different distance (an empty container's back wall/floor, or
     open sky/background above a partially-filled bin). This is checked
     against a relative threshold (25% of the frame's total depth span) so
     ordinary smooth perspective gradients within the material itself
     aren't mistaken for a fill boundary.
  4. If a jump clears the threshold, the two sides of that row are compared
     by mean depth -- the nearer side is "filled", the farther side is
     "empty" -- and fill % = filled rows / total rows.
  5. If no jump clears the threshold, the whole frame is one continuous
     depth band with no visible empty region -> reported as ~100% full
     (verified on the bundled sample: a truck bed packed with bottles to
     the rim gives a smooth gradient and no discontinuity, correctly
     reading as full).

This is a heuristic on top of a general depth model, not a calibrated
volumetric measurement: it has no notion of the container's true empty-
state geometry, assumes the frame is filled edge-to-edge by the container
opening (no unrelated background creeping in), and inherits whatever error
monocular metric depth has on the material's surface (shiny/transparent
material -- like the clear plastic bottles in the bundled sample -- is a
known hard case for monocular depth). Good for a sanity check of the
compose-two-models approach; not for billing/compliance-grade readings
without validating against a container of known geometry.
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_MODEL_DIRS = ("model", "models", "weights", "onnx", ".")
DEFAULT_HF_REPO = "77ukhtar/depth-anything-v2-metric-onnx"

HF_REF = re.compile(
    r"^(?:https?://huggingface\.co/)?"
    r"(?P<repo>[\w.-]+/[\w.-]+)"
    r"(?:/(?:blob|resolve)/(?P<rev>[^/]+)/(?P<file>.+?))?/?$"
)


def looks_like_hf_ref(value):
    if Path(value).expanduser().exists():
        return False
    if value.startswith(("http://", "https://")):
        return "huggingface.co/" in value
    if value.startswith((".", "/", "~")):
        return False
    if value.endswith(".onnx") and "/blob/" not in value and "/resolve/" not in value:
        return False
    return bool(HF_REF.match(value))


def fetch_from_hf(ref, local_dir="model"):
    from huggingface_hub import hf_hub_download, list_repo_files

    m = HF_REF.match(ref)
    repo_id, revision, filename = m.group("repo"), m.group("rev") or "main", m.group("file")
    token = os.environ.get("HF_TOKEN")
    files = list_repo_files(repo_id, revision=revision, token=token)
    if filename is None:
        onnx_files = [f for f in files if f.endswith(".onnx")]
        if not onnx_files:
            raise FileNotFoundError(f"{repo_id} has no .onnx.")
        if len(onnx_files) > 1:
            raise RuntimeError(f"{repo_id} has {len(onnx_files)} .onnx files — pick one with -m")
        filename = onnx_files[0]
    path = hf_hub_download(repo_id, filename, revision=revision, local_dir=local_dir, token=token)
    print(f"[model] fetched {repo_id}:{filename} -> {path}")
    return path


def resolve_model_path(explicit=None, search_dirs=DEFAULT_MODEL_DIRS):
    if explicit:
        if looks_like_hf_ref(explicit):
            return fetch_from_hf(explicit)
        p = Path(explicit).expanduser()
        if p.is_file():
            return str(p)
        if p.is_dir():
            search_dirs = (p,)
        else:
            raise FileNotFoundError(f"--model path does not exist: {p}")

    seen, candidates = set(), []
    for d in search_dirs:
        d = Path(d).expanduser()
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.onnx")):
            if any(part.startswith(".") for part in f.parts):
                continue
            rp = f.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            candidates.append(f)
        if candidates:
            break
    if not candidates:
        if DEFAULT_HF_REPO:
            return fetch_from_hf(DEFAULT_HF_REPO)
        raise FileNotFoundError("No .onnx found in: " + ", ".join(str(d) for d in search_dirs))
    if len(candidates) > 1:
        listing = "\n  ".join(f"{c}  ({c.stat().st_size / 1e6:.1f} MB)" for c in candidates)
        raise RuntimeError(f"Found {len(candidates)} .onnx files — pick one with -m:\n  {listing}")
    print(f"[model] auto-resolved: {candidates[0]}")
    return str(candidates[0])


# ---------------------------------------------------------------------------
DEPTH_SIZE = 518
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
JUMP_THRESHOLD_FRAC = 0.25   # min jump size, as a fraction of the frame's total depth span
SMOOTH_WINDOW = 15           # rows, odd-ish moving-average window to denoise the depth profile


class DepthEstimator:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        self.input_name = self.sess.get_inputs()[0].name

    def estimate(self, image):
        h, w = image.shape[:2]
        resized = cv2.resize(image, (DEPTH_SIZE, DEPTH_SIZE), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        blob = rgb.transpose(2, 0, 1)[None, ...].astype(np.float32)
        depth = self.sess.run(None, {self.input_name: blob})[0][0]  # [518, 518], metres
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)


def estimate_fill_level(depth_map):
    h = depth_map.shape[0]
    row_median = np.median(depth_map, axis=1)

    k = min(SMOOTH_WINDOW, h)
    kernel = np.ones(k) / k
    # replicate the edge rows before averaging: np.convolve(mode="same") zero-pads, which
    # ramps the first/last k//2 rows of the profile down towards 0 and invents a depth
    # discontinuity at the frame border that is not in the scene (it also inflates `span`).
    smoothed = np.convolve(np.pad(row_median, ((k - 1) // 2, k // 2), mode="edge"), kernel, mode="valid")

    span = smoothed.max() - smoothed.min()
    if span < 1e-6 or h < 2:
        return {"fill_pct": 100.0, "boundary_row": None, "profile": smoothed, "note": "flat depth profile"}

    # Measure the jump over the smoothing window, not row-to-row: a k-wide moving average
    # spreads a real step of height A across k rows, so no single row-to-row difference can
    # exceed A/k -- against a threshold of 0.25 * span (and span >= A) that test could never
    # fire, and every partially-filled container was reported as 100% full.
    step = min(k, h - 1)
    diffs = smoothed[step:] - smoothed[:-step]
    peak = int(np.argmax(np.abs(diffs)))
    jump_size = abs(diffs[peak])
    jump_idx = peak + step // 2          # boundary row ~= centre of the transition

    if jump_size < JUMP_THRESHOLD_FRAC * span:
        # no real discontinuity -- one continuous material band filling the whole frame
        return {"fill_pct": 100.0, "boundary_row": None, "profile": smoothed,
                 "note": "no depth discontinuity found -- frame reads as fully packed"}

    top_mean = smoothed[:jump_idx + 1].mean()
    bottom_mean = smoothed[jump_idx + 1:].mean()
    # the nearer (smaller-depth) side is the material; the farther side is empty space
    if top_mean <= bottom_mean:
        filled_rows = jump_idx + 1
        note = "material band at top of frame, empty space below"
    else:
        filled_rows = h - (jump_idx + 1)
        note = "material band at bottom of frame, empty space above"

    fill_pct = 100.0 * filled_rows / h
    return {"fill_pct": fill_pct, "boundary_row": jump_idx, "profile": smoothed, "note": note}


def write_image(path, img):
    # cv2.imwrite() just returns False when the target directory does not exist, which
    # otherwise leaves the script printing "wrote <path>" for a file that was never created.
    if not cv2.imwrite(path, img):
        raise IOError(f"could not write {path} (does its directory exist?)")


def draw_results(image, result, output_path):
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 1400

    if result["boundary_row"] is not None:
        y = result["boundary_row"]
        cv2.line(out, (0, y), (w, y), (0, 165, 255), max(2, int(4 * scale)))
        cv2.putText(out, "fill line", (10, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.8, 1.1 * scale), (0, 165, 255), max(2, int(3 * scale)), cv2.LINE_AA)

    fill_pct = result["fill_pct"]
    banner = f"CONTAINER FILL LEVEL: {fill_pct:.0f}%"
    banner_scale = max(1.6, 2.1 * scale)
    banner_thick = max(3, int(4 * scale))
    (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    pad = int(16 * scale) + 10
    cv2.rectangle(out, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
    if fill_pct >= 85:
        color = (0, 0, 255)
    elif fill_pct >= 50:
        color = (0, 200, 255)
    else:
        color = (0, 255, 0)
    cv2.putText(out, banner, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, banner_scale, color, banner_thick, cv2.LINE_AA)
    write_image(output_path, out)


def draw_depth_panel(depth_map, output_path):
    d = depth_map.astype(np.float32)
    norm = (d - d.min()) / (d.max() - d.min() + 1e-6) * 255
    colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_TURBO)
    write_image(output_path, colored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monocular container fill-level sanity test.")
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-o", "--output", default="output.jpg")
    parser.add_argument("-m", "--model", default=None,
                        help="Path/HF ref for the depth model. Omit to auto-discover.")
    parser.add_argument("--depth-panel", default=None,
                        help="Optional path to also save a colorized depth-map visualization")
    args = parser.parse_args()

    depth_model = DepthEstimator(resolve_model_path(args.model))

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    depth_map = depth_model.estimate(image)
    print(f"[depth ] map range: {depth_map.min():.2f}m - {depth_map.max():.2f}m")

    result = estimate_fill_level(depth_map)
    print(f"[fill  ] {result['fill_pct']:.1f}%  ({result['note']})")
    if result["boundary_row"] is not None:
        print(f"         boundary at row {result['boundary_row']} of {depth_map.shape[0]}")

    draw_results(image, result, args.output)
    print(f"[done  ] wrote {args.output}")

    if args.depth_panel:
        draw_depth_panel(depth_map, args.depth_panel)
        print(f"[done  ] wrote {args.depth_panel}")
