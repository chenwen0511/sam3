"""
SAM 3 single-image segmentation — output format matches scripts/yolo_seg.py.

Writes:
  {output_dir}/sam6d_results/detection_ism.json
  {output_dir}/sam6d_results/vis_ism.png  (unless --no-vis)

Example:
  python scripts/infer.py --image /path/to/rgb.png --output-dir outputs
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as cocomask

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEFAULT_CHECKPOINT_DIR = os.environ.get(
    "SAM3_CHECKPOINT_DIR", "/home/ubuntu/stephen/02-weight/sam3"
)
DEFAULT_CHECKPOINT = os.path.join(DEFAULT_CHECKPOINT_DIR, "sam3.pt")
INFER_SCRIPT_VERSION = "3"

_MODEL_CACHE: Dict[str, Sam3Processor] = {}

VIS_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 255, 128),
    (255, 64, 64),
    (64, 64, 255),
]


def _mask_to_rle(binary_mask: np.ndarray) -> Dict[str, object]:
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = cocomask.encode(mask)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def _xyxy_to_xywh(box: np.ndarray) -> List[int]:
    x1, y1, x2, y2 = box.tolist()
    return [
        int(round(x1)),
        int(round(y1)),
        int(round(max(0.0, x2 - x1))),
        int(round(max(0.0, y2 - y1))),
    ]


def _draw_detections_overlay(
    rgb_path: Path,
    instances: List[Tuple[np.ndarray, List[int], float]],
    output_path: Path,
    prompt: str,
) -> None:
    """Draw all instance masks/bboxes on the RGB image."""
    image = np.array(Image.open(rgb_path).convert("RGB"))
    overlay = image.copy()
    label = prompt.split()[0] if prompt else "obj"

    for idx, (mask, bbox_xywh, score) in enumerate(instances):
        color = VIS_COLORS[idx % len(VIS_COLORS)]
        color_arr = np.array(color, dtype=np.float32)
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * color_arr).astype(np.uint8)

        x, y, w, h = bbox_xywh
        x2, y2 = x + w, y + h
        cv2.rectangle(overlay, (x, y), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{label}#{idx} {score:.3f}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)


def resolve_checkpoint(checkpoint: str) -> str:
    path = os.path.abspath(checkpoint)
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ("sam3.pt", "model.safetensors"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint!r}. "
        f"Expected a .pt file or a directory containing sam3.pt"
    )


def _load_processor(checkpoint_path: str, device: str, threshold: float) -> Sam3Processor:
    cache_key = f"{checkpoint_path}|{device}|{threshold}"
    print(f"[sam3_seg_backend] load request: {checkpoint_path}")
    if cache_key not in _MODEL_CACHE:
        print(f"[sam3_seg_backend] loading SAM3 model: {checkpoint_path}")
        model = build_sam3_image_model(
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
        )
        _MODEL_CACHE[cache_key] = Sam3Processor(
            model, device=device, confidence_threshold=threshold
        )
        print(f"[sam3_seg_backend] model loaded and cached: {cache_key}")
    else:
        print(f"[sam3_seg_backend] model cache hit: {cache_key}")
    return _MODEL_CACHE[cache_key]


def run_sam3_segmentation(
    checkpoint_path: Path,
    rgb_path: Path,
    output_dir: Path,
    prompt: str = "white plate",
    threshold: float = 0.41,
    mask_threshold: float = 0.50,
    save_vis: bool = True,
) -> Path:
    """Run SAM3 text-prompt segmentation; writes all instances above threshold."""
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = resolve_checkpoint(str(checkpoint_path))

    t_load0 = time.perf_counter()
    processor = _load_processor(ckpt, device, threshold)
    t_load1 = time.perf_counter()
    print(f"[sam3_seg_backend] _load_processor elapsed_ms={(t_load1 - t_load0) * 1000:.3f}")

    image = Image.open(rgb_path).convert("RGB")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    t_pred0 = time.perf_counter()
    with autocast_ctx:
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)
    t_pred1 = time.perf_counter()
    print(f"[sam3_seg_backend] predict elapsed_ms={(t_pred1 - t_pred0) * 1000:.3f}")
    print(f"[sam3_seg_backend] load+predict elapsed_ms={(t_pred1 - t0) * 1000:.3f}")

    masks = output["masks_logits"] > mask_threshold
    scores = output["scores"]
    boxes = output["boxes"]
    detections: List[Dict[str, object]] = []
    vis_instances: List[Tuple[np.ndarray, List[int], float]] = []
    for i in range(len(scores)):
        mask = masks[i].squeeze(0).cpu().numpy().astype(bool)
        if not mask.any():
            continue
        score = float(scores[i].item())
        if score <= threshold:
            continue
        bbox_xywh = _xyxy_to_xywh(boxes[i].cpu().numpy())
        detections.append(
            {
                "scene_id": 0,
                "image_id": 0,
                "category_id": 1,
                "bbox": bbox_xywh,
                "score": score,
                "time": 0.0,
                "segmentation": _mask_to_rle(mask),
            }
        )
        vis_instances.append((mask, bbox_xywh, score))

    if not detections:
        raise RuntimeError(
            f"SAM3 returned no valid detections for prompt={prompt!r} "
            f"(threshold={threshold}, mask_threshold={mask_threshold})"
        )

    sam6d_results = output_dir / "sam6d_results"
    sam6d_results.mkdir(parents=True, exist_ok=True)
    json_path = sam6d_results / "detection_ism.json"
    json_path.write_text(json.dumps(detections), encoding="utf-8")
    print(
        f"[sam3_seg_backend] wrote {json_path} "
        f"({len(detections)} instance(s), threshold={threshold}, "
        f"mask_threshold={mask_threshold})"
    )
    for i, det in enumerate(detections):
        print(
            f"  [{i}] score={det['score']:.4f} bbox={det['bbox']}"
        )

    if save_vis:
        vis_path = sam6d_results / "vis_ism.png"
        _draw_detections_overlay(rgb_path, vis_instances, vis_path, prompt)
        print(f"[sam3_seg_backend] wrote {vis_path}")

    return json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM 3 segmentation (detection_ism.json, all instances above threshold)"
    )
    parser.add_argument("--image", type=str, required=True, help="RGB image path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output root (writes sam6d_results/detection_ism.json)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="white plate",
        help='Text prompt (default: "white plate")',
    )
    parser.add_argument(
        "--threshold",
        "--confidence",
        dest="threshold",
        type=float,
        default=0.41,
        help="Detection score threshold (--confidence is an alias)",
    )
    parser.add_argument(
        "--mask-threshold",
        "--mask_threshold",
        dest="mask_threshold",
        type=float,
        default=0.50,
        help="Mask binarization threshold",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to sam3.pt (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip writing sam6d_results/vis_ism.png",
    )
    return parser.parse_args()


def main():
    print(f"[sam3_seg_backend] infer.py v{INFER_SCRIPT_VERSION} path={__file__}")
    args = parse_args()
    rgb_path = Path(args.image).resolve()
    if not rgb_path.is_file():
        raise FileNotFoundError(f"Image not found: {rgb_path}")

    json_path = run_sam3_segmentation(
        checkpoint_path=Path(args.checkpoint),
        rgb_path=rgb_path,
        output_dir=Path(args.output_dir),
        prompt=args.prompt,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        save_vis=not args.no_vis,
    )
    print(json_path)


if __name__ == "__main__":
    main()



