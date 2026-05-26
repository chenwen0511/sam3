"""
SAM 3 single-image segmentation — output format matches scripts/yolo_seg.py.

由 FoundationPose ``seg/sam3_seg.py`` 以子进程调用（``GENPOSE2_SAM3_INFER_SCRIPT`` 默认指向本文件）。

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

from sam3.agent.helpers.mask_overlap_removal import mask_iom
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEFAULT_CHECKPOINT_DIR = os.environ.get(
    "SAM3_CHECKPOINT_DIR", "/home/ubuntu/stephen/02-weight/sam3"
)
DEFAULT_CHECKPOINT = os.path.join(DEFAULT_CHECKPOINT_DIR, "sam3.pt")
INFER_SCRIPT_VERSION = "4"

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


def _bbox_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _refine_mask_logits(
    mask_logits: torch.Tensor,
    mask_threshold: float,
    fill_hole_area: int,
    sprinkle_area: int,
) -> np.ndarray:
    """Fill small holes / remove sprinkles via connected components (same as SAM tracker)."""
    m = mask_logits.unsqueeze(0).unsqueeze(0).float()
    if fill_hole_area > 0 or sprinkle_area > 0:
        try:
            from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores

            m = fill_holes_in_mask_scores(
                m,
                max_area=max(fill_hole_area, sprinkle_area, 1),
                fill_holes=fill_hole_area > 0,
                remove_sprinkles=sprinkle_area > 0,
                fill_hole_area=fill_hole_area,
                sprinkle_removal_area=sprinkle_area,
            )
        except Exception as exc:
            print(f"[sam3_seg_backend] fill_holes_in_mask_scores skipped: {exc}")
    return (m.squeeze() > mask_threshold).cpu().numpy().astype(bool)


def _refine_mask_bool_cv(
    mask: np.ndarray, fill_hole_area: int, sprinkle_area: int
) -> np.ndarray:
    """CPU fallback: OpenCV connected-components hole fill / sprinkle removal."""
    out = mask.astype(np.uint8)
    if fill_hole_area > 0:
        inv = 1 - out
        n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] <= fill_hole_area:
                out[labels == i] = 1
    if sprinkle_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        fg_total = max(int(out.sum()), 1)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= sprinkle_area and area < fg_total // 2:
                out[labels == i] = 0
    return out.astype(bool)


def _filter_overlapping_masks(
    masks: List[np.ndarray],
    scores: List[float],
    boxes_xywh: List[List[int]],
    iom_thresh: float,
) -> Tuple[List[np.ndarray], List[float], List[List[int]]]:
    """Greedy IoM NMS — same strategy as sam3.agent remove_overlapping_masks."""
    n = len(masks)
    if n <= 1:
        return masks, scores, boxes_xywh

    masks_t = torch.from_numpy(np.stack(masks)).bool()
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    kept_idx: List[int] = []
    kept_masks: List[torch.Tensor] = []

    for i in order:
        cand = masks_t[i].unsqueeze(0)
        if not kept_masks:
            kept_idx.append(i)
            kept_masks.append(masks_t[i])
            continue
        iom_vals = mask_iom(cand, torch.stack(kept_masks)).squeeze(0)
        if torch.any(iom_vals > iom_thresh):
            continue
        kept_idx.append(i)
        kept_masks.append(masks_t[i])

    kept_idx.sort()
    return (
        [masks[i] for i in kept_idx],
        [scores[i] for i in kept_idx],
        [boxes_xywh[i] for i in kept_idx],
    )


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
    iom_threshold: float = 0.30,
    fill_hole_area: int = 16,
    sprinkle_area: int = 16,
    postprocess: bool = True,
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

    masks_logits = output["masks_logits"]
    scores = output["scores"]
    raw_masks: List[np.ndarray] = []
    raw_scores: List[float] = []
    raw_boxes: List[List[int]] = []
    for i in range(len(scores)):
        score = float(scores[i].item())
        if score <= threshold:
            continue
        logit = masks_logits[i].squeeze(0)
        if postprocess:
            mask = _refine_mask_logits(
                logit, mask_threshold, fill_hole_area, sprinkle_area
            )
            if not mask.any():
                mask = _refine_mask_bool_cv(
                    (logit > mask_threshold).cpu().numpy(),
                    fill_hole_area,
                    sprinkle_area,
                )
        else:
            mask = (logit > mask_threshold).cpu().numpy().astype(bool)
        if not mask.any():
            continue
        raw_masks.append(mask)
        raw_scores.append(score)
        raw_boxes.append(_bbox_from_mask(mask))

    n_before = len(raw_masks)
    if postprocess and n_before > 1:
        raw_masks, raw_scores, raw_boxes = _filter_overlapping_masks(
            raw_masks, raw_scores, raw_boxes, iom_threshold
        )
        print(
            f"[sam3_seg_backend] overlap filter: {n_before} -> {len(raw_masks)} "
            f"(iom_threshold={iom_threshold})"
        )

    detections: List[Dict[str, object]] = []
    vis_instances: List[Tuple[np.ndarray, List[int], float]] = []
    for mask, score, bbox_xywh in zip(raw_masks, raw_scores, raw_boxes):
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
        f"mask_threshold={mask_threshold}, postprocess={postprocess})"
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
    parser.add_argument(
        "--iom-threshold",
        type=float,
        default=0.30,
        help="IoM threshold to drop overlapping instances (default: 0.30, same as agent)",
    )
    parser.add_argument(
        "--fill-hole-area",
        type=int,
        default=16,
        help="Max hole area to fill via connected components (0=disable)",
    )
    parser.add_argument(
        "--sprinkle-area",
        type=int,
        default=16,
        help="Max sprinkle area to remove via connected components (0=disable)",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable overlap removal and mask morphological cleanup",
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
        iom_threshold=args.iom_threshold,
        fill_hole_area=args.fill_hole_area,
        sprinkle_area=args.sprinkle_area,
        postprocess=not args.no_postprocess,
    )
    print(json_path)


if __name__ == "__main__":
    main()



