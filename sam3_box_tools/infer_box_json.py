#!/usr/bin/env python3
"""Run SAM3 geometric prompting from a compact multi-box JSON document."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


LOGGER = logging.getLogger("sam3_box_json")


class BoxSelectionError(ValueError):
    """Raised when no SAM3 candidate safely corresponds to an input box."""


@dataclass(frozen=True)
class BoxPrompt:
    """One validated positive geometric prompt."""

    box_id: int
    label: str
    bbox_xyxy: Tuple[float, float, float, float]


def validate_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> None:
    """Validate one xyxy box against image bounds."""
    x1, y1, x2, y2 = box
    if not (0.0 <= x1 < x2 <= float(width)):
        raise ValueError(f"box x coordinates outside image bounds: {box}")
    if not (0.0 <= y1 < y2 <= float(height)):
        raise ValueError(f"box y coordinates outside image bounds: {box}")


def xyxy_to_normalized_cxcywh(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> List[float]:
    """Convert an image-space xyxy box to normalized cxcywh."""
    validate_box(box, width, height)
    x1, y1, x2, y2 = box
    return [
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def box_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    intersection_x1 = max(box_a[0], box_b[0])
    intersection_y1 = max(box_a[1], box_b[1])
    intersection_x2 = min(box_a[2], box_b[2])
    intersection_y2 = min(box_a[3], box_b[3])
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def select_candidate(
    input_box: Tuple[float, float, float, float],
    candidate_boxes: List[Tuple[float, float, float, float]],
    min_box_iou: float,
) -> Tuple[int, float]:
    """Select the SAM3 candidate with maximum IoU to the input box."""
    if not 0.0 <= min_box_iou <= 1.0:
        raise ValueError("min_box_iou must be in [0, 1]")
    if not candidate_boxes:
        raise BoxSelectionError("SAM3 returned no candidate boxes")

    overlaps = [box_iou(input_box, item) for item in candidate_boxes]
    selected_index = max(range(len(overlaps)), key=overlaps.__getitem__)
    selected_iou = overlaps[selected_index]
    if selected_iou < min_box_iou:
        raise BoxSelectionError(
            f"best candidate IoU {selected_iou:.6f} is below minimum "
            f"{min_box_iou:.6f}"
        )
    return selected_index, selected_iou


def _require_int(
    document: Dict[str, Any],
    key: str,
) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _coerce_bbox(value: Any) -> Tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("each bbox must be a four-element xyxy list")
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox values must be numeric") from exc
    return box  # type: ignore[return-value]


def load_box_document(
    path: Path,
    actual_width: int,
    actual_height: int,
) -> Tuple[Dict[str, Any], List[BoxPrompt]]:
    """Load a box document and return its positive, validated prompts."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid box JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError("box JSON root must be an object")
    if document.get("schema_version") != "1.0":
        raise ValueError("schema_version must be '1.0'")
    if document.get("box_format") != "xyxy":
        raise ValueError("box_format must be 'xyxy'")

    image_width = _require_int(document, "image_width")
    image_height = _require_int(document, "image_height")
    if image_width != actual_width:
        raise ValueError(
            "box JSON image_width does not match input image: "
            f"{image_width} != {actual_width}"
        )
    if image_height != actual_height:
        raise ValueError(
            "box JSON image_height does not match input image: "
            f"{image_height} != {actual_height}"
        )

    raw_boxes = document.get("boxes")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError("boxes must be a non-empty list")

    prompts: List[BoxPrompt] = []
    seen_ids = set()
    for index, item in enumerate(raw_boxes):
        if not isinstance(item, dict):
            raise ValueError(f"boxes[{index}] must be an object")
        if item.get("positive", True) is not True:
            continue

        box_id = item.get("id", index)
        if (
            isinstance(box_id, bool)
            or not isinstance(box_id, int)
            or box_id < 0
        ):
            raise ValueError(f"boxes[{index}].id must be a non-negative integer")
        if box_id in seen_ids:
            raise ValueError(f"duplicate positive box id: {box_id}")
        seen_ids.add(box_id)

        label = item.get("label", "")
        if not isinstance(label, str):
            raise ValueError(f"boxes[{index}].label must be a string")

        bbox = _coerce_bbox(item.get("bbox"))
        validate_box(bbox, actual_width, actual_height)
        prompts.append(
            BoxPrompt(
                box_id=box_id,
                label=label,
                bbox_xyxy=bbox,
            )
        )

    if not prompts:
        raise ValueError("box JSON contains no positive boxes")
    return document, prompts


def _tensor_to_candidate_boxes(value: Any) -> List[Tuple[float, float, float, float]]:
    rows = value.detach().float().cpu().reshape(-1, 4).tolist()
    return [
        tuple(float(component) for component in row)  # type: ignore[misc]
        for row in rows
    ]


def _tensor_to_scores(value: Any) -> List[float]:
    return [
        float(component)
        for component in value.detach().float().cpu().reshape(-1).tolist()
    ]


def _extract_binary_mask(value: Any, index: int, width: int, height: int) -> Any:
    import numpy as np

    mask_array = (
        value[index].detach().squeeze().float().cpu().numpy()
    )
    if mask_array.shape != (height, width):
        raise RuntimeError(
            f"selected mask shape {mask_array.shape} does not match "
            f"image shape {(height, width)}"
        )
    binary_mask = mask_array > 0.5
    if not np.any(binary_mask):
        raise RuntimeError("selected SAM3 mask is empty")
    return binary_mask


def _save_mask(path: Path, mask: Any) -> None:
    import numpy as np
    from PIL import Image

    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask_image.save(path)


def _save_overlay(
    path: Path,
    image: Any,
    mask: Any,
    input_box: Tuple[float, float, float, float],
    predicted_box: Tuple[float, float, float, float],
    label: str,
    score: float,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    image_array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    green = np.asarray([0.0, 255.0, 0.0], dtype=np.float32)
    image_array[mask] = 0.55 * image_array[mask] + 0.45 * green
    overlay = Image.fromarray(
        np.clip(image_array, 0, 255).astype(np.uint8),
        mode="RGB",
    )
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(input_box, outline=(255, 215, 0), width=4)
    draw.rectangle(predicted_box, outline=(0, 255, 0), width=3)
    title = f"{label or 'object'} score={score:.3f}"
    text_box = draw.textbbox((0, 0), title)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_x = max(0.0, input_box[0])
    text_y = max(0.0, input_box[1] - text_height - 8)
    draw.rectangle(
        (
            text_x,
            text_y,
            text_x + text_width + 8,
            text_y + text_height + 6,
        ),
        fill=(12, 18, 28),
    )
    draw.text(
        (text_x + 4, text_y + 3),
        title,
        fill=(255, 255, 255),
    )
    overlay.save(path)


def _write_result(path: Path, result: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def _run_inference(
    image_path: Path,
    boxes_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    threshold: float,
    min_box_iou: float,
) -> int:
    import numpy as np
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    if not boxes_path.is_file():
        raise FileNotFoundError(f"box JSON not found: {boxes_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if checkpoint_path.stat().st_size == 0:
        raise ValueError(f"checkpoint is empty: {checkpoint_path}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not 0.0 <= min_box_iou <= 1.0:
        raise ValueError("min_box_iou must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM3 box inference")

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image_width, image_height = image.size
    document, prompts = load_box_document(
        boxes_path,
        image_width,
        image_height,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    model = build_sam3_image_model(
        device="cuda",
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
    )
    processor = Sam3Processor(
        model,
        device="cuda",
        confidence_threshold=threshold,
    )
    model_load_seconds = time.perf_counter() - load_started

    encode_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        state = processor.set_image(image)
    torch.cuda.synchronize()
    image_encode_seconds = time.perf_counter() - encode_started

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for prompt in prompts:
        prompt_started = time.perf_counter()
        try:
            processor.reset_all_prompts(state)
            normalized_box = xyxy_to_normalized_cxcywh(
                prompt.bbox_xyxy,
                image_width,
                image_height,
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                state = processor.add_geometric_prompt(
                    state=state,
                    box=normalized_box,
                    label=True,
                )
            torch.cuda.synchronize()

            candidate_boxes = _tensor_to_candidate_boxes(state["boxes"])
            scores = _tensor_to_scores(state["scores"])
            if len(candidate_boxes) != len(scores):
                raise RuntimeError(
                    "SAM3 candidate boxes and scores have different lengths"
                )
            selected_index, selected_iou = select_candidate(
                prompt.bbox_xyxy,
                candidate_boxes,
                min_box_iou,
            )
            if selected_index >= len(scores):
                raise RuntimeError("selected candidate score is missing")
            selected_mask = _extract_binary_mask(
                state["masks"],
                selected_index,
                image_width,
                image_height,
            )
            selected_box = candidate_boxes[selected_index]
            selected_score = scores[selected_index]

            stem = f"box_{prompt.box_id:03d}"
            mask_path = output_dir / f"{stem}_mask.png"
            overlay_path = output_dir / f"{stem}_overlay.png"
            _save_mask(mask_path, selected_mask)
            _save_overlay(
                overlay_path,
                image,
                selected_mask,
                prompt.bbox_xyxy,
                selected_box,
                prompt.label,
                selected_score,
            )

            successes.append(
                {
                    "id": prompt.box_id,
                    "label": prompt.label,
                    "input_box_xyxy": list(prompt.bbox_xyxy),
                    "normalized_box_cxcywh": normalized_box,
                    "num_candidates": len(candidate_boxes),
                    "selected_candidate_index": selected_index,
                    "selected_score": round(selected_score, 6),
                    "selected_box_xyxy": [
                        round(value, 3) for value in selected_box
                    ],
                    "selected_box_iou": round(selected_iou, 6),
                    "mask_area_pixels": int(np.count_nonzero(selected_mask)),
                    "prompt_seconds": round(
                        time.perf_counter() - prompt_started,
                        4,
                    ),
                    "mask_path": str(mask_path.resolve()),
                    "overlay_path": str(overlay_path.resolve()),
                }
            )
        except BoxSelectionError as exc:
            failures.append(
                {
                    "id": prompt.box_id,
                    "label": prompt.label,
                    "input_box_xyxy": list(prompt.bbox_xyxy),
                    "error": str(exc),
                }
            )
            LOGGER.error("Box %s failed: %s", prompt.box_id, exc)

    result: Dict[str, Any] = {
        "ok": not failures,
        "image_path": str(image_path.resolve()),
        "boxes_path": str(boxes_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "image_size": [image_width, image_height],
        "source_annotation": document.get("source_annotation"),
        "threshold": threshold,
        "min_box_iou": min_box_iou,
        "model_load_seconds": round(model_load_seconds, 4),
        "image_encode_seconds": round(image_encode_seconds, 4),
        "peak_cuda_memory_mib": round(
            torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            2,
        ),
        "num_boxes": len(prompts),
        "num_successes": len(successes),
        "num_failures": len(failures),
        "successes": successes,
        "failures": failures,
    }
    result_path = output_dir / "result.json"
    _write_result(result_path, result)
    LOGGER.info("Result JSON: %s", result_path.resolve())
    LOGGER.info(
        "Boxes: total=%d success=%d failure=%d",
        len(prompts),
        len(successes),
        len(failures),
    )
    return 0 if not failures else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 positive-box prompting and keep one complete target "
            "mask per input box."
        )
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="SAM3 candidate confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min-box-iou",
        type=float,
        default=0.10,
        help="minimum candidate/input box IoU (default: 0.10)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    try:
        return _run_inference(
            image_path=args.image.resolve(),
            boxes_path=args.boxes.resolve(),
            checkpoint_path=args.checkpoint.resolve(),
            output_dir=args.output_dir.resolve(),
            threshold=args.threshold,
            min_box_iou=args.min_box_iou,
        )
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        KeyError,
        ImportError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
