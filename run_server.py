#!/usr/bin/env python3
"""
Minimal HTTP wrapper for ``scripts/infer.py``.

Start:
  python run_server.py --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health
  POST /infer

POST /infer body example:
{
  "image_path": "/abs/path/to/image.png",
  "prompt": "white plate",
  "threshold": 0.41,
  "mask_threshold": 0.5,
  "save_vis": false
}

Or send a base64 image instead of ``image_path``:
{
  "image_base64": "<base64 or data URL>",
  "prompt": "white plate",
  "points": [[348, 236]],
  "point_labels": [1],
  "return_vis_base64": true
}
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib.util
import json
import os
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent
INFER_MODULE_PATH = REPO_ROOT / "scripts" / "infer.py"
INFER_LOCK = threading.Lock()
DEFAULT_CHECKPOINT_DIR = os.environ.get(
    "SAM3_CHECKPOINT_DIR", "/home/ubuntu/stephen/02-weight/sam3"
)
DEFAULT_CHECKPOINT = os.path.join(DEFAULT_CHECKPOINT_DIR, "sam3.pt")
_INFER_MODULE = None


def _load_infer_module():
    if not INFER_MODULE_PATH.is_file():
        raise FileNotFoundError(f"infer script not found: {INFER_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "sam3_infer_service_module", INFER_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {INFER_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_infer_module():
    global _INFER_MODULE
    if _INFER_MODULE is None:
        _INFER_MODULE = _load_infer_module()
    return _INFER_MODULE


def _get_infer_script_version() -> str:
    if _INFER_MODULE is None:
        return "unloaded"
    return str(getattr(_INFER_MODULE, "INFER_SCRIPT_VERSION", "unknown"))


def _coerce_bool(value: Any, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _coerce_float(value: Any, *, field_name: str, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def _coerce_int(value: Any, *, field_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _strip_data_url_prefix(encoded_image: str) -> str:
    if encoded_image.startswith("data:"):
        parts = encoded_image.split(",", 1)
        if len(parts) != 2:
            raise ValueError("invalid data URL in image_base64")
        return parts[1]
    return encoded_image


def _load_request_image(payload: Dict[str, Any], stack: contextlib.ExitStack) -> Path:
    image_path = payload.get("image_path")
    image_base64 = payload.get("image_base64")

    if bool(image_path) == bool(image_base64):
        raise ValueError("provide exactly one of image_path or image_base64")

    if image_path:
        resolved = Path(str(image_path)).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"image not found: {resolved}")
        return resolved

    try:
        raw_bytes = base64.b64decode(
            _strip_data_url_prefix(str(image_base64)), validate=True
        )
    except Exception as exc:
        raise ValueError("image_base64 is not valid base64 data") from exc

    try:
        from PIL import Image

        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("image_base64 is not a valid image") from exc

    temp_dir = Path(stack.enter_context(TemporaryDirectory(prefix="sam3_api_input_")))
    temp_image_path = temp_dir / "input.png"
    image.save(temp_image_path)
    return temp_image_path


def _coerce_points(value: Any) -> List[List[float]]:
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ValueError("points must be a non-empty list of [x, y] pairs")
    normalized: List[List[float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each point must be [x, y]")
        normalized.append([float(item[0]), float(item[1])])
    return normalized


def _coerce_point_labels(value: Any, num_points: int) -> List[int]:
    if value is None:
        return [1] * num_points
    if not isinstance(value, list) or len(value) != num_points:
        raise ValueError("point_labels must be a list with the same length as points")
    labels: List[int] = []
    for item in value:
        label = int(item)
        if label not in (0, 1):
            raise ValueError("point_labels must contain only 0 (negative) or 1 (positive)")
        labels.append(label)
    return labels


def _load_request_output_dir(
    payload: Dict[str, Any], stack: contextlib.ExitStack
) -> tuple[Path, bool]:
    output_dir = payload.get("output_dir")
    if output_dir:
        resolved = Path(str(output_dir)).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved, True

    temp_dir = Path(stack.enter_context(TemporaryDirectory(prefix="sam3_api_output_")))
    return temp_dir, False


def _read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _run_inference(payload: Dict[str, Any], server_defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    infer_module = _get_infer_module()
    run_sam3_segmentation = infer_module.run_sam3_segmentation
    infer_script_version = str(
        getattr(infer_module, "INFER_SCRIPT_VERSION", "unknown")
    )

    prompt = str(payload.get("prompt") or server_defaults["prompt"])
    checkpoint = str(payload.get("checkpoint") or server_defaults["checkpoint"])
    threshold = _coerce_float(
        payload.get("threshold"),
        field_name="threshold",
        default=server_defaults["threshold"],
    )
    mask_threshold = _coerce_float(
        payload.get("mask_threshold"),
        field_name="mask_threshold",
        default=server_defaults["mask_threshold"],
    )
    iom_threshold = _coerce_float(
        payload.get("iom_threshold"),
        field_name="iom_threshold",
        default=server_defaults["iom_threshold"],
    )
    fill_hole_area = _coerce_int(
        payload.get("fill_hole_area"),
        field_name="fill_hole_area",
        default=server_defaults["fill_hole_area"],
    )
    sprinkle_area = _coerce_int(
        payload.get("sprinkle_area"),
        field_name="sprinkle_area",
        default=server_defaults["sprinkle_area"],
    )
    postprocess = _coerce_bool(
        payload.get("postprocess"),
        field_name="postprocess",
        default=server_defaults["postprocess"],
    )
    return_vis_base64 = _coerce_bool(
        payload.get("return_vis_base64"),
        field_name="return_vis_base64",
        default=False,
    )
    save_vis = _coerce_bool(
        payload.get("save_vis"),
        field_name="save_vis",
        default=server_defaults["save_vis"],
    ) or return_vis_base64
    points = _coerce_points(payload.get("points"))
    point_labels = _coerce_point_labels(payload.get("point_labels"), len(points)) if points else None

    with contextlib.ExitStack() as stack:
        image_path = _load_request_image(payload, stack)
        output_dir, output_dir_persisted = _load_request_output_dir(payload, stack)

        t0 = time.perf_counter()
        with INFER_LOCK:
            json_path = run_sam3_segmentation(
                checkpoint_path=Path(checkpoint),
                rgb_path=image_path,
                output_dir=output_dir,
                prompt=prompt,
                threshold=threshold,
                mask_threshold=mask_threshold,
                save_vis=save_vis,
                iom_threshold=iom_threshold,
                fill_hole_area=fill_hole_area,
                sprinkle_area=sprinkle_area,
                postprocess=postprocess,
                points=points or None,
                point_labels=point_labels,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections = _read_json_file(Path(json_path))
        vis_path = output_dir / "sam6d_results" / "vis_ism.png"

        response: Dict[str, Any] = {
            "ok": True,
            "prompt": prompt,
            "points": points or None,
            "point_labels": point_labels,
            "num_detections": len(detections),
            "detections": detections,
            "elapsed_ms": round(elapsed_ms, 3),
            "checkpoint": str(Path(checkpoint).expanduser()),
            "script_version": infer_script_version,
            "output_json_path": str(json_path) if output_dir_persisted else None,
            "visualization_path": str(vis_path)
            if output_dir_persisted and vis_path.is_file()
            else None,
        }
        if return_vis_base64:
            if not vis_path.is_file():
                raise RuntimeError("visualization was requested but was not generated")
            response["visualization_base64"] = _encode_file_base64(vis_path)

        return response


class Sam3RequestHandler(BaseHTTPRequestHandler):
    server_version = "SAM3HTTP/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "sam3-infer",
                    "script_version": _get_infer_script_version(),
                    "infer_module_loaded": _INFER_MODULE is not None,
                },
            )
            return

        if path == "/":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "sam3-infer",
                    "endpoints": {
                        "health": "GET /health",
                        "infer": "POST /infer",
                    },
                },
            )
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown route: {path}")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path != "/infer":
            self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown route: {path}")
            return

        try:
            payload = self._read_json_body()
            response = _run_inference(payload, self.server.server_defaults)
        except FileNotFoundError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except RuntimeError as exc:
            self._send_error_json(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            return
        except Exception as exc:
            traceback.print_exc()
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"internal server error: {exc}"
            )
            return

        self._send_json(HTTPStatus.OK, response)

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("missing Content-Length header")

        try:
            body_size = int(content_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length header") from exc

        max_request_bytes = int(self.server.server_defaults["max_request_mb"] * 1024 * 1024)
        if body_size <= 0:
            raise ValueError("request body is empty")
        if body_size > max_request_bytes:
            raise ValueError(
                f"request body is too large: {body_size} bytes "
                f"(limit={max_request_bytes} bytes)"
            )

        raw_body = self.rfile.read(body_size)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def log_message(self, format: str, *args: Any) -> None:
        print(
            "[sam3_http] "
            f"{self.address_string()} - {self.log_date_time_string()} - {format % args}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 inference HTTP server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=18002, help="Bind port")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Default SAM3 checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Plastic Reel Conncted With Tape",
        help='Default text prompt when request omits "prompt"',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.221,
        help="Default detection score threshold",
    )
    parser.add_argument(
        "--mask-threshold",
        dest="mask_threshold",
        type=float,
        default=0.50,
        help="Default mask binarization threshold",
    )
    parser.add_argument(
        "--iom-threshold",
        dest="iom_threshold",
        type=float,
        default=0.30,
        help="Default IoM overlap threshold",
    )
    parser.add_argument(
        "--fill-hole-area",
        dest="fill_hole_area",
        type=int,
        default=16,
        help="Default max hole area to fill",
    )
    parser.add_argument(
        "--sprinkle-area",
        dest="sprinkle_area",
        type=int,
        default=16,
        help="Default max sprinkle area to remove",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Generate visualization by default for every request",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable postprocess by default for every request",
    )
    parser.add_argument(
        "--max-request-mb",
        type=int,
        default=64,
        help="Maximum JSON request size in MB",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Sam3RequestHandler)
    server.server_defaults = {
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        "threshold": args.threshold,
        "mask_threshold": args.mask_threshold,
        "iom_threshold": args.iom_threshold,
        "fill_hole_area": args.fill_hole_area,
        "sprinkle_area": args.sprinkle_area,
        "save_vis": args.save_vis,
        "postprocess": not args.no_postprocess,
        "max_request_mb": args.max_request_mb,
    }

    print(
        "[sam3_http] serving on "
        f"http://{args.host}:{args.port} "
        f"(infer module path: {INFER_MODULE_PATH})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[sam3_http] shutting down", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
