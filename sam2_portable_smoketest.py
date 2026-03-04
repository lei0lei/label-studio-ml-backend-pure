import argparse
import os
import sys
import tempfile
import traceback

import numpy as np
from PIL import Image


def build_parser():
    parser = argparse.ArgumentParser(description="SAM2 portable smoke test")
    parser.add_argument(
        "--checkpoint",
        default="my_ml_backend/sam2.1_hiera_tiny.pt",
        help="Path to SAM2 checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM2 model config path (relative to sam2 package root conventions)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional image path for testing. If omitted, a temporary image will be generated.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cuda or cpu",
    )
    return parser


def resolve_paths(args):
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint = args.checkpoint
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(workspace_root, checkpoint)
    image_path = args.image
    if image_path and not os.path.isabs(image_path):
        image_path = os.path.join(workspace_root, image_path)
    return workspace_root, checkpoint, image_path


def create_temp_image():
    img = (np.random.rand(256, 256, 3) * 255).astype("uint8")
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.fromarray(img).save(path)
    return path


def main():
    parser = build_parser()
    args = parser.parse_args()
    workspace_root, checkpoint_path, image_path = resolve_paths(args)

    if not os.path.exists(checkpoint_path):
        print(f"[FAIL] Checkpoint not found: {checkpoint_path}")
        return 2

    if args.config:
        os.environ["MODEL_CONFIG"] = args.config
    if args.device:
        os.environ["DEVICE"] = args.device

    my_ml_backend_dir = os.path.join(workspace_root, "my_ml_backend")
    if my_ml_backend_dir not in sys.path:
        sys.path.insert(0, my_ml_backend_dir)

    temp_image_used = False
    if not image_path:
        image_path = create_temp_image()
        temp_image_used = True

    try:
        from model import Sam2ImageWrapper

        wrapper = Sam2ImageWrapper(checkpoint_path)

        with Image.open(image_path) as img:
            width, height = img.size

        bbox = [
            int(width * 0.2),
            int(height * 0.2),
            int(width * 0.8),
            int(height * 0.8),
        ]

        results = wrapper(image_path, bboxes=[bbox])

        print("[OK] SAM2 loaded and inference executed")
        print(f"checkpoint={checkpoint_path}")
        print(f"model_config={os.getenv('MODEL_CONFIG')}")
        print(f"image={image_path}")
        print(f"results_count={len(results)}")

        if results:
            masks = getattr(results[0], "masks", None)
            polygons = getattr(masks, "xyn", []) if masks is not None else []
            print(f"polygon_count={len(polygons)}")
        else:
            print("[WARN] Inference returned empty results (model loaded successfully)")

        return 0
    except Exception:
        print("[FAIL] SAM2 load or inference failed")
        traceback.print_exc()
        return 1
    finally:
        if temp_image_used and os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    raise SystemExit(main())
