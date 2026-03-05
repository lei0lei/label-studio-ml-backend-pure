import os
import logging
from typing import Dict, Optional

from .base import BackendAdapter


logger = logging.getLogger(__name__)


class _ArrayProxy:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _SimpleBoxes:
    def __init__(self, cls_values, conf_values):
        self.cls = _ArrayProxy(cls_values)
        self.conf = _ArrayProxy(conf_values)


class _SimpleMasks:
    def __init__(self, polygons):
        self.xyn = polygons


class _SimpleInferenceResult:
    def __init__(self, image_shape, polygons, scores, binary_masks=None):
        self.orig_shape = image_shape
        self.masks = _SimpleMasks(polygons)
        self.boxes = _SimpleBoxes(
            cls_values=[0.0 for _ in polygons],
            conf_values=[float(score) for score in scores],
        )
        self.names = {0: 'object'}
        self.binary_masks = binary_masks or []


class Sam2ImageWrapper:
    def __init__(self, checkpoint_path: str):
        os.environ.setdefault('PYTORCH_JIT', '0')
        try:
            import cv2
            import numpy as np
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(f"SAM2 dependencies are not available: {exc}") from exc

        self.cv2 = cv2
        self.np = np
        self.torch = torch

        model_config = os.getenv('MODEL_CONFIG') or self._infer_model_config(checkpoint_path)
        device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

        sam2_model = build_sam2(model_config, checkpoint_path, device=device)
        try:
            self.predictor = SAM2ImagePredictor(sam2_model)
        except OSError as exc:
            if 'Failed to get source' not in str(exc):
                raise
            logger.warning('SAM2 predictor init failed with torch.jit in portable env, fallback to non-jit mode: %s', exc)
            original_script = torch.jit.script
            try:
                torch.jit.script = lambda obj, *args, **kwargs: obj
                self.predictor = SAM2ImagePredictor(sam2_model)
            finally:
                torch.jit.script = original_script

    def _infer_model_config(self, checkpoint_path: str):
        checkpoint_name = os.path.basename(checkpoint_path).lower()

        mapping = [
            ('sam2.1_hiera_tiny', 'configs/sam2.1/sam2.1_hiera_t.yaml'),
            ('sam2.1_hiera_small', 'configs/sam2.1/sam2.1_hiera_s.yaml'),
            ('sam2.1_hiera_base_plus', 'configs/sam2.1/sam2.1_hiera_b+.yaml'),
            ('sam2.1_hiera_large', 'configs/sam2.1/sam2.1_hiera_l.yaml'),
            ('sam2_hiera_tiny', 'configs/sam2/sam2_hiera_t.yaml'),
            ('sam2_hiera_small', 'configs/sam2/sam2_hiera_s.yaml'),
            ('sam2_hiera_base_plus', 'configs/sam2/sam2_hiera_b+.yaml'),
            ('sam2_hiera_large', 'configs/sam2/sam2_hiera_l.yaml'),
        ]
        for key, cfg in mapping:
            if key in checkpoint_name:
                return cfg

        return 'configs/sam2.1/sam2.1_hiera_t.yaml'

    def _mask_to_polygon(self, mask, image_width: int, image_height: int):
        contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=self.cv2.contourArea)
        if largest is None or len(largest) < 3:
            return None

        contour = largest.squeeze(axis=1)
        if contour.ndim != 2 or contour.shape[0] < 3:
            return None

        polygon = self.np.asarray(contour, dtype=self.np.float32)
        polygon[:, 0] = polygon[:, 0] / float(image_width)
        polygon[:, 1] = polygon[:, 1] / float(image_height)
        return polygon

    def __call__(self, local_path: str, points=None, labels=None, bboxes=None):
        image_bgr = self.cv2.imread(local_path, self.cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f'Failed to load image for SAM2: {local_path}')

        original_height, original_width = image_bgr.shape[:2]
        max_side = int(os.getenv('SAM2_MAX_IMAGE_SIDE', '2048'))
        scale_x = 1.0
        scale_y = 1.0

        if max_side > 0:
            longest_side = max(original_width, original_height)
            if longest_side > max_side:
                resize_scale = float(max_side) / float(longest_side)
                resized_width = max(1, int(round(original_width * resize_scale)))
                resized_height = max(1, int(round(original_height * resize_scale)))
                image_bgr = self.cv2.resize(
                    image_bgr,
                    (resized_width, resized_height),
                    interpolation=self.cv2.INTER_AREA,
                )
                scale_x = float(resized_width) / float(original_width)
                scale_y = float(resized_height) / float(original_height)
                logger.info(
                    'SAM2 resized image to reduce memory: %sx%s -> %sx%s (max_side=%s)',
                    original_width,
                    original_height,
                    resized_width,
                    resized_height,
                    max_side,
                )

        image = self.cv2.cvtColor(image_bgr, self.cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        self.predictor.set_image(image)

        point_coords = self.np.asarray(points, dtype=self.np.float32) if points else None
        if point_coords is not None and (scale_x != 1.0 or scale_y != 1.0):
            point_coords[:, 0] = point_coords[:, 0] * scale_x
            point_coords[:, 1] = point_coords[:, 1] * scale_y
        point_labels = self.np.asarray(labels, dtype=self.np.int32) if labels else None
        box_input = None
        if bboxes:
            box_array = self.np.asarray(bboxes, dtype=self.np.float32)
            if scale_x != 1.0 or scale_y != 1.0:
                box_array[:, 0] = box_array[:, 0] * scale_x
                box_array[:, 2] = box_array[:, 2] * scale_x
                box_array[:, 1] = box_array[:, 1] * scale_y
                box_array[:, 3] = box_array[:, 3] * scale_y
            box_input = box_array[0] if len(box_array) == 1 else box_array

        if point_coords is None and box_input is None:
            logger.warning('SAM2 requires point or box prompts, but no prompts were provided')
            return []

        with self.torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_input,
                multimask_output=True,
            )

        masks_np = self.np.asarray(masks)
        scores_np = self.np.asarray(scores).reshape(-1)

        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        elif masks_np.ndim == 4:
            masks_np = masks_np.reshape(-1, masks_np.shape[-2], masks_np.shape[-1])

        polygons = []
        polygon_scores = []
        binary_masks = []
        for index in range(len(masks_np)):
            binary_mask = (masks_np[index] > 0).astype('uint8')

            if image_width != original_width or image_height != original_height:
                binary_mask = self.cv2.resize(
                    binary_mask,
                    (original_width, original_height),
                    interpolation=self.cv2.INTER_NEAREST,
                )

            polygon = self._mask_to_polygon(
                binary_mask,
                image_width=original_width,
                image_height=original_height,
            )
            if polygon is None:
                continue
            polygons.append(polygon)
            polygon_scores.append(float(scores_np[index]) if index < len(scores_np) else 1.0)
            binary_masks.append(binary_mask)

        if not polygons:
            return []

        return [_SimpleInferenceResult((original_height, original_width), polygons, polygon_scores, binary_masks=binary_masks)]


class Sam2BackendAdapter(BackendAdapter):
    backend_name = 'sam2'
    supports_training = False

    @property
    def model_cls(self):
        return Sam2ImageWrapper

    def _extract_sam_prompts(self, context: Optional[Dict], task: Dict):
        if not isinstance(context, dict):
            return None, None, None

        ctx_results = context.get('result')
        if not isinstance(ctx_results, list) or not ctx_results:
            return None, None, None

        task_id = task.get('id')
        task_ctx_results = []
        for item in ctx_results:
            if not isinstance(item, dict):
                continue
            ctx_task_id = item.get('task_id') or item.get('task')
            if task_id is None or ctx_task_id is None or str(ctx_task_id) == str(task_id):
                task_ctx_results.append(item)

        if not task_ctx_results:
            task_ctx_results = [item for item in ctx_results if isinstance(item, dict)]

        point_coords = []
        point_labels = []
        bboxes = []

        for item in task_ctx_results:
            value = item.get('value') or {}
            if not isinstance(value, dict):
                continue

            width = item.get('original_width') or value.get('original_width')
            height = item.get('original_height') or value.get('original_height')
            x_pct = value.get('x')
            y_pct = value.get('y')
            if width is None or height is None or x_pct is None or y_pct is None:
                continue

            x = float(x_pct) * float(width) / 100.0
            y = float(y_pct) * float(height) / 100.0

            region_type = item.get('type')
            if region_type == 'keypointlabels':
                point_coords.append([x, y])
                point_labels.append(int(item.get('is_positive', 1)))
                continue

            if region_type == 'rectanglelabels':
                box_w_pct = value.get('width')
                box_h_pct = value.get('height')
                if box_w_pct is None or box_h_pct is None:
                    continue
                box_w = float(box_w_pct) * float(width) / 100.0
                box_h = float(box_h_pct) * float(height) / 100.0
                bboxes.append([x, y, x + box_w, y + box_h])

        points_arg = point_coords if point_coords else None
        labels_arg = point_labels if point_labels else None
        bboxes_arg = bboxes if bboxes else None
        return points_arg, labels_arg, bboxes_arg

    def run(self, selected_model, local_path: str, model_task: str, imgsz: int, context: Optional[Dict], task: Dict):
        points_arg, labels_arg, bboxes_arg = self._extract_sam_prompts(context, task)
        infer_kwargs = {}
        if points_arg is not None:
            infer_kwargs['points'] = points_arg
        if labels_arg is not None:
            infer_kwargs['labels'] = labels_arg
        if bboxes_arg is not None:
            infer_kwargs['bboxes'] = bboxes_arg
        return selected_model(local_path, **infer_kwargs)
