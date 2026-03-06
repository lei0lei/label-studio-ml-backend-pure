"""SAM2 backend adapter and lightweight output wrappers.

Includes prompt extraction, image-level predictor wrapper, and conversion of
SAM2 outputs into a shape compatible with shared post-processing utilities.
"""

import os
import logging
import base64
from typing import Any, Dict, Optional, Tuple

from .base import BackendAdapter


logger = logging.getLogger(__name__)


class _ArrayProxy:
    """Minimal tensor-like proxy exposing ``cpu().numpy()`` chain."""

    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _SimpleBoxes:
    """Simplified boxes object carrying class and confidence arrays."""

    def __init__(self, cls_values, conf_values):
        self.cls = _ArrayProxy(cls_values)
        self.conf = _ArrayProxy(conf_values)


class _SimpleMasks:
    """Simplified masks object carrying normalized polygon points."""

    def __init__(self, polygons):
        self.xyn = polygons


class _SimpleInferenceResult:
    """Unified inference result shape expected by presentation builders."""

    def __init__(self, image_shape, polygons, scores, binary_masks=None, sam2_features=None):
        self.orig_shape = image_shape
        self.masks = _SimpleMasks(polygons)
        self.boxes = _SimpleBoxes(
            cls_values=[0.0 for _ in polygons],
            conf_values=[float(score) for score in scores],
        )
        self.names = {0: 'object'}
        self.binary_masks = binary_masks or []
        self.sam2_features = sam2_features


class Sam2ImageWrapper:
    """Image predictor wrapper around SAM2 interactive segmentation model."""

    def __init__(self, checkpoint_path: str):
        """Initialize SAM2 predictor and handle portable non-JIT fallback."""
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
        """Infer SAM2 config path from checkpoint filename pattern."""
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
        """Extract largest external contour and normalize to [0, 1] polygon."""
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

    def _encode_feature_tensor(self, tensor):
        if tensor is None:
            return None
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach()
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        if hasattr(tensor, 'numpy'):
            tensor = tensor.numpy()

        array = self.np.ascontiguousarray(tensor)
        max_bytes = int(os.getenv('SAM2_FEATURE_MAX_BYTES', '0') or '0')
        if max_bytes > 0 and int(array.nbytes) > max_bytes:
            logger.warning('Skip SAM2 feature export because tensor is too large: bytes=%s limit=%s', array.nbytes, max_bytes)
            return None

        return {
            'shape': list(array.shape),
            'dtype': str(array.dtype),
            'data_b64': base64.b64encode(array.tobytes()).decode('ascii'),
        }

    def _decode_feature_tensor(self, payload):
        if not isinstance(payload, dict):
            return None

        shape = payload.get('shape')
        dtype = payload.get('dtype')
        data_b64 = payload.get('data_b64')
        if not isinstance(shape, (list, tuple)) or not dtype or not data_b64:
            return None

        try:
            shape_tuple = tuple(int(v) for v in shape)
            np_dtype = self.np.dtype(dtype)
            raw = base64.b64decode(data_b64)
        except Exception as exc:
            logger.warning('Failed to decode SAM2 feature tensor metadata: %s', exc)
            return None

        array = self.np.frombuffer(raw, dtype=np_dtype)
        expected_size = int(self.np.prod(shape_tuple)) if shape_tuple else 1
        if int(array.size) != expected_size:
            logger.warning('Invalid SAM2 feature tensor size: got=%s expected=%s', array.size, expected_size)
            return None

        try:
            array = array.reshape(shape_tuple)
        except Exception as exc:
            logger.warning('Failed to reshape SAM2 feature tensor to %s: %s', shape_tuple, exc)
            return None

        device = getattr(self.predictor, 'device', self.torch.device('cpu'))
        return self.torch.as_tensor(self.np.ascontiguousarray(array), device=device)

    def _normalize_image_hw(self, image_hw) -> Optional[Tuple[int, int]]:
        if isinstance(image_hw, dict):
            height = image_hw.get('height') or image_hw.get('original_height')
            width = image_hw.get('width') or image_hw.get('original_width')
            if height and width:
                return int(height), int(width)
            return None

        if isinstance(image_hw, (list, tuple)) and len(image_hw) >= 2:
            return int(image_hw[0]), int(image_hw[1])

        return None

    def _restore_features_from_payload(self, sam2_features, image_hw=None):
        if not isinstance(sam2_features, dict):
            return False

        image_embed = self._decode_feature_tensor(sam2_features.get('image_embed'))
        high_res_payload = sam2_features.get('high_res_feats')
        high_res_feats = []

        if isinstance(high_res_payload, (list, tuple)):
            for feat_payload in high_res_payload:
                decoded_feat = self._decode_feature_tensor(feat_payload)
                if decoded_feat is not None:
                    high_res_feats.append(decoded_feat)

        if image_embed is None or not high_res_feats:
            logger.warning('SAM2 feature payload is incomplete, fallback to image encoding path')
            return False

        self.predictor.reset_predictor()
        self.predictor._features = {
            'image_embed': image_embed,
            'high_res_feats': high_res_feats,
        }
        normalized_hw = self._normalize_image_hw(image_hw)
        if normalized_hw is not None:
            self.predictor._orig_hw = [normalized_hw]
        self.predictor._is_batch = False
        self.predictor._is_image_set = True
        return True

    def _is_feature_export_enabled(self):
        return os.getenv('SAM2_RETURN_FEATURES', '0').strip().lower() in {'1', 'true', 'yes'}

    def _collect_sam2_features(self):
        enabled = self._is_feature_export_enabled()
        if not enabled:
            return None

        predictor_features = getattr(self.predictor, '_features', None)
        if not isinstance(predictor_features, dict):
            logger.warning('SAM2_RETURN_FEATURES is enabled but predictor._features is unavailable')
            return None

        payload = {}

        image_embed = self._encode_feature_tensor(predictor_features.get('image_embed'))
        if image_embed is not None:
            payload['image_embed'] = image_embed

        high_res_feats = predictor_features.get('high_res_feats')
        if isinstance(high_res_feats, (list, tuple)):
            encoded_high_res = []
            for feat in high_res_feats:
                encoded_feat = self._encode_feature_tensor(feat)
                if encoded_feat is not None:
                    encoded_high_res.append(encoded_feat)
            if encoded_high_res:
                payload['high_res_feats'] = encoded_high_res

        if not payload:
            logger.warning('SAM2_RETURN_FEATURES is enabled but no feature tensors were exported')
            return None

        payload['format'] = 'numpy-bytes-base64'
        return payload

    def __call__(self, local_path: str, points=None, labels=None, bboxes=None, sam2_features=None, image_hw=None):
        """Run SAM2 with interactive prompts and return normalized result list."""
        original_height = None
        original_width = None
        image_height = None
        image_width = None
        scale_x = 1.0
        scale_y = 1.0
        used_feature_payload = False

        if self._restore_features_from_payload(sam2_features, image_hw=image_hw):
            used_feature_payload = True
            normalized_hw = self._normalize_image_hw(image_hw)
            if normalized_hw is None:
                predictor_hw = getattr(self.predictor, '_orig_hw', None)
                if isinstance(predictor_hw, list) and predictor_hw:
                    normalized_hw = self._normalize_image_hw(predictor_hw[0])
            if normalized_hw is None:
                raise RuntimeError('SAM2 feature payload provided but original image size is missing')
            original_height, original_width = normalized_hw
            image_height, image_width = original_height, original_width
            logger.info('Using client-provided SAM2 features to skip image embedding step')
        else:
            image_bgr = self.cv2.imread(local_path, self.cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f'Failed to load image for SAM2: {local_path}')

            original_height, original_width = image_bgr.shape[:2]
            max_side = int(os.getenv('SAM2_MAX_IMAGE_SIDE', '2048'))

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

        exported_features = None
        if self._is_feature_export_enabled():
            if used_feature_payload and isinstance(sam2_features, dict):
                exported_features = sam2_features
            else:
                exported_features = self._collect_sam2_features()

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
            auto_full_image_box = os.getenv('SAM2_AUTO_FULL_IMAGE_BOX', '0').strip().lower() in {'1', 'true', 'yes'}
            if not auto_full_image_box:
                logger.warning('SAM2 requires point or box prompts, but no prompts were provided')
                return []

            box_input = self.np.asarray([0.0, 0.0, float(image_width - 1), float(image_height - 1)], dtype=self.np.float32)
            logger.info(
                'No SAM2 prompts provided, fallback to full-image box prompt: [0,0,%s,%s]',
                image_width - 1,
                image_height - 1,
            )

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

        return [
            _SimpleInferenceResult(
                (original_height, original_width),
                polygons,
                polygon_scores,
                binary_masks=binary_masks,
                sam2_features=exported_features,
            )
        ]


class Sam2BackendAdapter(BackendAdapter):
    """Adapter that runs SAM2 prompt-based segmentation inference."""

    backend_name = 'sam2'
    supports_training = False

    @property
    def model_cls(self):
        """Return wrapper class used to initialize SAM2 image predictor."""
        return Sam2ImageWrapper

    def _extract_sam_prompts(self, context: Optional[Dict], task: Dict):
        """Extract point/box prompts for the current task from LS context."""
        sam2_features = None
        image_hw = None

        if not isinstance(context, dict):
            return None, None, None, sam2_features, image_hw

        context_features = context.get('sam2_features')
        if isinstance(context_features, dict):
            sam2_features = context_features

        context_hw = context.get('sam2_image_hw')
        if isinstance(context_hw, (list, tuple, dict)):
            image_hw = context_hw

        ctx_results = context.get('result')
        if not isinstance(ctx_results, list) or not ctx_results:
            return None, None, None, sam2_features, image_hw

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
            if sam2_features is None:
                item_meta = item.get('meta')
                if isinstance(item_meta, dict) and isinstance(item_meta.get('sam2_features'), dict):
                    sam2_features = item_meta.get('sam2_features')

            value = item.get('value') or {}
            if not isinstance(value, dict):
                continue

            width = item.get('original_width') or value.get('original_width')
            height = item.get('original_height') or value.get('original_height')
            x_pct = value.get('x')
            y_pct = value.get('y')
            if width is None or height is None or x_pct is None or y_pct is None:
                continue

            if image_hw is None:
                image_hw = (int(float(height)), int(float(width)))

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
        return points_arg, labels_arg, bboxes_arg, sam2_features, image_hw

    def run(self, selected_model, local_path: str, model_task: str, imgsz: int, context: Optional[Dict], task: Dict):
        """Execute SAM2 inference with prompts derived from interaction context."""
        points_arg, labels_arg, bboxes_arg, sam2_features, image_hw = self._extract_sam_prompts(context, task)
        infer_kwargs = {}
        if points_arg is not None:
            infer_kwargs['points'] = points_arg
        if labels_arg is not None:
            infer_kwargs['labels'] = labels_arg
        if bboxes_arg is not None:
            infer_kwargs['bboxes'] = bboxes_arg
        if sam2_features is not None:
            infer_kwargs['sam2_features'] = sam2_features
        if image_hw is not None:
            infer_kwargs['image_hw'] = image_hw
        return selected_model(local_path, **infer_kwargs)
