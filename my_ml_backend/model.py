
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import os
import logging
import re
import math
from urllib.parse import urljoin, urlparse, unquote
from backend_core.domain.routing import (
    build_route_spec,
    family_exists,
    is_task_supported,
    normalize_family,
    normalize_task,
)
from backend_core.backends.registry import BackendRegistry
from backend_core.model_loader import ModelLoader
try:
    from label_studio_sdk.converter.utils import convert_yolo_obb_to_annotation
except ImportError:
    convert_yolo_obb_to_annotation = None
try:
    from label_studio_sdk.converter import brush as ls_brush
except ImportError:
    ls_brush = None


logger = logging.getLogger(__name__)
class Model(LabelStudioMLBase):
    """Unified Label Studio ML Backend with model-family + task routing"""

    def setup(self):
        default_model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        self.default_model_path = os.getenv('YOLOV8_MODEL_PATH', default_model_path)
        sam2_default = os.getenv('SAM2_MODEL_PATH', os.path.join(os.path.dirname(__file__), 'sam2.1_hiera_tiny.pt'))
        self.default_sam2_model_path = sam2_default if os.path.exists(sam2_default) else self.default_model_path
        self.model_dir = os.getenv('YOLOV8_MODEL_DIR', os.path.dirname(self.default_model_path))
        logger.info(
            "Model router initialized: model_dir=%s default_model=%s default_sam2=%s",
            self.model_dir,
            self.default_model_path,
            self.default_sam2_model_path,
        )
        self.backend_registry = BackendRegistry()
        self.model_loader = ModelLoader(
            model_dir=self.model_dir,
            default_model_path=self.default_model_path,
            default_sam2_model_path=self.default_sam2_model_path,
        )
        self.set("model_version", "model-router-1.0")
        self.from_name = "label"
        self.to_name = "image"
        self.labels_in_config = []
        self.result_type = "rectanglelabels"
        self.result_label_key = "rectanglelabels"

    def _sanitize_model_name(self, model_name: str):
        if not model_name:
            return 'best'
        safe_name = re.sub(r'[^0-9a-zA-Z_\-.]', '', str(model_name))
        return safe_name or 'best'

    def _sanitize_model_segment(self, segment: str):
        if not segment:
            return None
        safe_segment = re.sub(r'[^0-9a-zA-Z_\-]', '', str(segment))
        return safe_segment or None

    def _normalize_task(self, model_task: Optional[str]):
        return normalize_task(model_task)

    def _normalize_family(self, model_family: Optional[str]):
        return normalize_family(model_family)

    def _family_exists(self, model_family: str):
        return family_exists(model_family)

    def _is_task_supported(self, model_family: str, model_task: str):
        return is_task_supported(model_family, model_task)

    def _resolve_backend(self, model_family: str):
        return build_route_spec('best', 'detect', model_family).backend

    def _select_tag_candidates(self, model_task: str):
        if model_task == 'segment':
            self.result_type = 'polygonlabels'
            self.result_label_key = 'polygonlabels'
            return [
                ('BrushLabels', 'Image'),
                ('BrushLabels', 'HyperText'),
                ('PolygonLabels', 'Image'),
                ('PolygonLabels', 'HyperText'),
            ]

        if model_task == 'obb':
            # OBB 只支持 RectangleLabels，避免 Labels 控件不显示
            self.result_type = 'rectanglelabels'
            self.result_label_key = 'rectanglelabels'
            return [
                ('RectangleLabels', 'Image'),
                ('RectangleLabels', 'HyperText'),
            ]

        self.result_type = 'rectanglelabels'
        self.result_label_key = 'rectanglelabels'
        return [
            ('RectangleLabels', 'Image'),
            ('Labels', 'Image'),
            ('RectangleLabels', 'HyperText'),
        ]

    def _resolve_model_path(self, model_name: str, model_task: Optional[str] = None, model_family: Optional[str] = None, backend: Optional[str] = None):
        safe_name = self._sanitize_model_name(model_name)
        safe_task = self._sanitize_model_segment(model_task)
        safe_family = self._sanitize_model_segment(model_family)

        # 模型文件名按路由中的完整 model_name 查找（例如 obb_640 -> obb_640.pt）

        if safe_name in ('default', 'best'):
            cache_key = f"{safe_task or 'default'}::{safe_family or 'default'}::{safe_name}"
            if backend == 'sam2':
                return self.default_sam2_model_path, cache_key
            return self.default_model_path, cache_key

        candidate_dirs = []
        if safe_task and safe_family:
            candidate_dirs.append(os.path.join(self.model_dir, safe_task, safe_family))
        candidate_dirs.append(self.model_dir)

        model_filename = f"{safe_name}.pt"
        for candidate_dir in candidate_dirs:
            model_path = os.path.join(candidate_dir, model_filename)
            if os.path.exists(model_path):
                cache_key = f"{safe_task or 'default'}::{safe_family or 'default'}::{safe_name}"
                return model_path, cache_key

        logger.warning(
            "Model file not found for model_task=%s model_family=%s model_name=%s, fallback to default model",
            safe_task,
            safe_family,
            safe_name,
        )
        fallback_key = f"{safe_task or 'default'}::{safe_family or 'default'}::best"
        return self.default_model_path, fallback_key

    def _get_or_load_model(self, model_name: str, model_task: Optional[str] = None, model_family: Optional[str] = None):
        route_spec = build_route_spec(model_name, model_task, model_family)
        return self.model_loader.get_or_load_model(route_spec, self.backend_registry)

    def _resolve_tag_mapping(self, model_task: str):
        candidates = self._select_tag_candidates(model_task)
        for control_type, object_type in candidates:
            try:
                from_name, to_name, _ = self.get_first_tag_occurence(control_type, object_type)
                self.from_name = from_name
                self.to_name = to_name
                self.labels_in_config = self.parsed_label_config.get(from_name, {}).get('labels', [])
                if control_type == 'BrushLabels':
                    self.result_type = 'brushlabels'
                    self.result_label_key = 'brushlabels'
                elif control_type == 'PolygonLabels':
                    self.result_type = 'polygonlabels'
                    self.result_label_key = 'polygonlabels'
                logger.info(
                    "Resolved label config mapping: control=%s object=%s from_name=%s to_name=%s labels=%d",
                    control_type,
                    object_type,
                    from_name,
                    to_name,
                    len(self.labels_in_config)
                )
                return
            except Exception:
                continue

        logger.warning(
            "No compatible control/object tag found in label config. Using fallback from_name=%s to_name=%s",
            self.from_name,
            self.to_name
        )

    def _pick_output_label(self, label_index: Optional[int], model_label: Optional[str]):
        if not self.labels_in_config:
            return model_label or 'object'
        if label_index is not None and label_index < len(self.labels_in_config):
            return self.labels_in_config[label_index]
        if model_label and model_label in self.labels_in_config:
            return model_label
        return self.labels_in_config[0]

    def _build_detection_results(self, inference_result):
        results = []
        boxes = getattr(inference_result, 'boxes', None)
        if boxes is None or boxes.data is None:
            return results

        width = float(inference_result.orig_shape[1])
        height = float(inference_result.orig_shape[0])

        for box in boxes.data.cpu().numpy():
            x_min, y_min, x_max, y_max, conf, cls = [float(v) for v in box[:6]]
            label_index = int(cls)
            score = float(conf)
            model_label = str(inference_result.names.get(label_index, label_index))
            output_label = self._pick_output_label(label_index, model_label)
            value = {
                "rectanglelabels": [output_label],
                "x": float(100 * x_min / width),
                "y": float(100 * y_min / height),
                "width": float(100 * (x_max - x_min) / width),
                "height": float(100 * (y_max - y_min) / height),
                "rotation": 0.0,
            }
            results.append({
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": value,
                "score": score,
            })
        return results

    def _build_obb_results(self, inference_result):
        results = []
        obb = getattr(inference_result, 'obb', None)
        if obb is None:
            return self._build_detection_results(inference_result)

        def _to_numpy(tensor_like):
            if tensor_like is None:
                return None
            if hasattr(tensor_like, 'cpu'):
                tensor_like = tensor_like.cpu()
            if hasattr(tensor_like, 'numpy'):
                return tensor_like.numpy()
            return tensor_like

        def _angle_distance_deg(a_deg, b_deg):
            delta = abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)
            return delta

        def _canonicalize_obb_points(points, angle_hint_deg=None):
            raw_points = [(float(p[0]), float(p[1])) for p in points[:4]]
            if len(raw_points) < 4:
                return None

            cx = sum(p[0] for p in raw_points) / 4.0
            cy = sum(p[1] for p in raw_points) / 4.0
            ordered = sorted(raw_points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

            tl_index = min(range(4), key=lambda idx: ordered[idx][0] + ordered[idx][1])
            rotated = ordered[tl_index:] + ordered[:tl_index]

            seq_forward = rotated
            seq_reverse = [rotated[0], rotated[3], rotated[2], rotated[1]]

            def _rotation_from_seq(seq):
                p0 = seq[0]
                p1 = seq[1]
                return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

            if angle_hint_deg is not None:
                rot_fwd = _rotation_from_seq(seq_forward)
                rot_rev = _rotation_from_seq(seq_reverse)
                if _angle_distance_deg(rot_rev, angle_hint_deg) < _angle_distance_deg(rot_fwd, angle_hint_deg):
                    return seq_reverse
                return seq_forward

            p0 = seq_forward[0]
            if seq_forward[1][0] < p0[0] and seq_reverse[1][0] >= p0[0]:
                return seq_reverse
            return seq_forward

        # 优先使用四点像素坐标，最稳妥
        xyxyxyxy = _to_numpy(getattr(obb, 'xyxyxyxy', None))
        xywhr = getattr(obb, 'xywhr', None)
        xywhn = getattr(obb, 'xywhn', None)
        if xyxyxyxy is None and xywhr is None and xywhn is None:
            return self._build_detection_results(inference_result)

        cls_data = getattr(obb, 'cls', None)
        conf_data = getattr(obb, 'conf', None)

        width = float(inference_result.orig_shape[1])
        height = float(inference_result.orig_shape[0])
        xywhr_np = _to_numpy(xywhr)
        xywhn_np = _to_numpy(xywhn)
        cls_np = _to_numpy(cls_data) if cls_data is not None else []
        conf_np = _to_numpy(conf_data) if conf_data is not None else []

        if xyxyxyxy is not None:
            row_count = len(xyxyxyxy)
        elif xywhr_np is not None:
            row_count = len(xywhr_np)
        else:
            row_count = len(xywhn_np)

        for index in range(row_count):
            label_index = int(float(cls_np[index])) if index < len(cls_np) else None
            score = float(conf_np[index]) if index < len(conf_np) else 1.0
            model_label = str(inference_result.names.get(label_index, label_index)) if label_index is not None else None
            output_label = self._pick_output_label(label_index, model_label)

            if xyxyxyxy is not None:
                points = xyxyxyxy[index]
                if points is None or len(points) < 4:
                    continue

                angle_hint_deg = None
                if xywhr_np is not None and index < len(xywhr_np):
                    angle_hint_deg = math.degrees(float(xywhr_np[index][4]))

                canonical_points = _canonicalize_obb_points(points, angle_hint_deg=angle_hint_deg)
                if canonical_points is None:
                    continue

                if convert_yolo_obb_to_annotation is not None:
                    flat_points = []
                    for point in canonical_points:
                        flat_points.extend([float(point[0]), float(point[1])])
                    converted_value = convert_yolo_obb_to_annotation(flat_points, width, height)
                    x_percent = float(converted_value["x"])
                    y_percent = float(converted_value["y"])
                    w_percent = float(converted_value["width"])
                    h_percent = float(converted_value["height"])
                    rotation_deg = float(converted_value["rotation"])
                else:
                    p0x, p0y = canonical_points[0]
                    p1x, p1y = canonical_points[1]
                    p2x, p2y = canonical_points[2]
                    p3x, p3y = canonical_points[3]

                    center_x = (p0x + p1x + p2x + p3x) / 4.0
                    center_y = (p0y + p1y + p2y + p3y) / 4.0

                    width_px = math.hypot(p1x - p0x, p1y - p0y)
                    height_px = math.hypot(p3x - p0x, p3y - p0y)
                    if width_px <= 0.0 or height_px <= 0.0:
                        continue

                    rotation_deg = math.degrees(math.atan2(p1y - p0y, p1x - p0x))

                    rotation_rad = math.radians(rotation_deg)
                    cos_theta = math.cos(rotation_rad)
                    sin_theta = math.sin(rotation_rad)

                    x0 = center_x - (width_px / 2.0) * cos_theta + (height_px / 2.0) * sin_theta
                    y0 = center_y - (width_px / 2.0) * sin_theta - (height_px / 2.0) * cos_theta

                    x_percent = x0 / width * 100.0
                    y_percent = y0 / height * 100.0
                    w_percent = width_px / width * 100.0
                    h_percent = height_px / height * 100.0
            else:
                if xywhr_np is not None:
                    cx, cy, width_px, height_px, angle_rad = [float(v) for v in xywhr_np[index][:5]]
                else:
                    cx_norm, cy_norm, width_norm, height_norm, angle_rad = [float(v) for v in xywhn_np[index][:5]]
                    cx = cx_norm * width
                    cy = cy_norm * height
                    width_px = width_norm * width
                    height_px = height_norm * height

                if width_px <= 0.0 or height_px <= 0.0:
                    continue

                cos_theta = math.cos(angle_rad)
                sin_theta = math.sin(angle_rad)
                # 由中心点反推旋转矩形左上角（以矩形局部坐标的 p0 为锚点）
                x0 = cx - (width_px / 2.0) * cos_theta + (height_px / 2.0) * sin_theta
                y0 = cy - (width_px / 2.0) * sin_theta - (height_px / 2.0) * cos_theta

                x_percent = x0 / width * 100.0
                y_percent = y0 / height * 100.0
                w_percent = width_px / width * 100.0
                h_percent = height_px / height * 100.0

                angle_ccw_deg = math.degrees(angle_rad)
                rotation_deg = angle_ccw_deg

            value = {
                "rectanglelabels": [output_label],
                "x": float(x_percent),
                "y": float(y_percent),
                "width": float(w_percent),
                "height": float(h_percent),
                "rotation": float(rotation_deg),
                "original_width": int(width),
                "original_height": int(height),
            }
            results.append({
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": value,
                "score": score,
            })

        logger.info("Built OBB results: %d boxes", len(results))
        return results

    def _build_segment_results(self, inference_result):
        results = []
        masks = getattr(inference_result, 'masks', None)
        if masks is None:
            return results

        polygons = getattr(masks, 'xyn', None)
        if polygons is None:
            return results

        if self.result_type == 'brushlabels':
            binary_masks = getattr(inference_result, 'binary_masks', None) or []
            if ls_brush is None:
                logger.error('label_studio_sdk brush converter is unavailable, cannot output brushlabels')
                return results

            boxes = getattr(inference_result, 'boxes', None)
            cls_np = boxes.cls.cpu().numpy() if boxes is not None and boxes.cls is not None else []
            conf_np = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else []
            model_names = getattr(inference_result, 'names', {})
            image_width = int(float(inference_result.orig_shape[1]))
            image_height = int(float(inference_result.orig_shape[0]))

            def _resolve_model_label_for_brush(label_index: Optional[int]):
                if label_index is None:
                    return None
                if isinstance(model_names, dict):
                    return str(model_names.get(label_index, label_index))
                if isinstance(model_names, (list, tuple)) and 0 <= label_index < len(model_names):
                    return str(model_names[label_index])
                return str(label_index)

            for index, mask in enumerate(binary_masks):
                if mask is None:
                    continue
                label_index = int(float(cls_np[index])) if index < len(cls_np) else None
                score = float(conf_np[index]) if index < len(conf_np) else 1.0
                model_label = _resolve_model_label_for_brush(label_index)
                output_label = self._pick_output_label(label_index, model_label)
                rle = ls_brush.mask2rle(mask.astype('uint8') * 255)
                value = {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [output_label],
                }
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'brushlabels',
                    'value': value,
                    'original_width': image_width,
                    'original_height': image_height,
                    'image_rotation': 0,
                    'score': score,
                })
            return results

        boxes = getattr(inference_result, 'boxes', None)
        cls_np = boxes.cls.cpu().numpy() if boxes is not None and boxes.cls is not None else []
        conf_np = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else []
        model_names = getattr(inference_result, 'names', {})

        def _resolve_model_label(label_index: Optional[int]):
            if label_index is None:
                return None
            if isinstance(model_names, dict):
                return str(model_names.get(label_index, label_index))
            if isinstance(model_names, (list, tuple)) and 0 <= label_index < len(model_names):
                return str(model_names[label_index])
            return str(label_index)

        for index, poly in enumerate(polygons):
            if poly is None or len(poly) < 3:
                continue
            label_index = int(float(cls_np[index])) if index < len(cls_np) else None
            score = float(conf_np[index]) if index < len(conf_np) else 1.0
            model_label = _resolve_model_label(label_index)
            output_label = self._pick_output_label(label_index, model_label)
            points = []
            for point in poly:
                x = max(0.0, min(100.0, float(point[0] * 100.0)))
                y = max(0.0, min(100.0, float(point[1] * 100.0)))
                points.append([x, y])
            value = {
                "polygonlabels": [output_label],
                "points": points,
                "closed": True,
            }
            results.append({
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "polygonlabels",
                "value": value,
                "score": score,
            })
        return results

    def _run_inference(self, selected_model, local_path: str, model_task: str, model_family: str, imgsz: int, context: Optional[Dict], task: Dict):
        route_spec = build_route_spec('best', model_task, model_family)
        adapter = self.backend_registry.get_adapter(route_spec.backend)
        if adapter is None:
            logger.error("No backend adapter found for backend=%s", route_spec.backend)
            return []
        return adapter.run(
            selected_model=selected_model,
            local_path=local_path,
            model_task=model_task,
            imgsz=imgsz,
            context=context,
            task=task,
        )

    def _postprocess(self, inference_result, model_task: str):
        if model_task == 'segment':
            return self._build_segment_results(inference_result)
        if model_task == 'obb':
            return self._build_obb_results(inference_result)
        return self._build_detection_results(inference_result)

    def _resolve_local_path(self, image_url: str, task_id=None):
        local_upload_path = self._resolve_upload_local_file(image_url, task_id=task_id)
        if local_upload_path:
            return local_upload_path

        ls_host = (
            os.getenv('LABEL_STUDIO_URL')
            or os.getenv('LABEL_STUDIO_HOST')
            or os.getenv('LABEL_STUDIO_HOSTNAME')
        )
        normalized_url = image_url
        if isinstance(image_url, str) and image_url.startswith('/') and ls_host:
            normalized_url = urljoin(ls_host.rstrip('/') + '/', image_url.lstrip('/'))
            logger.info(
                "Normalized relative image url for task_id=%s: %s -> %s",
                task_id,
                image_url,
                normalized_url
            )

        try:
            return self.get_local_path(normalized_url, task_id=task_id)
        except Exception as exc:
            ls_token = os.getenv('LABEL_STUDIO_API_KEY') or os.getenv('LABEL_STUDIO_ACCESS_TOKEN')
            if not ls_token:
                logger.warning(
                    "LABEL_STUDIO_API_KEY/LABEL_STUDIO_ACCESS_TOKEN is not set, API download may fail with 401"
                )
            if not ls_host:
                if isinstance(image_url, str) and image_url.startswith('/'):
                    logger.error(
                        "Relative image url received but LABEL_STUDIO_URL/LABEL_STUDIO_HOSTNAME is not set"
                    )
                raise exc

            logger.warning(
                "get_local_path failed for task_id=%s, retry with ls_host=%s", task_id, ls_host
            )
            return self.get_local_path(
                normalized_url,
                task_id=task_id,
                ls_host=ls_host,
                ls_access_token=ls_token
            )

    def _resolve_upload_local_file(self, image_url: str, task_id=None):
        if not isinstance(image_url, str) or not image_url:
            return None

        parsed = urlparse(image_url)
        url_path = parsed.path if parsed.scheme else image_url
        upload_prefix = '/data/upload/'
        if not url_path.startswith(upload_prefix):
            return None

        decoded_upload_path = unquote(url_path[len(upload_prefix):])
        relative_upload_path = os.path.normpath(decoded_upload_path.replace('/', os.sep)).lstrip('\\/')

        candidate_base_dirs = []
        env_base_dir = os.getenv('LABEL_STUDIO_BASE_DATA_DIR')
        if env_base_dir:
            candidate_base_dirs.append(env_base_dir)

        local_appdata = os.getenv('LOCALAPPDATA')
        if local_appdata:
            candidate_base_dirs.append(os.path.join(local_appdata, 'label-studio', 'label-studio'))

        user_profile = os.path.expanduser('~')
        candidate_base_dirs.append(os.path.join(user_profile, '.local', 'share', 'label-studio'))

        for base_dir in candidate_base_dirs:
            candidate_path = os.path.join(base_dir, 'media', 'upload', relative_upload_path)
            if os.path.exists(candidate_path):
                logger.info(
                    "Resolved upload file from local filesystem for task_id=%s: %s",
                    task_id,
                    candidate_path
                )
                return candidate_path

        logger.debug(
            "Upload file not found in local candidate directories for task_id=%s: %s",
            task_id,
            url_path
        )
        return None

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        对输入任务进行目标检测，返回Label Studio格式的结果
        """
        model_name = kwargs.get('model_name')
        if not model_name and isinstance(context, dict):
            model_name = context.get('model_name')
        model_task = kwargs.get('model_task')
        if not model_task and isinstance(context, dict):
            model_task = context.get('model_task')
        model_family = kwargs.get('model_family')
        if not model_family and isinstance(context, dict):
            model_family = context.get('model_family')

        route_spec = build_route_spec(model_name, model_task, model_family)
        model_name = route_spec.model_name
        model_task = route_spec.model_task
        model_family = route_spec.model_family
        imgsz = route_spec.imgsz

        if not self._family_exists(model_family):
            logger.error("Unsupported model family: %s", model_family)
            return ModelResponse(predictions=[])
        if not self._is_task_supported(model_family, model_task):
            logger.error("Unsupported task=%s for family=%s", model_task, model_family)
            return ModelResponse(predictions=[])

        self._resolve_tag_mapping(model_task=model_task)

        logger.info(
            "Predict request received: tasks=%d model_task=%s model_family=%s model_name=%s",
            len(tasks),
            model_task,
            model_family,
            model_name,
        )

        model_version = f"{model_task}/{model_family}/{model_name}"
        selected_model = self.model_loader.get_or_load_model(route_spec, self.backend_registry)
        logger.info(
            "Using model_task=%s model_family=%s model_name=%s",
            model_task,
            model_family,
            model_name
        )

        predictions = []
        for task in tasks:
            task_id = task.get('id')
            image_url = task['data'].get('image') or task['data'].get('image_url')
            logger.info("Processing task_id=%s image_url=%s", task_id, image_url)
            if not image_url:
                logger.warning("task_id=%s has no image url", task_id)
                predictions.append({"result": [], "score": 0.0, "model_version": model_version})
                continue

            if selected_model is None:
                logger.warning("task_id=%s skipped because model is not available", task_id)
                predictions.append({"result": [], "score": 0.0, "model_version": model_version})
                continue

            # 下载图片到本地
            try:
                local_path = self._resolve_local_path(image_url, task_id=task_id)
            except Exception as exc:
                logger.error("task_id=%s failed to resolve local image path: %s", task_id, exc)
                predictions.append({"result": [], "score": 0.0, "model_version": model_version})
                continue

            results = self._run_inference(
                selected_model=selected_model,
                local_path=local_path,
                model_task=model_task,
                model_family=model_family,
                imgsz=imgsz,
                context=context,
                task=task,
            )
            inference_result = results[0] if results else None
            if inference_result is None:
                predictions.append({"result": [], "score": 0.0, "model_version": model_version})
                continue

            result = self._postprocess(inference_result, model_task=model_task)

            logger.info(
                "Inference done for task_id=%s: boxes=%d, labels=%s",
                task_id,
                len(result),
                [
                    (r.get("value", {}).get("rectanglelabels") or r.get("value", {}).get("polygonlabels") or ["?"])[0]
                    for r in result
                ][:10]
            )
            predictions.append({
                "result": result,
                "score": float(max((float(r.get("score", 0.0)) for r in result), default=0.0)),
                "model_version": model_version
            })

        logger.info("Predict response ready: predictions=%d", len(predictions))
        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        # 可选：实现训练逻辑
        pass


