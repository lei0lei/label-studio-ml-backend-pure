import logging
import math
from typing import Optional

try:
    from label_studio_sdk.converter.utils import convert_yolo_obb_to_annotation
except ImportError:
    convert_yolo_obb_to_annotation = None

try:
    from label_studio_sdk.converter import brush as ls_brush
except ImportError:
    ls_brush = None


class LabelStudioResultBuilder:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def _pick_output_label(self, labels_in_config, label_index: Optional[int], model_label: Optional[str]):
        if not labels_in_config:
            return model_label or 'object'
        if label_index is not None and label_index < len(labels_in_config):
            return labels_in_config[label_index]
        if model_label and model_label in labels_in_config:
            return model_label
        return labels_in_config[0]

    def build_detection_results(self, inference_result, from_name: str, to_name: str, labels_in_config):
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
            output_label = self._pick_output_label(labels_in_config, label_index, model_label)
            value = {
                "rectanglelabels": [output_label],
                "x": float(100 * x_min / width),
                "y": float(100 * y_min / height),
                "width": float(100 * (x_max - x_min) / width),
                "height": float(100 * (y_max - y_min) / height),
                "rotation": 0.0,
            }
            results.append({
                "from_name": from_name,
                "to_name": to_name,
                "type": "rectanglelabels",
                "value": value,
                "score": score,
            })
        return results

    def build_obb_results(self, inference_result, from_name: str, to_name: str, labels_in_config):
        results = []
        obb = getattr(inference_result, 'obb', None)
        if obb is None:
            return self.build_detection_results(inference_result, from_name, to_name, labels_in_config)

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

        xyxyxyxy = _to_numpy(getattr(obb, 'xyxyxyxy', None))
        xywhr = getattr(obb, 'xywhr', None)
        xywhn = getattr(obb, 'xywhn', None)
        if xyxyxyxy is None and xywhr is None and xywhn is None:
            return self.build_detection_results(inference_result, from_name, to_name, labels_in_config)

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
            output_label = self._pick_output_label(labels_in_config, label_index, model_label)

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
                "from_name": from_name,
                "to_name": to_name,
                "type": "rectanglelabels",
                "value": value,
                "score": score,
            })

        self.logger.info("Built OBB results: %d boxes", len(results))
        return results

    def build_segment_results(self, inference_result, from_name: str, to_name: str, labels_in_config, result_type: str):
        results = []
        masks = getattr(inference_result, 'masks', None)
        if masks is None:
            return results

        polygons = getattr(masks, 'xyn', None)
        if polygons is None:
            return results

        if result_type == 'brushlabels':
            binary_masks = getattr(inference_result, 'binary_masks', None) or []
            if ls_brush is None:
                self.logger.error('label_studio_sdk brush converter is unavailable, cannot output brushlabels')
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
                output_label = self._pick_output_label(labels_in_config, label_index, model_label)
                rle = ls_brush.mask2rle(mask.astype('uint8') * 255)
                value = {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [output_label],
                }
                results.append({
                    'from_name': from_name,
                    'to_name': to_name,
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
            output_label = self._pick_output_label(labels_in_config, label_index, model_label)
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
                "from_name": from_name,
                "to_name": to_name,
                "type": "polygonlabels",
                "value": value,
                "score": score,
            })
        return results

    def build(self, inference_result, model_task: str, from_name: str, to_name: str, labels_in_config, result_type: str):
        if model_task == 'segment':
            return self.build_segment_results(
                inference_result=inference_result,
                from_name=from_name,
                to_name=to_name,
                labels_in_config=labels_in_config,
                result_type=result_type,
            )
        if model_task == 'obb':
            return self.build_obb_results(
                inference_result=inference_result,
                from_name=from_name,
                to_name=to_name,
                labels_in_config=labels_in_config,
            )
        return self.build_detection_results(
            inference_result=inference_result,
            from_name=from_name,
            to_name=to_name,
            labels_in_config=labels_in_config,
        )
