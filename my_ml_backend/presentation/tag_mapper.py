import logging
from dataclasses import dataclass


@dataclass
class TagMapping:
    from_name: str
    to_name: str
    labels_in_config: list
    result_type: str
    result_label_key: str


class TagMapper:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def _select_tag_candidates(self, model_task: str):
        if model_task == 'segment':
            return 'polygonlabels', 'polygonlabels', [
                ('BrushLabels', 'Image'),
                ('BrushLabels', 'HyperText'),
                ('PolygonLabels', 'Image'),
                ('PolygonLabels', 'HyperText'),
            ]

        if model_task == 'obb':
            return 'rectanglelabels', 'rectanglelabels', [
                ('RectangleLabels', 'Image'),
                ('RectangleLabels', 'HyperText'),
            ]

        return 'rectanglelabels', 'rectanglelabels', [
            ('RectangleLabels', 'Image'),
            ('Labels', 'Image'),
            ('RectangleLabels', 'HyperText'),
        ]

    def resolve(self, model_task: str, get_first_tag_occurence, parsed_label_config, fallback_from_name: str, fallback_to_name: str):
        default_result_type, default_label_key, candidates = self._select_tag_candidates(model_task)

        for control_type, object_type in candidates:
            try:
                from_name, to_name, _ = get_first_tag_occurence(control_type, object_type)
                labels_in_config = parsed_label_config.get(from_name, {}).get('labels', [])
                result_type = default_result_type
                result_label_key = default_label_key
                if control_type == 'BrushLabels':
                    result_type = 'brushlabels'
                    result_label_key = 'brushlabels'
                elif control_type == 'PolygonLabels':
                    result_type = 'polygonlabels'
                    result_label_key = 'polygonlabels'

                self.logger.info(
                    "Resolved label config mapping: control=%s object=%s from_name=%s to_name=%s labels=%d",
                    control_type,
                    object_type,
                    from_name,
                    to_name,
                    len(labels_in_config)
                )
                return TagMapping(
                    from_name=from_name,
                    to_name=to_name,
                    labels_in_config=labels_in_config,
                    result_type=result_type,
                    result_label_key=result_label_key,
                )
            except Exception:
                continue

        self.logger.warning(
            "No compatible control/object tag found in label config. Using fallback from_name=%s to_name=%s",
            fallback_from_name,
            fallback_to_name,
        )
        return TagMapping(
            from_name=fallback_from_name,
            to_name=fallback_to_name,
            labels_in_config=[],
            result_type=default_result_type,
            result_label_key=default_label_key,
        )
