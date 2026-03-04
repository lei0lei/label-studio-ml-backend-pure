from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple
import re


FAMILY_TASKS: Dict[str, Set[str]] = {
    'yolov5': {'detect', 'segment'},
    'yolov8': {'detect', 'segment', 'obb'},
    'yolov9': {'detect', 'segment', 'obb'},
    'yolov10': {'detect', 'segment', 'obb'},
    'yolo11': {'detect', 'segment', 'obb'},
    'yolo26': {'detect', 'segment', 'obb'},
    'sam2': {'segment'},
}


@dataclass(frozen=True)
class RouteSpec:
    model_name: str
    model_task: str
    model_family: str
    imgsz: int
    backend: str


def normalize_task(model_task: Optional[str]) -> str:
    value = (model_task or 'detect').strip().lower()
    aliases = {
        'det': 'detect',
        'detection': 'detect',
        'bbox': 'detect',
        'box': 'detect',
        'seg': 'segment',
        'segmentation': 'segment',
        'oriented': 'obb',
        'rotated': 'obb',
    }
    return aliases.get(value, value)


def normalize_family(model_family: Optional[str]) -> str:
    value = (model_family or 'yolov8').strip().lower().replace('_', '').replace('-', '')
    aliases = {
        'yolov26': 'yolo26',
        'yolo26': 'yolo26',
        'yolov11': 'yolo11',
        'yolo11': 'yolo11',
    }
    return aliases.get(value, value)


def family_exists(model_family: str) -> bool:
    if model_family in FAMILY_TASKS:
        return True
    if model_family.startswith('yolo'):
        return True
    return False


def is_task_supported(model_family: str, model_task: str) -> bool:
    if model_family in FAMILY_TASKS:
        return model_task in FAMILY_TASKS[model_family]
    if model_family.startswith('yolo'):
        return model_task in {'detect', 'segment', 'obb'}
    return False


def resolve_backend(model_family: str) -> str:
    if model_family.startswith('sam2') or model_family == 'sam2':
        return 'sam2'
    return 'yolo'


def parse_model_name_and_imgsz(raw_model_name: Optional[str]) -> Tuple[str, int]:
    model_name = str(raw_model_name or 'best')
    imgsz = 640
    size_match = re.match(r'^(?P<base>.+)_(?P<size>\d+)$', model_name)
    if size_match:
        imgsz = int(size_match.group('size'))
    return model_name, imgsz


def build_route_spec(raw_model_name: Optional[str], raw_model_task: Optional[str], raw_model_family: Optional[str]) -> RouteSpec:
    model_name, imgsz = parse_model_name_and_imgsz(raw_model_name)
    model_task = normalize_task(raw_model_task)
    model_family = normalize_family(raw_model_family)
    backend = resolve_backend(model_family)
    return RouteSpec(
        model_name=model_name,
        model_task=model_task,
        model_family=model_family,
        imgsz=imgsz,
        backend=backend,
    )
