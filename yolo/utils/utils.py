"""
레거시 유틸리티 함수들 - 호환성 유지를 위한 래퍼

새로운 코드에서는 FigmaUtilityManager 클래스 사용을 권장합니다.
이 모듈의 함수들은 기존 코드 호환성을 위해 유지됩니다.
"""

import os
import json
import base64
import io
from PIL import Image
from typing import List, Dict, Tuple
from ..core.models import MatchResult
from .figma_utility import FigmaUtilityManager, get_default_manager


# 레거시 함수들 - 새로운 클래스 기반 구현으로 위임
def load_figma_json(json_path: str) -> dict:
    """Figma JSON 파일을 로드합니다.

    레거시 함수: FigmaUtilityManager.load_figma_json() 사용을 권장합니다.
    """
    return FigmaUtilityManager.load_figma_json(json_path)


def decode_base64_image(base64_str: str) -> Image.Image:
    """Base64 문자열을 PIL Image로 변환합니다.

    레거시 함수: FigmaUtilityManager.decode_base64_image() 사용을 권장합니다.
    """
    return FigmaUtilityManager.decode_base64_image(base64_str)


def get_min_x(element: dict, min_val: float) -> float:
    """Figma 요소의 최소 x 좌표를 계산합니다.

    레거시 함수: FigmaUtilityManager.get_min_x() 사용을 권장합니다.
    """
    return FigmaUtilityManager.get_min_x(element, min_val) 


def _collect_yolo_annotations(
    node: Dict,
    base_x: float,
    base_y: float,
    img_w: float,
    img_h: float,
    type_map: Dict[str, int]
) -> List[Dict]:
    """
    node와 그 자식들을 순회하며 YOLO 포맷 annotation을 수집.

    레거시 함수: FigmaUtilityManager.collect_yolo_annotations() 사용을 권장합니다.
    """
    manager = get_default_manager()
    return manager.collect_yolo_annotations(node, base_x, base_y, img_w, img_h, type_map)


def frame_to_dict(
    frame: Dict,
    type_map: Dict[str, int] = None
) -> List[Dict]:
    """
    하나의 frame dict에서 YOLO 어노테이션을 생성합니다.

    레거시 함수: FigmaUtilityManager.frame_to_dict() 사용을 권장합니다.
    """
    manager = get_default_manager()
    return manager.frame_to_dict(frame, type_map)


def get_figma_match_info(matches: List[MatchResult]) -> List[MatchResult]:
    """
    interaction_list: 각 노드별 인터랙션 정보(dict)
    matches: 이미 매칭된 MatchResult 객체들이 담긴 리스트
    반환: 각 인터랙션마다 IOU가 가장 높은 MatchResult 객체 (interaction 정보가 덧붙여진 후)

    레거시 함수: FigmaUtilityManager.get_figma_match_info() 사용을 권장합니다.
    """
    manager = get_default_manager()
    return manager.get_figma_match_info(matches)


def get_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    두 개의 바운딩박스 bbox1, bbox2를 받아서 IoU를 계산하여 반환합니다.
    bbox format: (x1, y1, x2, y2)
      - (x1, y1): 좌상단 좌표
      - (x2, y2): 우하단 좌표

    반환값:
        float: Intersection over Union 값 (0.0 ~ 1.0)

    레거시 함수: FigmaUtilityManager.calculate_iou() 사용을 권장합니다.
    """
    return FigmaUtilityManager.calculate_iou(bbox1, bbox2)