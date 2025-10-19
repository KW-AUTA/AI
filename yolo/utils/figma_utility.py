"""
Figma Utility Manager - 클래스 기반 Figma 유틸리티 처리

이 모듈은 Figma 관련 유틸리티 함수들을 클래스 기반으로 리팩토링한 버전입니다.
의존성 주입 패턴과 단일 책임 원칙을 적용했습니다.
"""

import json
import base64
import io
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ..core.models import MatchResult


@dataclass(frozen=True)
class FigmaUtilityConfig:
    """Figma 유틸리티 설정"""
    default_type_map: Dict[str, int] = None

    @classmethod
    def default(cls) -> 'FigmaUtilityConfig':
        """기본 설정 반환"""
        return cls(
            default_type_map={
                'TEXT': 0,
                'RECTANGLE': 1,
                'VECTOR': 1,
                'GROUP': 2,
                'INSTANCE': 3,
            }
        )

    def __post_init__(self):
        if self.default_type_map is None:
            object.__setattr__(self, 'default_type_map', self.default().default_type_map)


class FigmaUtilityManager:
    """
    Figma 관련 유틸리티 기능을 제공하는 클래스

    주요 기능:
    - Figma JSON 파일 로드
    - Base64 이미지 디코딩
    - YOLO 어노테이션 수집
    - IOU 계산
    - 매칭 정보 처리
    """

    def __init__(self, config: Optional[FigmaUtilityConfig] = None):
        """
        Args:
            config: Figma 유틸리티 설정. None이면 기본값 사용
        """
        self.config = config or FigmaUtilityConfig.default()

    @staticmethod
    def load_figma_json(json_path: str) -> dict:
        """Figma JSON 파일을 로드합니다.

        Args:
            json_path: JSON 파일 경로

        Returns:
            로드된 JSON 딕셔너리
        """
        with open(json_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def decode_base64_image(base64_str: str) -> Image.Image:
        """Base64 문자열을 PIL Image로 변환합니다.

        Args:
            base64_str: Base64 인코딩된 이미지 문자열

        Returns:
            PIL Image 객체
        """
        return Image.open(
            io.BytesIO(base64.b64decode(base64_str.split(',', 1)[1]))
        ).convert('RGB')

    @staticmethod
    def get_min_x(element: dict, min_val: float = 0) -> float:
        """Figma 요소의 최소 x 좌표를 재귀적으로 계산합니다.

        Args:
            element: Figma 요소 딕셔너리
            min_val: 초기 최소값

        Returns:
            최소 x 좌표
        """
        if 'children' in element:
            for child in element['children']:
                if child['data']['type'] == 'FRAME':
                    min_val = min(min_val, FigmaUtilityManager.get_min_x(child, min_val))
                else:
                    min_val = min(min_val, child['data']['absolutePosition']['x'])
        return min_val

    def collect_yolo_annotations(
        self,
        node: Dict,
        base_x: float,
        base_y: float,
        img_w: float,
        img_h: float,
        type_map: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """
        node와 그 자식들을 순회하며 YOLO 포맷 annotation을 수집합니다.

        Args:
            node: Figma 노드 딕셔너리
            base_x: 기준 x 좌표
            base_y: 기준 y 좌표
            img_w: 이미지 너비
            img_h: 이미지 높이
            type_map: 타입 매핑 딕셔너리

        Returns:
            수집된 어노테이션 리스트
        """
        if type_map is None:
            type_map = self.config.default_type_map

        anns = []
        nx = node.get('absoluteX', 0) - base_x
        ny = node.get('absoluteY', 0) - base_y
        w = node.get('width', 0)
        h = node.get('height', 0)
        cls = type_map.get(node.get('type', ''))

        if cls is not None and w > 0 and h > 0:
            ann = {
                'id': node.get('id', ''),
                'name': node.get('name', ''),
                'x': nx,
                'y': ny,
                'w': w,
                'h': h,
            }
            if 'interactions' in node:
                ann['interactions'] = node.get('interactions')
            anns.append(ann)

        for ch in node.get('children', []):
            anns.extend(self.collect_yolo_annotations(ch, base_x, base_y, img_w, img_h, type_map))

        return anns

    def frame_to_dict(
        self,
        frame: Dict,
        type_map: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """
        하나의 frame dict에서 YOLO 어노테이션을 생성합니다.

        Args:
            frame: Figma 프레임 딕셔너리
            type_map: 타입 매핑 딕셔너리

        Returns:
            YOLO 어노테이션 리스트
        """
        if type_map is None:
            type_map = self.config.default_type_map

        base_x = frame.get('absoluteX', 0)
        base_y = frame.get('absoluteY', 0)
        W = frame.get('width', 1)
        H = frame.get('height', 1)

        anns: List[Dict] = []
        for child in frame.get('children', []):
            anns.extend(self.collect_yolo_annotations(child, base_x, base_y, W, H, type_map))

        return anns

    @staticmethod
    def calculate_iou(
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """
        두 개의 바운딩박스 bbox1, bbox2를 받아서 IoU를 계산합니다.

        Args:
            bbox1: (x1, y1, x2, y2) 형식의 첫 번째 박스
            bbox2: (x1, y1, x2, y2) 형식의 두 번째 박스

        Returns:
            Intersection over Union 값 (0.0 ~ 1.0)
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # 교집합 영역의 좌표 계산
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        # 교집합 너비/높이 계산 (겹치지 않으면 0)
        inter_width = max(0.0, xi_max - xi_min)
        inter_height = max(0.0, yi_max - yi_min)
        inter_area = inter_width * inter_height

        # 각 박스의 면적 계산
        area1 = max(0.0, (x1_max - x1_min)) * max(0.0, (y1_max - y1_min))
        area2 = max(0.0, (x2_max - x2_min)) * max(0.0, (y2_max - y2_min))

        # 합집합 면적 = area1 + area2 - inter_area
        union_area = area1 + area2 - inter_area

        # IoU 계산 (union_area가 0이면 0 반환)
        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def get_figma_match_info(self, matches: List[MatchResult]) -> List[MatchResult]:
        """
        매칭된 결과에서 interaction이 있는 요소들을 IOU 기반으로 찾습니다.

        Args:
            matches: 매칭된 MatchResult 객체 리스트

        Returns:
            Interaction 정보가 추가된 MatchResult 리스트
        """
        interaction_match: List[MatchResult] = []

        for figma_info in matches.figma:
            # 인터랙션의 bounding box 좌표 계산
            int_x1 = figma_info['x']
            int_y1 = figma_info['y']
            int_x2 = figma_info['x'] + figma_info['w']
            int_y2 = figma_info['y'] + figma_info['h']

            # 가장 높은 IOU를 가진 매치 찾기
            best_iou = 0.0
            best_match = None

            for match in matches:
                f_x1, f_y1, f_x2, f_y2 = match.figma_element.box
                iou = self.calculate_iou(
                    (int_x1, int_y1, int_x2, int_y2),
                    (f_x1, f_y1, f_x2, f_y2)
                )
                if iou > best_iou:
                    best_iou = iou
                    best_match = match

            if best_iou < 0.1:
                continue

            if best_match is not None:
                if 'interactions' in figma_info:
                    inters = figma_info.get('interactions', [])
                    if inters and inters[0].get('navigation') == 'NAVIGATE':
                        best_match.interaction = inters[0]['navigation']
                        best_match.figma_dest = inters[0]['destinationId']
                best_match.figma_id = figma_info.get('id')
                best_match.figma_name = figma_info.get('name')
                interaction_match.append(best_match)
                print(best_match.figma_name, '\n')

        return interaction_match

    @classmethod
    def create(cls, config: Optional[FigmaUtilityConfig] = None) -> 'FigmaUtilityManager':
        """팩토리 메서드: FigmaUtilityManager 인스턴스 생성

        Args:
            config: 설정 객체 (None이면 기본값 사용)

        Returns:
            FigmaUtilityManager 인스턴스
        """
        return cls(config)


# 편의를 위한 전역 인스턴스
_default_manager = None


def get_default_manager() -> FigmaUtilityManager:
    """기본 FigmaUtilityManager 인스턴스를 반환합니다."""
    global _default_manager
    if _default_manager is None:
        _default_manager = FigmaUtilityManager.create()
    return _default_manager


# 편의 함수들 (IOU 계산은 정적 메서드이므로 직접 제공)
def calculate_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """IOU 계산 편의 함수"""
    return FigmaUtilityManager.calculate_iou(bbox1, bbox2)


# 별칭 (기존 코드 호환성)
get_iou = calculate_iou
