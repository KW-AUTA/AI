import os
import json
import base64
import io
from PIL import Image
from typing import List, Dict, Tuple
from .element_matcher import MatchResult


def load_figma_json(json_path: str) -> dict:
    """Figma JSON 파일을 로드합니다."""
    with open(json_path, 'r') as f:
        return json.load(f)

def decode_base64_image(base64_str: str) -> Image.Image:
    """Base64 문자열을 PIL Image로 변환합니다."""
    return Image.open(io.BytesIO(base64.b64decode(base64_str.split(',',1)[1]))).convert('RGB')

def get_min_x(element: dict, min_val: float) -> float:
    """Figma 요소의 최소 x 좌표를 계산합니다."""
    if 'children' in element:
        for child in element['children']:
            if child['type'] == 'FRAME':
                min_val = min(min_val, get_min_x(child, min_val))
            else:
                min_val = min(min_val, child['absoluteX'])
    return min_val 


def _collect_yolo_annotations(
    node: Dict,
    base_x: float,
    base_y: float,
    img_w: float,
    img_h: float,
    type_map: Dict[str,int]
) -> List[Tuple[str, str, Tuple[int,float,float,float,float]]]:
    """
    node와 그 자식들을 순회하며 YOLO 포맷 annotation을 수집.
    """
    anns = []
    nx = node.get('absoluteX',0) - base_x
    ny = node.get('absoluteY',0) - base_y
    w  = node.get('width',0)
    h  = node.get('height',0)
    cls= type_map.get(node.get('type',''))
    if cls is not None and w>0 and h>0:
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
        anns.extend(_collect_yolo_annotations(ch, base_x, base_y, img_w, img_h, type_map))
    return anns


def frame_to_dict(
    frame: Dict,
    type_map: Dict[str,int] = {
        'TEXT':         0,
        'RECTANGLE':    1,
        'VECTOR':       1,
        'GROUP':        2,
        'INSTANCE':     3,
    }
) -> List[Tuple[int,float,float,float,float]]:
    """
    하나의 frame dict에서 YOLO .txt 파일 하나를 생성.
    파일명에 슬래시가 있으면 '_'로 바꿔 주고, 중복 방지까지 수행합니다.
    """
    base_x = frame.get('absoluteX',0)
    base_y = frame.get('absoluteY',0)
    W      = frame.get('width',1)
    H      = frame.get('height',1)

    # 1) 어노테이션 수집
    anns: List[Tuple[str, str, Tuple[int,float,float,float,float]]] = []
    for child in frame.get('children', []):
        anns.extend(_collect_yolo_annotations(child, base_x, base_y, W, H, type_map))
    return anns


def get_figma_match_info(figma_infos: List[Dict], matches: List[MatchResult]) -> List[MatchResult]:
    """
    interaction_list: 각 노드별 인터랙션 정보(dict)
    matches: 이미 매칭된 MatchResult 객체들이 담긴 리스트
    반환: 각 인터랙션마다 IOU가 가장 높은 MatchResult 객체 (interaction 정보가 덧붙여진 후)
    """
    interaction_match: List[MatchResult] = []

    for figma_info in figma_infos:
        # 1) 이 인터랙션의 bounding box 좌표 계산
        int_x1 = figma_info['x']
        int_y1 = figma_info['y']
        int_x2 = figma_info['x'] + figma_info['w']
        int_y2 = figma_info['y'] + figma_info['h']

        # 2) 가장 높은 IOU를 줄 match를 찾기 위해 초기화
        best_iou = 0.0
        best_match = None

        for match in matches:
            # match.figma_box가 (x1, y1, x2, y2)인지 확인
            f_x1, f_y1, f_x2, f_y2 = match.figma_box
            iou = get_iou((int_x1, int_y1, int_x2, int_y2), (f_x1, f_y1, f_x2, f_y2))
            if iou > best_iou:
                best_iou = iou
                best_match = match
        if best_match is not None:
            if 'interactions' in figma_info:
                inters = figma_info.get('interactions', [])
                if inters and inters[0].get('navigation') == 'NAVIGATE':
                    best_match.interaction  = inters[0]['navigation']
                    best_match.figma_dest   = inters[0]['destinationId']
            best_match.figma_id     = figma_info.get('id')
            best_match.figma_name   = figma_info.get('name')
            interaction_match.append(best_match)
            print(best_match.figma_name, '\n')
            

    return interaction_match



def get_iou(bbox1: Tuple[float,float,float,float], 
            bbox2: Tuple[float,float,float,float]) -> float:
    """
    두 개의 바운딩박스 bbox1, bbox2를 받아서 IoU를 계산하여 반환합니다.
    bbox format: (x1, y1, x2, y2) 
      - (x1, y1): 좌상단 좌표
      - (x2, y2): 우하단 좌표
    
    반환값:
        float: Intersection over Union 값 (0.0 ~ 1.0)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # 1) 교집합 영역의 좌표 계산
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    # 2) 교집합 너비/높이 계산 (겹치지 않으면 0)
    inter_width  = max(0.0, xi_max - xi_min)
    inter_height = max(0.0, yi_max - yi_min)
    inter_area   = inter_width * inter_height

    # 3) 각 박스의 면적 계산
    area1 = max(0.0, (x1_max - x1_min)) * max(0.0, (y1_max - y1_min))
    area2 = max(0.0, (x2_max - x2_min)) * max(0.0, (y2_max - y2_min))

    # 4) 합집합 면적 = area1 + area2 - inter_area
    union_area = area1 + area2 - inter_area

    # 5) IoU 계산 (union_area가 0이면 0 반환)
    if union_area <= 0:
        return 0.0

    iou = inter_area / union_area
    return iou