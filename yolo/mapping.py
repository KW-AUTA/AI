import os
import logging
from PIL import Image, ImageDraw, ImageFont
from .element_matcher import ElementExtractor
from .models import ExtractedElement, FigmaFare, FigmaElement, MatchResult
from .visualizer import Visualizer
from .web_navigator import WebNavigator
from .utils import load_figma_json, decode_base64_image, get_min_x
import numpy as np
import time
import random
import torch
import requests
from routes.dto.response import RoutingMappingInfo, InteractionMappingInfo, GeneralMappingInfo, BaseMappingInfo
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from .tree_loader import TreeNode
from contextlib import contextmanager
import concurrent.futures
import cProfile
import pstats
from .element_matcher import ElementExtractor, non_max_suppression
import matplotlib.pyplot as plt
from PIL import ImageChops
import io
from torchvision import transforms
from .error_list import *
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# timecheck
@contextmanager
def time_check(name: str):
    start_time = time.time()
    yield
    end_time = time.time()
    logging.info(f"{name}: {end_time - start_time} seconds")

def timecheck(func):
    def wrapper(*args, **kwargs):
        with time_check(func.__name__):
            return func(*args, **kwargs)
    return wrapper

def seed_everything(seed: int = 42):
    """랜덤 시드 설정""" 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

def catergorize_match(match: MatchResult) -> List[str]:
    """매칭된 요소들을 카테고리별로 분류"""
    category = []
    if abs(match.figma.extracted.box[0] - match.web.box[0]) > 10:
        category.append('different_x')
    if abs(match.figma.extracted.box[1] - match.web.box[1]) > 10:
        category.append('different_y')
    
    w_f = match.figma.extracted.box[2] - match.figma.extracted.box[0]    
    h_f = match.figma.extracted.box[3] - match.figma.extracted.box[1]
    w_w = match.web.box[2] - match.web.box[0]
    h_w = match.web.box[3] - match.web.box[1]

    if abs(w_f - w_w) > 10 or abs(h_f - h_w) > 10:
        category.append('different_size')
    
    if len(category) == 0:
        category.append('same')
    
    logging.debug(f"Categorized match for : {category}")
    return category

def get_mapping_info(matches: List[MatchResult]) -> List[BaseMappingInfo]:
    """매칭 결과를 매핑 정보로 변환"""
    mapping_infos = []
    for match in matches:
        mapping_info = BaseMappingInfo(
            componentName=match.figma.name,
            destinationFigmaPage=match.figma.dest if match.figma.dest else "",
            destinationUrl=match.web.dest if match.web.dest else "",
            actualUrl=match.web.dest if match.web.dest else "",
            failReason=", ".join(match.errorCategories) if match.errorCategories != [NORMAL] else "",
            isSuccess=match.errorCategories == [NORMAL],
            isRouting=match.isRouting
        )
        mapping_infos.append(mapping_info)
    logging.info(f"Generated {len(mapping_infos)} mapping infos.")
    return mapping_infos

def process_images(figma_url: str, web_navigator: WebNavigator, target_height: int) -> Tuple[Image.Image, Image.Image, List[Dict]]:
    """이미지 처리 및 데이터 추출"""
    logging.info("Processing images...")
    response = requests.get(figma_url)
    response.raise_for_status()

    figma_json = response.json()
    root = figma_json['tree'][0]
    frame_img = decode_base64_image(root['data']['image'])
    frames = figma_json['tree']
    web_img = web_navigator.capture_full_page_with_scroll(frame_img, target_height)
    scale = web_img.width / frame_img.width
    web_img = web_img.resize((frame_img.width, int(web_img.height / scale)))
    logging.info("Finished processing images.")
    return frame_img, web_img, frames

def extract_texts(img: Image.Image, matcher: ElementExtractor, boxes: List[Tuple[int, int, int, int]]) -> List[str]:
    """기존 순차 처리 방식으로 텍스트 추출"""
    texts = []
    for idx, box in enumerate(boxes):
        text_margin = 10
        margin_box = (box[0] - text_margin, box[1] - text_margin, box[2] + text_margin, box[3] + text_margin)
        text = matcher.extract_text(img, margin_box, box_idx=idx)
        texts.append(text)
    return texts

def extract_elements(img: Image.Image, start_height: int, windowing_height: int, matcher: ElementExtractor, iou_threshold: float = 0.5) -> List[ExtractedElement]:
    """이미지에서 요소 추출 (순차 처리 버전)"""
    logging.info(f"Extracting elements from image with height {img.height} using sequential windowing...")
    
    windows_to_process = []
    current_height = 0
    while current_height < img.height:
        crop_img = img.crop((0, current_height, img.width, min(current_height + windowing_height, img.height)))
        windows_to_process.append((crop_img, current_height))
        current_height += windowing_height * 0.5

    all_boxes = []
    all_scores = []
    all_cls = []
    all_features = []

    # 각 창에 대해 순차적으로 탐지 및 특징 추출 수행
    def process_window(crop_img, original_height):
        # 1. 탐지 및 특징 추출을 한 번에 수행 (multi-resolution 활성화)
        boxes, scores, cls, feat_map, original_img_size = matcher.detect_boxes_yolo(
            crop_img,
            extract_features=True,
            use_multires=False
        )
        if len(boxes) == 0:
            return None, None, None, None
        # 2. 특징 추출
        features = matcher.extract_features_from_map(feat_map, boxes, original_img_size)
        
        # 3. Y 좌표 조정
        boxes[:, 1] += original_height
        boxes[:, 3] += original_height
        
        return boxes, scores, cls, features
    logging.info(f"Processing {len(windows_to_process)} windows...")
    start_time = time.time()
    for crop_img, h in windows_to_process:
        boxes, scores, cls, features = process_window(crop_img, h)
        if boxes is not None:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_cls.append(cls)
            all_features.append(features)
    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time} seconds for figma extraction")

    if not all_boxes:
        logging.info("No elements were extracted.")
        return []

    # 모든 결과를 하나로 합침
    final_boxes = np.vstack(all_boxes)
    final_scores = np.concatenate(all_scores)
    final_cls = np.concatenate(all_cls)
    final_features = np.vstack(all_features)

    nms_boxes = final_boxes
    nms_cls = final_cls
    nms_features = final_features
    
    extracted_elements = []
    if nms_boxes.size > 0:
        for i, box in enumerate(nms_boxes):
            feature = nms_features[i]
            cls = nms_cls[i]
            extracted_elements.append(ExtractedElement(box=box, feature=feature, text=None, cls=cls))

    logging.info(f"Extracted and filtered {len(extracted_elements)} elements.")
    return extracted_elements


def get_interaction_by_id(id: str, interactions: List[Dict]) -> Dict:
    for interaction in interactions:
        if interaction['interactionType']['sourceId'] == id:
            return interaction
    return None


def check_interaction_overlay(match: MatchResult, web_navigator: WebNavigator, figma_raw: List[Dict], web_img: Image.Image, interaction: Dict) -> List[BaseMappingInfo]:
    """Enhanced overlay interaction checking with ScreenSim model"""
    return_matches = []

    # 메인 윈도우 핸들 저장
    main_window = web_navigator.driver.current_window_handle

    # 새 탭에서 현재 페이지 복제
    print("duplicate current page")
    web_navigator.driver.execute_script("window.open(window.location.href, '_blank');")
    # 탭 전환
    print("switch to new tab")
    web_navigator.driver.switch_to.window(web_navigator.driver.window_handles[-1])
    time.sleep(1)

    center_x = float(match.web.box[0] + match.web.box[2]) / 2
    center_y = float(match.web.box[1] + match.web.box[3]) / 2
    element, xpath = web_navigator.get_element_at_coordinate_and_xpath(center_x, center_y)
    time.sleep(1)

    if element is not None and xpath is not None:
        current_url = web_navigator.driver.current_url
        web_navigator.scroll_to_y(center_y)
        before_img = Image.open(io.BytesIO(web_navigator.driver.get_screenshot_as_png()))
        time.sleep(1)
        element.click()
        time.sleep(1)
        click_url = web_navigator.driver.current_url

        if click_url != current_url:
            return_matches.append(InteractionMappingInfo(
                type="INTERACTION",
                componentName=match.figma.name,
                expectedAction="OVERLAY",
                actualAction="NAVIGATE",
                failReason=I_ERROR_NAVIGATE_NOT_OVERLAY,
                isSuccess=False,
            ))
            # 세션 무효화 방지: 메인 윈도우로 먼저 전환 후 탭 닫기
            try:
                current_tab = web_navigator.driver.current_window_handle
                web_navigator.driver.switch_to.window(main_window)
                if len(web_navigator.driver.window_handles) > 1 and current_tab != main_window:
                    web_navigator.driver.switch_to.window(current_tab)
                    web_navigator.driver.close()
                    web_navigator.driver.switch_to.window(main_window)
            except Exception as e:
                logging.warning(f"Error closing window: {e}")
            return return_matches

        overlay_figma_img = get_img_by_id(interaction['interactionType']['destinationId'], figma_raw)
        screenshot_data = web_navigator.driver.get_screenshot_as_png()
        overlay_web_img = Image.open(io.BytesIO(screenshot_data))

        if overlay_figma_img is not None:
            diff_img = ImageChops.difference(before_img, overlay_web_img)

            if diff_img.getbbox() is None:
                return_matches.append(InteractionMappingInfo(
                    type="INTERACTION",
                    componentName=match.figma.name,
                    expectedAction="OVERLAY",
                    actualAction="None",
                    failReason=I_ERROR_OVERLAY_NOT_FOUND,
                    isSuccess=False,
                ))
            elif diff_img.getbbox() is not None:
                # Extract the overlay region from diff_img
                bbox = diff_img.getbbox()
                overlay_region = overlay_web_img.crop(bbox)

                # Compare using ScreenSim model
                overlay_region_array = np.array(overlay_region)
                overlay_figma_img_array = np.array(overlay_figma_img)

                Image.fromarray(overlay_region_array).save("overlay_region_array.png")
                Image.fromarray(overlay_figma_img_array).save("overlay_figma_img_array.png")

                current_dir = os.path.dirname(__file__)
                model_path = os.path.join(current_dir, "..", "models_weights", "screensim-resnet-uda+web7k.torchscript")
                m = torch.jit.load(model_path)
                img_transforms = transforms.Compose([
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                # numpy array를 PIL Image로 변환
                if isinstance(overlay_region_array, np.ndarray):
                    overlay_region_array = Image.fromarray(overlay_region_array.astype(np.uint8))
                if isinstance(overlay_figma_img_array, np.ndarray):
                    overlay_figma_img_array = Image.fromarray(overlay_figma_img_array.astype(np.uint8))

                overlay_region_array = img_transforms(overlay_region_array)
                overlay_figma_img_array = img_transforms(overlay_figma_img_array)

                embedding_region = m(overlay_region_array.unsqueeze(0))
                embedding_figma = m(overlay_figma_img_array.unsqueeze(0))
                dist_same = torch.linalg.norm(embedding_region - embedding_figma)
                margin = (0.2 + 0.5) / 2  # average of margin_pos and margin_neg
                print("same pair dist: {:.3f} same screen? ".format(float(dist_same)) + str((float(dist_same) < margin)))

                if float(dist_same) < margin:
                    return_matches.append(InteractionMappingInfo(
                        type="INTERACTION",
                        componentName=match.figma.name,
                        expectedAction="OVERLAY",
                        actualAction="OVERLAY",
                        failReason="",
                        isSuccess=True,
                    ))
                else:
                    return_matches.append(InteractionMappingInfo(
                        type="INTERACTION",
                        componentName=match.figma.name,
                        expectedAction="OVERLAY",
                        actualAction="OVERLAY",
                        failReason=I_ERROR_DIFFERENT_OVERLAY,
                        isSuccess=False,
                    ))

        # 세션 무효화 방지: 메인 윈도우로 먼저 전환 후 탭 닫기
        try:
            current_tab = web_navigator.driver.current_window_handle
            web_navigator.driver.switch_to.window(main_window)
            if len(web_navigator.driver.window_handles) > 1 and current_tab != main_window:
                web_navigator.driver.switch_to.window(current_tab)
                web_navigator.driver.close()
                web_navigator.driver.switch_to.window(main_window)
        except Exception as e:
            logging.warning(f"Error closing window in overlay check: {e}")

    return return_matches


def check_interaction_navigate(match: MatchResult, web_navigator: WebNavigator, figma_raw: List[Dict], web_img: Image.Image, interaction: Dict) -> List[BaseMappingInfo]:
    """Enhanced navigate interaction checking with afterInteraction BACK support"""
    return_matches = []
    original_web_image = Image.open(io.BytesIO(web_navigator.driver.get_screenshot_as_png()))

    center_x = float(match.web.box[0] + match.web.box[2]) / 2
    center_y = float(match.web.box[1] + match.web.box[3]) / 2
    element, xpath = web_navigator.get_element_at_coordinate_and_xpath(center_x, center_y)

    if element is not None and xpath is not None:
        urls = web_navigator.get_url_in_new_tab(xpath)
        return_matches.append(RoutingMappingInfo(
            type="ROUTING",
            componentName=match.figma.name,
            destinationFigmaPage=interaction['interactionType']['destinationId'],
            destinationUrl=urls,
            actualUrl=urls,
            failReason=NORMAL,
            isSuccess=True,
        ))
        logging.debug(f"Found URL for {match.figma.name}: {urls}")
    else:
        return_matches.append(RoutingMappingInfo(
            type="ROUTING",
            componentName=match.figma.name,
            destinationFigmaPage=interaction['interactionType']['destinationId'],
            destinationUrl="",
            actualUrl="",
            failReason=NORMAL,
            isSuccess=True,
        ))

    # Handle afterInteraction BACK
    if "afterInteraction" in interaction and len(interaction['afterInteraction']) > 0:
        for after_interaction in interaction['afterInteraction']:
            if after_interaction['interactionType']['navigation'] == 'BACK':
                web_navigator.driver.back()
                back_image = Image.open(io.BytesIO(web_navigator.driver.get_screenshot_as_png()))
                diff_img = ImageChops.difference(original_web_image, back_image)

                if diff_img.getbbox() is not None:
                    return_matches.append(InteractionMappingInfo(
                        type="INTERACTION",
                        componentName=match.figma.name,
                        expectedAction="BACK",
                        actualAction="BACK",
                        failReason=NORMAL,
                        isSuccess=True,
                    ))
                else:
                    return_matches.append(InteractionMappingInfo(
                        type="INTERACTION",
                        componentName=match.figma.name,
                        expectedAction="BACK",
                        actualAction="BACK",
                        failReason=I_ERROR_DIFFERENT_BACK,
                        isSuccess=False,
                    ))

    web_navigator.driver.close()
    web_navigator.driver.switch_to.window(web_navigator.driver.window_handles[0])
    return return_matches


def match_interaction(matcher: ElementExtractor, matches: List[MatchResult], web_navigator: WebNavigator, web_img: Image.Image, interactions: List[Dict], figma_tree: TreeNode, figma_raw: List[Dict]) -> List[BaseMappingInfo]:
    """Enhanced interaction matching using check_interaction_overlay and check_interaction_navigate"""
    return_matches = []
    for match in matches:
        interaction = get_interaction_by_id(match.figma.id, interactions)
        print("##", interaction['interactionType']['navigation'], "##")

        if interaction['interactionType']['navigation'] == 'NAVIGATE':
            return_matches.extend(check_interaction_navigate(match, web_navigator, figma_raw, web_img, interaction))
        elif interaction['interactionType']['navigation'] == 'OVERLAY':
            return_matches.extend(check_interaction_overlay(match, web_navigator, figma_raw, web_img, interaction))

    return return_matches
def process_matches(matcher: ElementExtractor, matches: List[MatchResult], web_navigator: WebNavigator, interactions: List[Dict], figma_tree: TreeNode, figma_raw: List[Dict]) -> List[BaseMappingInfo]:
    """매칭 결과 처리"""
    logging.info("Processing matches...")
    return_matches = []

    for match in matches:
        return_matches.append(GeneralMappingInfo(
            type="GENERAL",
            componentName=match.figma.name,
            failReason=", ".join(match.errorCategories) if match.errorCategories != [NORMAL] else "",
            isSuccess=match.errorCategories == [NORMAL],
        ))        
    return return_matches

def load_figma_json(json_url: str) -> Dict:
    logging.info(f"Loading Figma JSON from {json_url}")
    response = requests.get(json_url)
    response.raise_for_status()
    figma_json = response.json()
    logging.info("Figma JSON loaded successfully.")
    return figma_json

def convert_raw_to_tree(figma_tree: Dict, root_image: Image.Image, matcher: ElementExtractor, level: int = 0) -> TreeNode:
    logging.debug(f"Converting raw data to tree at level {level}")
    converted_tree = TreeNode(FigmaElement(
        id=figma_tree['data']['id'],
        name=figma_tree['data']['name'],
        absolute_position=figma_tree['data']['absolutePosition'],
        relative_position=figma_tree['data']['relativePosition'],
        absolute_render_position=figma_tree['data']['absoluteRenderPosition'],
        relative_render_position=figma_tree['data']['relativeRenderPosition'],
    ))
    for child in figma_tree['children']:
        converted_tree.add_child(convert_raw_to_tree(child, root_image, matcher, level + 1))

    return converted_tree
    
def get_frame_by_name(figma_tree: Dict, name: str) -> Optional[TreeNode]:
    if figma_tree['data']['name'] == name:
        return figma_tree
    for child in figma_tree['children']:
        result = get_frame_by_name(child, name)
        if result is not None:
            return result
    return None

def find_best_iou_node(
    root: TreeNode,
    target_rect: Tuple[float, float, float, float]
) -> Tuple[Optional[TreeNode], float]:
    """
    root:    트리의 루트 노드
    target_rect: (x1, y1, x2, y2) 형태의 비교 대상 박스

    반환값: (best_node, best_iou)
      - best_node: IoU가 가장 큰 TreeNode (없으면 None)
      - best_iou:  해당 최대 IoU 값
    """
    best_node: Optional[TreeNode] = None
    best_iou: float = 0.0

    def traverse(node: TreeNode):
        nonlocal best_node, best_iou

        # 1) 노드의 절대 좌표를 rect 박스로 변환
        figma_rect = node.data.absolute_position  # {"x":…, "y":…, "width":…, "height":…}
        node_rect = (
            figma_rect["x"],
            figma_rect["y"],
            figma_rect["x"] + figma_rect["width"],
            figma_rect["y"] + figma_rect["height"],
        )

        # 2) IoU 계산
        iou = calculate_iou(node_rect, target_rect)

        # 3) 최대값 갱신
        if iou > best_iou:
            best_iou = iou
            best_node = node

        # 4) 자식 노드도 재귀 탐색
        for child in node.children:
            traverse(child)

    traverse(root)
    return best_node, best_iou

def match_all_extracted(
    root: TreeNode,
    extracted_list: List[ExtractedElement],
    interactions: List[Dict]
) -> List[FigmaFare]:
    """
    Iterates through each tree node to find the best matching extracted element,
    prioritizing nodes with interactions.
    It resolves conflicts to ensure each extracted element is matched with at most one node.
    """

    logging.info(f"Matching tree nodes to {len(extracted_list)} extracted boxes, prioritizing interactions.")
    
    interaction_source_ids = {interaction['interactionType']['sourceId'] for interaction in interactions}
    
    all_nodes = []
    def get_all_nodes(node: TreeNode):
        all_nodes.append(node)
        for child in node.children:
            get_all_nodes(child)
    get_all_nodes(root)

    interaction_nodes = [node for node in all_nodes if node.data.id in interaction_source_ids]
    other_nodes = [node for node in all_nodes if node.data.id not in interaction_source_ids]

    final_matches = []
    remaining_extracted = list(extracted_list)
    
    # --- Helper function to perform matching and conflict resolution ---
    def find_and_resolve_matches(nodes_to_match, available_extracted):
        node_matches = []
        for node in nodes_to_match:
            figma_rect_dict = node.data.absolute_position
            node_rect = (
                figma_rect_dict["x"],
                figma_rect_dict["y"],
                figma_rect_dict["x"] + figma_rect_dict["width"],
                figma_rect_dict["y"] + figma_rect_dict["height"],
            )

            best_extracted = None
            best_iou = 0.0
            
            for extracted in available_extracted:
                iou = ElementExtractor.calculate_iou(node_rect, extracted.box)
                if iou > best_iou:
                    best_iou = iou
                    best_extracted = extracted
            
            if best_extracted is not None and best_iou > 0:
                node_matches.append({'node': node, 'extracted': best_extracted, 'iou': best_iou})

        # Resolve conflicts
        best_node_for_extracted = {}
        for match in node_matches:
            extracted_id = id(match['extracted'])
            if extracted_id not in best_node_for_extracted or match['iou'] > best_node_for_extracted[extracted_id]['iou']:
                best_node_for_extracted[extracted_id] = match
        
        resolved_matches = list(best_node_for_extracted.values())
        matched_extracted_ids = {id(m['extracted']) for m in resolved_matches}
        
        return resolved_matches, matched_extracted_ids

    # --- 1. Match interaction nodes ---
    logging.info(f"Attempting to match {len(interaction_nodes)} interaction nodes.")
    interaction_match_results, matched_extracted_ids_1 = find_and_resolve_matches(interaction_nodes, remaining_extracted)
    for match_info in interaction_match_results:
        final_matches.append(
            FigmaFare(
                id=match_info['node'].data.id,
                name=match_info['node'].data.name,
                box=match_info['node'].data.absolute_position,
                extracted=match_info['extracted']
            )
        )
    logging.info(f"Matched {len(final_matches)} interaction nodes.")

    # --- 2. Match other nodes with remaining extracted elements ---
    remaining_extracted = [ext for ext in remaining_extracted if id(ext) not in matched_extracted_ids_1]
    logging.info(f"Attempting to match {len(other_nodes)} other nodes with {len(remaining_extracted)} remaining boxes.")
    
    if other_nodes and remaining_extracted:
        other_match_results, _ = find_and_resolve_matches(other_nodes, remaining_extracted)
        for match_info in other_match_results:
            final_matches.append(
                FigmaFare(
                    id=match_info['node'].data.id,
                    name=match_info['node'].data.name,
                    box=match_info['node'].data.absolute_position,
                    extracted=match_info['extracted']
                )
            )
        logging.info(f"Matched {len(other_match_results)} other nodes.")


    logging.info(f"Found {len(final_matches)} unique matched nodes in total.")
    return final_matches

def prune_tree(
    node: TreeNode,
    matched: List[FigmaFare]
) -> Optional[TreeNode]:
    """
    Prune the tree so that only matched nodes and their ancestors remain.
    Returns a new TreeNode or None if the subtree contains no matches.
    """
    matched_ids = {m.id for m in matched}

    def _prune_recursive(current_node: TreeNode) -> Optional[TreeNode]:
        """Helper function to recursively prune the tree."""
        # Prune children first
        pruned_children = []
        for child in current_node.children:
            pruned_child = _prune_recursive(child)
            if pruned_child:
                pruned_children.append(pruned_child)

        # Check if the current node is matched or is an ancestor of a matched node
        is_matched = current_node.data.id in matched_ids
        is_ancestor = len(pruned_children) > 0

        if is_matched or is_ancestor:
            # Create a new node to avoid modifying the original tree
            new_node = TreeNode(current_node.data)
            for child in pruned_children:
                new_node.add_child(child)
            return new_node

        # Prune the node if it's not matched and not an ancestor
        return None

    return _prune_recursive(node)

def get_id_from_tree(figma_tree: TreeNode) -> List[str]:
    ids = []
    for node in figma_tree.children:
        ids.append(node.data.id)
        ids.extend(get_id_from_tree(node))
    return ids

def fare_figma_extracted(
    figma_tree: TreeNode,
    figma_extracted: List[ExtractedElement],
    interactions: List[Dict]  # Add interactions parameter
) -> List[FigmaFare]:
    """
    Prune figma_tree to include only nodes best-matched to extracted YOLO boxes,
    prioritizing nodes with interactions.
    """
    # Pass interactions to the matching function
    matched_nodes = match_all_extracted(figma_tree, figma_extracted, interactions)
    logging.info("Finished pruning tree.")
    return matched_nodes

def load_figma_data(json_url: str) -> List[Dict]:
    figma_json = load_figma_json(json_url)
    figma_tree = figma_json['tree']
    interactions = figma_json['interactions']
    return figma_tree, interactions

def tree_to_mermaid(root, name_map=None):
    """
    TreeNode를 Mermaid flowchart 형식으로 변환합니다.
    """
    lines = ["flowchart TD"]
    if name_map is None:
        name_map = {}
    counter = [0]

    def node_name(node):
        # Mermaid 노드 식별자 (n0, n1, …) 생성
        if node not in name_map:
            name_map[node] = f"n{counter[0]}"
            counter[0] += 1
        return name_map[node]

    def recurse(node):
        nid = node_name(node)
        label = node.data.name.replace('"', '\"')
        lines.append(f'    {nid}["{label}"]')
        for child in node.children:
            cid = node_name(child)
            recurse(child)
            lines.append(f"    {nid} --> {cid}")

    recurse(root)
    return "\n".join(lines)

def get_img_by_id(id: str, figma_raw: List[Dict]) -> Image.Image:
    for figma_element in figma_raw:
        if figma_element['data']['id'] == id:
            min_x = get_min_x(figma_element, 0)
            img = decode_base64_image(figma_element['data']['image'])
            img = img.crop((-min_x, 0, figma_element['data']['absolutePosition']['width'] - min_x, figma_element['data']['absolutePosition']['height'])) 
            return img
        
    return None


def categorize_match(match: MatchResult) -> List[str]:
    """매칭된 요소들을 카테고리별로 분류"""
    category = []
    if abs(match.figma.extracted.box[0] - match.web.box[0]) > 10:
        category.append(G_ERROR_COORDINATE_X)
    if abs(match.figma.extracted.box[1] - match.web.box[1]) > 10:
        category.append(G_ERROR_COORDINATE_Y)

    w_f = match.figma.extracted.box[2] - match.figma.extracted.box[0]
    h_f = match.figma.extracted.box[3] - match.figma.extracted.box[1]
    w_w = match.web.box[2] - match.web.box[0]
    h_w = match.web.box[3] - match.web.box[1]

    if abs(w_f - w_w) > 10 or abs(h_f - h_w) > 10:
        category.append(G_ERROR_SIZE)

    if len(category) == 0:
        category.append(NORMAL)

    logging.debug(f"Categorized match for : {category}")
    return category

def _save_match_crops(figma_img: Image.Image, web_img: Image.Image, matches: List[MatchResult], out_root: Optional[str] = None) -> str:
    """Save side-by-side crops for each match with score annotations."""
    def _clamp_box(box, w, h):
        x1, y1, x2, y2 = map(float, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1: x2 = min(w, x1 + 1)
        if y2 <= y1: y2 = min(h, y1 + 1)
        return int(x1), int(y1), int(x2), int(y2)

    ts = time.strftime('%Y%m%d-%H%M%S')
    if out_root is None:
        out_root = os.path.join(os.path.dirname(__file__), 'debug_sim', 'crops', ts)
    os.makedirs(out_root, exist_ok=True)

    fw, fh = figma_img.size
    ww, wh = web_img.size

    def _safe_name(s: str) -> str:
        return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in (s or ''))[:40]

    for idx, m in enumerate(matches):
        if m.figma is None or m.web is None:
            continue
        fx1, fy1, fx2, fy2 = _clamp_box(m.figma.extracted.box, fw, fh)
        wx1, wy1, wx2, wy2 = _clamp_box(m.web.box, ww, wh)
        crop_f = figma_img.crop((fx1, fy1, fx2, fy2))
        crop_w = web_img.crop((wx1, wy1, wx2, wy2))

        # Compose side-by-side with header for text
        pad = 6
        header_h = 54
        h = max(crop_f.height, crop_w.height) + header_h + pad * 2
        w = crop_f.width + crop_w.width + pad * 3
        canvas = Image.new('RGB', (w, h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        # Header text: scores and texts
        header = (
            f"score={m.score:.2f} | feat={m.feature_similarity:.2f} "
            f"text={m.text_similarity:.2f} size={m.size_similarity:.2f} coord={m.coordinate_similarity:.2f}"
        )
        sub = (
            f"fig: '{(m.figma.extracted.text or '').strip()}'  |  web: '{(m.web.text or '').strip()}'"
        )
        draw.text((pad, pad), header, fill=(255, 255, 255))
        draw.text((pad, pad + 24), sub, fill=(200, 200, 200))

        # Paste crops
        y0 = header_h + pad
        x = pad
        canvas.paste(crop_f, (x, y0))
        x += crop_f.width + pad
        canvas.paste(crop_w, (x, y0))

        fname = f"{idx:03d}_{_safe_name(m.figma.name)}.png" if hasattr(m.figma, 'name') else f"{idx:03d}.png"
        canvas.save(os.path.join(out_root, fname))

    return out_root


def extract_elements_worker(args):
    """스레드 안전한 요소 추출 워커"""
    image, target_height, task_name = args
    
    try:
        logging.info(f"🔥 Starting {task_name} elements extraction...")
        start_time = time.time()
        
        # 각 워커마다 새로운 matcher 인스턴스 생성
        local_matcher = ElementExtractor(resize_size=(736, 736))
        extracted_elements = extract_elements(image, 0, target_height, local_matcher)
        
        end_time = time.time()
        logging.info(f"✅ {task_name} elements extraction completed: {end_time - start_time:.2f} seconds")
        logging.info(f"{task_name} elements extracted: {len(extracted_elements)}")
        
        return extracted_elements
        
    except Exception as e:
        logging.error(f"❌ Error in {task_name} elements extraction: {e}")
        raise e

# 🔧 멀티프로세싱용 전역 변수
_global_matcher = None

def dummy_worker(x):
    """오버헤드 측정용 더미 워커 함수"""
    return x * 2

def init_extraction_worker():
    """멀티프로세싱 워커 초기화 - 각 프로세스마다 YOLO 모델 로드 (CPU 모드)"""
    global _global_matcher
    try:
        import torch
        import os
        
        # GPU 사용 비활성화 - 이것이 핵심!
        torch.cuda.set_device(-1)  # CPU 모드 강제
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU 숨기기
        
        logging.info("🔧 Initializing YOLO model in worker process (CPU mode)...")
        _global_matcher = ElementExtractor(resize_size=(736, 736))
        
        # YOLO 모델을 CPU 모드로 강제 설정
        _global_matcher.yolo.model.to('cpu')
        
        logging.info("✅ YOLO model initialized in worker process (CPU mode)")
    except Exception as e:
        logging.error(f"❌ Error initializing worker: {e}")
        raise e


def extract_elements_mp_worker(args):
    """멀티프로세싱용 요소 추출 워커"""
    image_data, target_height, task_name = args
    global _global_matcher
    
    try:
        logging.info(f"🔥 Starting {task_name} elements extraction in process...")
        start_time = time.time()
        
        # 이미지 데이터를 PIL Image로 변환
        if isinstance(image_data, bytes):
            import io
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        
        # 전역 matcher 사용
        extracted_elements = extract_elements(image, 0, target_height, _global_matcher)
        
        end_time = time.time()
        logging.info(f"✅ {task_name} elements extraction completed: {end_time - start_time:.2f} seconds")
        logging.info(f"{task_name} elements extracted: {len(extracted_elements)}")
        
        return extracted_elements
        
    except Exception as e:
        logging.error(f"❌ Error in {task_name} elements extraction: {e}")
        raise e


def extract_elements_multiprocessing(figma_image: Image.Image, web_image: Image.Image, target_height: int = 720) -> Tuple[List[ExtractedElement], List[ExtractedElement]]:
    """
    멀티프로세싱을 사용한 요소 추출
    Figma 요소 추출 + Web 요소 추출을 동시에 처리
    """
    logging.info("🚀 Starting multiprocessing elements extraction...")
    
    # 이미지를 bytes로 변환 (프로세스 간 전달용)
    import io
    
    # Figma 이미지를 bytes로 변환
    figma_buffer = io.BytesIO()
    figma_image.save(figma_buffer, format='PNG')
    figma_data = figma_buffer.getvalue()
    
    # Web 이미지를 bytes로 변환
    web_buffer = io.BytesIO()
    web_image.save(web_buffer, format='PNG')
    web_data = web_buffer.getvalue()
    
    # 멀티프로세싱으로 요소 추출
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=2, initializer=init_extraction_worker) as executor:
        # Figma와 Web 요소 추출을 동시에
        figma_future = executor.submit(extract_elements_mp_worker, (figma_data, target_height, "Figma"))
        web_future = executor.submit(extract_elements_mp_worker, (web_data, target_height, "Web"))
        
        # 결과 수집
        figma_extracted = figma_future.result()
        web_extracted = web_future.result()
    
    end_time = time.time()
    logging.info(f"🎯 Multiprocessing elements extraction completed: {end_time - start_time:.2f} seconds")
    
    return figma_extracted, web_extracted


def extract_elements_parallel(figma_image: Image.Image, web_image: Image.Image, matcher: ElementExtractor):
    """기존 ThreadPoolExecutor 버전 (백업용)"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        figma_future = executor.submit(extract_elements_worker, (figma_image, 720, matcher, "Figma"))
        web_future = executor.submit(extract_elements_worker, (web_image, 720, matcher, "Web"))
        figma_extracted = figma_future.result()
        web_extracted = web_future.result()
    return figma_extracted, web_extracted



def analyze_multiprocessing_overhead(figma_image: Image.Image, web_image: Image.Image, matcher: ElementExtractor) -> Dict[str, float]:
    """
    멀티프로세싱 오버헤드 세부 분석
    """
    import io
    target_height = 720
    overhead_analysis = {}
    
    logging.info("🔬 Analyzing multiprocessing overhead...")
    
    # 1. 이미지 직렬화 시간 측정
    logging.info("📦 Measuring image serialization overhead...")
    start_time = time.time()
    
    figma_buffer = io.BytesIO()
    figma_image.save(figma_buffer, format='PNG')
    figma_data = figma_buffer.getvalue()
    
    web_buffer = io.BytesIO()
    web_image.save(web_buffer, format='PNG')
    web_data = web_buffer.getvalue()
    
    serialization_time = time.time() - start_time
    overhead_analysis['serialization'] = serialization_time
    logging.info(f"  Serialization time: {serialization_time:.2f}s")
    
    # 2. 이미지 역직렬화 시간 측정
    logging.info("📥 Measuring image deserialization overhead...")
    start_time = time.time()
    
    figma_restored = Image.open(io.BytesIO(figma_data))
    web_restored = Image.open(io.BytesIO(web_data))
    
    deserialization_time = time.time() - start_time
    overhead_analysis['deserialization'] = deserialization_time
    logging.info(f"  Deserialization time: {deserialization_time:.2f}s")
    
    # 3. YOLO 모델 로딩 시간 측정
    logging.info("🔧 Measuring YOLO model loading overhead...")
    start_time = time.time()
    
    new_matcher = ElementExtractor(resize_size=(736, 736))
    
    model_loading_time = time.time() - start_time
    overhead_analysis['model_loading'] = model_loading_time
    logging.info(f"  Model loading time: {model_loading_time:.2f}s")
    
    # 4. 프로세스 생성 오버헤드 추정
    logging.info("⚡ Measuring process creation overhead...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(dummy_worker, 1)
        future2 = executor.submit(dummy_worker, 2)
        future1.result()
        future2.result()
    
    process_creation_time = time.time() - start_time
    overhead_analysis['process_creation'] = process_creation_time
    logging.info(f"  Process creation time: {process_creation_time:.2f}s")
    
    # 5. 총 오버헤드 계산
    total_overhead = (serialization_time + deserialization_time + 
                     model_loading_time * 2 + process_creation_time)  # 모델 로딩은 2개 프로세스
    overhead_analysis['total_estimated'] = total_overhead
    
    logging.info(f"📊 Overhead Analysis Summary:")
    logging.info(f"  Serialization:     {serialization_time:.2f}s")
    logging.info(f"  Deserialization:   {deserialization_time:.2f}s") 
    logging.info(f"  Model Loading x2:  {model_loading_time * 2:.2f}s")
    logging.info(f"  Process Creation:  {process_creation_time:.2f}s")
    logging.info(f"  Total Estimated:   {total_overhead:.2f}s")
    
    return overhead_analysis


def test_overhead_optimization(figma_image: Image.Image, web_image: Image.Image, matcher: ElementExtractor) -> Dict[str, float]:
    """
    오버헤드 최적화 테스트
    """
    results = {}
    target_height = 720
    
    # 1. 기본 멀티프로세싱
    logging.info("🔬 Testing baseline multiprocessing...")
    start_time = time.time()
    figma_mp, web_mp = extract_elements_multiprocessing(figma_image, web_image, target_height)
    baseline_time = time.time() - start_time
    results['baseline_mp'] = baseline_time
    
    # 2. 오버헤드 분석
    overhead_analysis = analyze_multiprocessing_overhead(figma_image, web_image, matcher)
    results.update(overhead_analysis)
    
    # 3. 실제 vs 예상 오버헤드 비교
    sequential_time_estimated = baseline_time - overhead_analysis['total_estimated']
    results['estimated_pure_work'] = sequential_time_estimated
    
    logging.info(f"🎯 Overhead Optimization Analysis:")
    logging.info(f"  Baseline MP time:    {baseline_time:.2f}s")
    logging.info(f"  Estimated overhead:  {overhead_analysis['total_estimated']:.2f}s")
    logging.info(f"  Estimated pure work: {sequential_time_estimated:.2f}s")
    logging.info(f"  Overhead percentage: {(overhead_analysis['total_estimated'] / baseline_time * 100):.1f}%")
    
    return results

def get_frame_by_name_from_raw(figma_raw: List[Dict], name: str) -> Optional[Dict]:
    for data in figma_raw:
        if data['data']['name'] == name:
            return data
    return None

# overlay 비교: 성공, 실패(뜸? 같음?)
def mapping(base_url: str, current_page: str, json_url: str, test_performance: bool = False):
    """메인 실행 함수"""
    logging.info(f"Starting mapping process for base_url: {base_url} and json_url: {json_url}")
    target_height = 1024
    seed_everything(42)
    web_navigator = WebNavigator(headless="new", base_url=base_url)
    visualizer = Visualizer()

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        # 1. 이미지 처리
        logging.info("Step 1: Image Processing")
        # OCR 디버그 모드 확인 (환경 변수로 제어)
        debug_ocr = os.environ.get('DEBUG_OCR', 'false').lower() in ('true', '1', 'yes')
        matcher = ElementExtractor(resize_size=(1024, 1024), debug_ocr=debug_ocr)

        if debug_ocr:
            logging.info("🔍 OCR debug mode enabled - preprocessing steps will be visualized")

        # 유사도 매칭을 위한 SimilarityMatcher 생성 (개선된 유사도 계산 사용)
        from .matcher import create_similarity_matcher
        similarity_matcher = create_similarity_matcher()

        figma_raw, figma_interactions = load_figma_data(json_url)

        root_frame = get_frame_by_name_from_raw(figma_raw, current_page)
        root_image = get_img_by_id(root_frame['data']['id'], figma_raw)
        figma_tree = convert_raw_to_tree(root_frame, root_image, matcher)


        logging.info("Start Web page capturing...")
        web_img = web_navigator.capture_full_page_with_scroll(root_image, target_height)
        logging.info("finished web page capturing\n")

        logging.info("Start Figma elements extracting...")
        start_time = time.time()
        # 🚀 안전한 멀티프로세싱을 사용한 병렬 요소 추출 (GPU 락 방지)
        logging.info("🔥 Using SAFE multiprocessing for parallel extraction (CPU mode)...")
        figma_extracted, web_extracted = extract_elements_multiprocessing_safe(root_image, web_img, target_height)
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time} seconds for figma elements extracting\n")
        logging.info(f"Figma elements extracted: {len(figma_extracted)}")
        logging.info(f"Web elements extracted: {len(web_extracted)}")

        visualizer.visualize_boxes(root_image, [f.box for f in figma_extracted], "Figma elements extracted", show=False, save_only=True, save_path="figma_elements_extracted.png")

        fare_figma = fare_figma_extracted(figma_tree, figma_extracted, figma_interactions)
        visualizer.visualize_boxes(root_image, [f.extracted.box for f in fare_figma], "Figma elements extracted", show=False, save_only=True, save_path="figma_elements_extracted_with_interaction.png")
        logging.info("finished figma elements extracting\n")

        logging.info(f"Web elements extracted: {len(web_extracted)}")




        logging.info(f'Start OCR for figma elements')
        start_time = time.time()
        for f in fare_figma:
            f.extracted.text = extract_texts(root_image, matcher, [f.extracted.box])[0]
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time} seconds for OCR")
        figma_extracted_boxes = [f.extracted.box for f in fare_figma]
        logging.info(f'Start OCR for web elements')
        start_time = time.time()
        for e in web_extracted:
            e.text = extract_texts(web_img, matcher, [e.box])[0]
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time} seconds for OCR")
        web_extracted_boxes = [e.box for e in web_extracted]


        logging.info("Step 3: Separating Figma elements based on interactions")
        interaction_source_ids = {interaction['interactionType']['sourceId'] for interaction in figma_interactions}
        fare_figma_interaction = [f for f in fare_figma if f.id in interaction_source_ids]
        fare_figma_no_interaction = [f for f in fare_figma if f.id not in interaction_source_ids]
        logging.info(f"Found {len(fare_figma_interaction)} elements with interactions and {len(fare_figma_no_interaction)} without.")

        logging.info("Step 4a: Matching elements WITH interactions (using SimilarityMatcher)")
        # 개선된 SimilarityMatcher 사용
        matches_interaction, unmatched_figma_interaction, unmatched_web_interaction = \
            similarity_matcher.find_matches(fare_figma_interaction, web_extracted, root_image, web_img)
        logging.info(f"Found {len(matches_interaction)} matches for interaction elements.")

        # Get the web elements that have been matched
        matched_web_elements_ids = {id(match.web) for match in matches_interaction}
        web_extracted_remaining = [web_el for web_el in web_extracted if id(web_el) not in matched_web_elements_ids]
        
        logging.info(f"{len(web_extracted_remaining)} web elements remaining for second match.")
        logging.info("Step 4b: Matching elements WITHOUT interactions (using SimilarityMatcher)")
        if fare_figma_no_interaction and web_extracted_remaining:
            # 개선된 SimilarityMatcher 사용
            matches_no_interaction, unmatched_figma_no_interaction, unmatched_web_no_interaction = \
                similarity_matcher.find_matches(fare_figma_no_interaction, web_extracted_remaining, root_image, web_img)
            logging.info(f"Found {len(matches_no_interaction)} matches for non-interaction elements.")
        else:
            matches_no_interaction = []
            unmatched_figma_no_interaction = fare_figma_no_interaction
            unmatched_web_no_interaction = web_extracted_remaining
            logging.info("No non-interaction elements or remaining web elements to match.")

        # 교집합 남기도록 필터링
        unmatched_figma = []    
        for figma_el in unmatched_figma_interaction:
            if figma_el in unmatched_figma_no_interaction:
                unmatched_figma.append(figma_el)
        for figma_el in unmatched_figma_no_interaction:
            if figma_el not in unmatched_figma_interaction:
                unmatched_figma.append(figma_el)
        unmatched_web = []
        for web_el in unmatched_web_interaction:
            if web_el in unmatched_web_no_interaction:
                unmatched_web.append(web_el)
        for web_el in unmatched_web_no_interaction:
            if web_el not in unmatched_web_interaction:
                unmatched_web.append(web_el)


        # Combine matches
        matches = matches_interaction + matches_no_interaction
        visualizer.visualize_matches(root_image, web_img, matches, "Matching Visualization", show=False, save_only=True, save_path="matching_visualization.png")
        crops_dir = _save_match_crops(root_image, web_img, matches)

        # Save debug crops for all matches (conditional based on debug flag)


        # visualizer.visualize_boxes(root_image, [m.figma.extracted.box for m in unmatched_figma], "Unmatched Figma elements")
        # visualizer.visualize_boxes(web_img, [m.web.box for m in unmatched_web], "Unmatched Web elements")
        # return []
        logging.info(f"Found {len(matches)} total matches.")

        logging.info("Step 5: Processing Matches")
        return_matches = []
        # 리스트를 extend로 합쳐서 중첩된 리스트 구조를 평평하게 만듦
        # return_matches.extend(match_interaction(matcher, matches_interaction, web_navigator, web_img, figma_interactions, figma_tree, figma_raw))
        # return_matches.extend(process_matches(matcher, matches_no_interaction, web_navigator, figma_interactions, figma_tree, figma_raw))

        return return_matches

    finally:
        if web_navigator.driver is not None:
            logging.info("Closing WebDriver.")
            web_navigator.quit()

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.dump_stats('profile_results.prof')
        logging.info("Profiling results saved to profile_results.prof")

        # 유사도 분포 시각화
        try:
            logging.info("Generating similarity distribution visualization...")
            import subprocess
            current_dir = os.path.dirname(__file__)
            visualize_script = os.path.join(current_dir, 'visualize_topk.py')
            subprocess.run(['python', visualize_script], check=False, capture_output=True)
            logging.info("Similarity distribution visualization completed.")
        except Exception as e:
            logging.warning(f"Failed to generate similarity distribution: {e}")


def test_multiprocessing_extraction(base_url: str, json_url: str):
    """
    멀티프로세싱 요소 추출 테스트 전용 함수
    """
    logging.info("🧪 Starting multiprocessing extraction test...")
    return mapping(base_url, json_url, test_performance=True)


def quick_mp_vs_threading_comparison(base_url: str, json_url: str):
    """
    빠른 멀티프로세싱 vs 스레딩 비교
    """
    logging.info("⚡ Quick multiprocessing vs threading comparison...")
    
    # 멀티프로세싱 처리
    logging.info("Testing multiprocessing extraction...")
    start_time = time.time()
    mapping(base_url, json_url, test_performance=False)
    mp_time = time.time() - start_time
    
    logging.info(f"🏁 Multiprocessing completed in: {mp_time:.2f}s")
    
    return {
        'multiprocessing': mp_time,
        'status': 'completed'
    }

# 기존 mapping 함수에 오버헤드 분석 옵션 추가

def extract_elements_multiprocessing_safe(figma_image: Image.Image, web_image: Image.Image, target_height: int = 720) -> Tuple[List[ExtractedElement], List[ExtractedElement]]:
    """
    멀티프로세싱을 사용한 요소 추출 (GPU 락 경합 해결된 안전 버전)
    """
    logging.info("🚀 Starting SAFE multiprocessing elements extraction (CPU mode)...")
    
    # 이미지를 bytes로 변환 (프로세스 간 전달용)
    import io
    
    # Figma 이미지를 bytes로 변환
    figma_buffer = io.BytesIO()
    figma_image.save(figma_buffer, format='PNG')
    figma_data = figma_buffer.getvalue()
    
    # Web 이미지를 bytes로 변환
    web_buffer = io.BytesIO()
    web_image.save(web_buffer, format='PNG')
    web_data = web_buffer.getvalue()
    
    # 멀티프로세싱으로 요소 추출 (CPU 모드)
    start_time = time.time()
    
    # multiprocessing start method를 'spawn'으로 설정 (더 안전)
    ctx = multiprocessing.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=2, initializer=init_extraction_worker, mp_context=ctx) as executor:
        # Figma와 Web 요소 추출을 동시에
        figma_future = executor.submit(extract_elements_mp_worker, (figma_data, target_height, "Figma"))
        web_future = executor.submit(extract_elements_mp_worker, (web_data, target_height, "Web"))
        
        # 결과 수집
        figma_extracted = figma_future.result()
        web_extracted = web_future.result()
    
    end_time = time.time()
    logging.info(f"🎯 SAFE Multiprocessing elements extraction completed: {end_time - start_time:.2f} seconds")
    
    return figma_extracted, web_extracted
