import os
from PIL import Image
from .element_matcher import ElementMatcher, MatchResult
from .visualizer import Visualizer
from .web_navigator import WebNavigator
from .utils import load_figma_json, decode_base64_image, get_min_x, get_figma_match_info, frame_to_dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
import random
import torch
import requests
from routes.dto.response.mapping_response import MappingInfo, MappingResponse
from typing import List, Dict, Tuple, Optional

def seed_everything(seed: int = 42):
    """랜덤 시드 설정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def catergorize_match(match: MatchResult) -> List[str]:
    """매칭된 요소들을 카테고리별로 분류"""
    category = []
    if abs(match.figma_box[0] - match.web_box[0]) > 10:
        category.append('different_x')
    if abs(match.figma_box[1] - match.web_box[1]) > 10:
        category.append('different_y')
    
    w_f = match.figma_box[2] - match.figma_box[0]    
    h_f = match.figma_box[3] - match.figma_box[1]
    w_w = match.web_box[2] - match.web_box[0]
    h_w = match.web_box[3] - match.web_box[1]

    if abs(w_f - w_w) > 10 or abs(h_f - h_w) > 10:
        category.append('different_size')
    
    if len(category) == 0:
        category.append('same')
    match.errorCategories = category
    return category

def get_mapping_info(matches: List[MatchResult]) -> List[MappingInfo]:
    """매칭 결과를 매핑 정보로 변환"""
    mapping_infos = []
    for match in matches:
        mapping_info = MappingInfo(
            componentName=match.figma_name,
            destinationFigmaPage=match.figma_dest if match.figma_dest else "",
            destinationUrl=match.web_dest if match.web_dest else "",
            actualUrl=match.web_dest if match.web_dest else "",
            failReason=", ".join(match.errorCategories) if match.errorCategories != ['same'] else "",
            isSuccess=match.errorCategories == ['same'],
            isRouting=bool(match.figma_dest)
        )
        mapping_infos.append(mapping_info)
    return mapping_infos

def process_images(figma_url: str, web_navigator: WebNavigator, target_height: int) -> Tuple[Image.Image, Image.Image, List[Dict]]:
    """이미지 처리 및 데이터 추출"""
    response = requests.get(figma_url)
    response.raise_for_status()

    frames = response.json()
    root = frames[0]
    print('#########################')
    print(root)
    print('#########################')
    root_img = decode_base64_image(root['image'])
    
    web_img = web_navigator.capture_full_page_with_scroll(root_img, target_height)
    scale = web_img.width / root_img.width
    web_img = web_img.resize((root_img.width, int(web_img.height / scale)))
    
    return root_img, web_img, frames

def extract_elements(img: Image.Image, start_height: int, target_height: int, matcher: ElementMatcher) -> Dict:
    """이미지에서 요소 추출"""
    all_features = []
    all_texts = []
    all_boxes = []
    
    # 이미지 전체 높이를 target_height 단위로 처리
    current_height = 0
    while current_height < img.height:
        # 현재 높이에서 target_height만큼의 영역을 크롭
        crop_img = img.crop((0, current_height, img.width, min(current_height + target_height, img.height)))
        
        # 박스 검출
        temp_boxes = matcher.detect_boxes_yolo(crop_img)
        if len(temp_boxes) > 0:
            # 박스 좌표를 원본 이미지 기준으로 조정
            temp_boxes[:, 1] += current_height
            temp_boxes[:, 3] += current_height
            boxes = np.column_stack([temp_boxes[:, 0], temp_boxes[:, 1], temp_boxes[:, 2], temp_boxes[:, 3]])
            
            # 각 박스에 대해 특징과 텍스트 추출
            for box in boxes:
                feature = matcher.extract_features(crop_img, box)
                text = matcher.extract_text(crop_img, box)
                all_features.append(feature)
                all_texts.append(text)
                all_boxes.append(box)
        
        current_height += target_height
    
    return {
        'feature': all_features,
        'text': all_texts,
        'boxes': np.array(all_boxes) if all_boxes else np.array([])
    }

def calculate_similarity(matcher: ElementMatcher, figma_data: List[Dict], web_data: List[Dict], root_img: Image.Image, web_img: Image.Image) -> Dict[str, np.ndarray]:
    """유사도 계산"""
    text_sim, feature_sim, size_sim, coordinate_sim = \
        matcher.calculate_similarity(root_img, web_img, figma_data, web_data)
        
    return {
        'text': (text_sim - np.min(text_sim)) / (np.max(text_sim) - np.min(text_sim) + 1e-8),
        'feature': (feature_sim - np.min(feature_sim)) / (np.max(feature_sim) - np.min(feature_sim) + 1e-8),
        'size': (size_sim - np.min(size_sim)) / (np.max(size_sim) - np.min(size_sim) + 1e-8),
        'coordinate': (coordinate_sim - np.min(coordinate_sim)) / (np.max(coordinate_sim) - np.min(coordinate_sim) + 1e-8)
    }

def process_matches(matches: List[MatchResult], web_navigator: WebNavigator, frames: List[Dict]) -> List[MatchResult]:
    """매칭 결과 처리"""
    for match in matches:
        match.errorCategories = catergorize_match(match)

    figma_info = frame_to_dict(frames[0])
    matches = get_figma_match_info(figma_info, matches)
    interaction_matches = [match for match in matches if match.interaction == 'NAVIGATE']
    
    for match in interaction_matches:
        center_x = float(match.web_box[0] + match.web_box[2]) / 2
        center_y = float(match.web_box[1] + match.web_box[3]) / 2

        if web_navigator.driver is not None:
            element, xpath = web_navigator.get_element_at_coordinate_and_xpath(center_x, center_y)
            if element is not None and xpath is not None:
                urls = web_navigator.get_url_in_new_tab(xpath)
                match.web_dest = urls
                
    return matches

def mapping(base_url: str, json_url: str):
    """메인 실행 함수"""
    target_height = 720
    seed_everything(42)
    print(base_url)
    web_navigator = WebNavigator(base_url=base_url)
    
    try:
        # 1. 이미지 처리
        root_img, web_img, frames = process_images(json_url, web_navigator, target_height)
        
        # 2. 요소 추출
        matcher = ElementMatcher()
        figma_data = [extract_elements(root_img, 0, target_height, matcher)]
        web_data = [extract_elements(web_img, 0, target_height, matcher)]
        
        # 3. 유사도 계산
        sim_dict = calculate_similarity(matcher, figma_data, web_data, root_img, web_img)
        
        # 4. 매칭 처리
        matches, _ = matcher.get_matches(sim_dict, figma_data[0]['boxes'], web_data[0]['boxes'], 0.8)
        visualizer = Visualizer()
        # 5. 매칭 결과 처리
        matches = process_matches(matches, web_navigator, frames)
        visualizer.visualize_matches(root_img, web_img, matches, "Matching Visualization")

        # 6. 매핑 정보 생성
        mapping_infos = get_mapping_info(matches)
        return mapping_infos
        
    finally:
        if web_navigator.driver is not None:
            web_navigator.quit()