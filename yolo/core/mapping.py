import os
import logging
from PIL import Image
from .element_matcher import ElementExtractor as LegacyElementExtractor
from .pipeline import create_pipeline, UIMatchingPipeline, PipelineConfig
from .models import ExtractedElement, FigmaFare, MatchResult
from ..visualization.visualizer import Visualizer
from ..web.web_navigator import WebNavigator
from ..utils.utils import load_figma_json, decode_base64_image, get_min_x
import numpy as np
import time
import random
import torch
import requests
import cv2
from routes.dto.response import RoutingMappingInfo, InteractionMappingInfo, GeneralMappingInfo, BaseMappingInfo
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from ..utils.tree_loader import TreeManager, TreeNode
from contextlib import contextmanager
import concurrent.futures
import cProfile
import pstats
import matplotlib.pyplot as plt
from PIL import ImageChops
import io
from torchvision import transforms
from ..utils.error_list import *
import ray
# 로깅 설정	
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import non_max_suppression from element_matcher

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

def extract_texts(img: Image.Image, matcher: LegacyElementExtractor, boxes: List[Tuple[int, int, int, int]]) -> List[str]:
	"""기존 순차 처리 방식으로 텍스트 추출"""
	texts = []
	for box in boxes:
		margin_box = (box[0], box[1], box[2], box[3])
		text = matcher.extract_text(img, margin_box)
		texts.append(text)
	return texts

def extract_text_worker(args):
	"""OCR 병렬 처리를 위한 워커 함수"""
	img_bytes, box, text_margin = args
	
	try:
		# 이미지 복원
		import io
		from PIL import Image
		import cv2
		import numpy as np
		import tesserocr
		
		img = Image.open(io.BytesIO(img_bytes))
		
		# tesserocr API 초기화 (각 워커마다 독립적)
		tessdata_dir = "/usr/local/share/tessdata"
		api = tesserocr.PyTessBaseAPI(path=tessdata_dir, lang='kor+eng')
		api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
		api.SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가-힣 ")
		
		# 박스에 마진 적용
		x1, y1, x2, y2 = map(int, box)
		margin_box = (x1 - text_margin, y1 - text_margin, x2 + text_margin, y2 + text_margin)
		
		# 이미지 크롭 및 전처리
		crop_img = img.crop(margin_box)
		img_np = np.array(crop_img)
		gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
		
		# 노이즈 제거
		denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
		
		# 대비 향상
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		enhanced = clahe.apply(denoised)
		
		# 이진화
		_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		
		# OCR 실행
		binary_pil = Image.fromarray(binary)
		api.SetImage(binary_pil)
		text = api.GetUTF8Text()
		
		# 텍스트 정제
		text = ' '.join(text.split())
		text = ''.join(c for c in text if c.isalnum() or c.isspace() or '\uAC00' <= c <= '\uD7A3')
		
		# API 정리
		api.End()
		
		return text.strip()
		
	except Exception as e:
		print(f"Warning: OCR worker failed: {e}")
		return ""


def get_start_x(tree: TreeNode) -> int:
	manager = TreeManager.from_tree_node(tree)
	return int(manager.get_min_render_x())

def extract_elements(img: Image.Image, start_x: int, windowing_height: int, matcher: LegacyElementExtractor, iou_threshold: float = 0.5, 
					speed_mode: str = "balanced") -> List[ExtractedElement]:
	"""이미지에서 요소 추출 (최적화된 슬라이딩 윈도우 버전)
	
	speed_mode 옵션:
	- "fast": 최대 속도 우선 (큰 윈도우, 적은 오버랩, 적극적인 빈 윈도우 스킵)
	- "balanced": 속도와 정확도 균형 (적당한 최적화)
	- "accurate": 정확도 우선 (기존 설정 유지)
	"""
	logging.info(f"Extracting elements from image with height {img.height} using {speed_mode} mode...")
	
	# speed_mode에 따른 최적화 설정
	if speed_mode == "fast":
		# 최대 속도 모드
		if img.height > 2000:
			optimized_window_height = int(windowing_height * 2.0)  # 100% 증가
			overlap_ratio = 0.9   # 10% 오버랩
		else:
			optimized_window_height = int(windowing_height * 1.5)  # 50% 증가
			overlap_ratio = 0.85  # 15% 오버랩
		empty_threshold = 0.9  # 더 적극적인 빈 윈도우 스킵
		# 순차 처리이므로 max_workers는 사용하지 않음
		
	elif speed_mode == "balanced":
		# 균형 모드 (기존 최적화)
		if img.height > 3000:
			optimized_window_height = int(windowing_height * 1.5)  # 50% 증가
			overlap_ratio = 0.85  # 15% 오버랩
		elif img.height > 2000:
			optimized_window_height = int(windowing_height * 1.2)  # 20% 증가
			overlap_ratio = 0.8   # 20% 오버랩
		else:
			optimized_window_height = windowing_height
			overlap_ratio = 0.75  # 기본값 유지
		empty_threshold = 0.95
		# 순차 처리이므로 max_workers는 사용하지 않음
		
	else:  # "accurate"
		# 정확도 우선 모드
		optimized_window_height = windowing_height
		overlap_ratio = 0.75  # 기존 오버랩 유지
		empty_threshold = 0.98  # 매우 보수적인 빈 윈도우 스킵
		# 순차 처리이므로 max_workers는 사용하지 않음
	
	def is_window_empty(crop_img, threshold=empty_threshold):
		"""윈도우가 대부분 비어있는지 확인 (흰색 배경 등)"""
		# 이미지를 grayscale로 변환
		gray = crop_img.convert('L')
		# 픽셀 값의 분산을 계산 (분산이 낮으면 단색에 가까움)
		pixels = np.array(gray)
		# 흰색 픽셀 비율 계산 (240 이상을 흰색으로 간주)
		white_ratio = np.sum(pixels > 240) / pixels.size
		return white_ratio > threshold
	
	windows_to_process = []
	current_height = 0
	start_x = 0
	skipped_windows = 0
	
	while current_height < img.height:
		crop_img = img.crop((start_x, current_height, img.width, min(current_height + optimized_window_height, img.height)))
		
		# 빈 윈도우 체크 (작은 윈도우는 스킵하지 않음)
		if crop_img.height > 100 and is_window_empty(crop_img):
			skipped_windows += 1
			current_height += optimized_window_height * overlap_ratio
			continue
			
		windows_to_process.append((crop_img, current_height))
		current_height += optimized_window_height * overlap_ratio
	
	if skipped_windows > 0:
		logging.info(f"Skipped {skipped_windows} empty windows for optimization")

	all_boxes = []
	all_scores = []
	all_cls = []
	all_features = []

	# 각 창에 대해 순차적으로 탐지 및 특징 추출 수행
	def process_window(crop_img, original_height):
		# 1. 탐지 및 특징 추출을 한 번에 수행
		boxes, scores, cls, feat_map, original_img_size = matcher.detect_boxes_yolo(crop_img, extract_features=True)
		if len(boxes) == 0:
			return None, None, None, None
		# 2. 특징 추출
		features = matcher.extract_features_from_map(feat_map, boxes, original_img_size)
		
		# 3. Y 좌표 조정
		boxes[:, 1] += original_height
		boxes[:, 3] += original_height
		
		return boxes, scores, cls, features
	
	logging.info(f"Processing {len(windows_to_process)} windows sequentially (optimized for multiprocessing environment)...")
	start_time = time.time()
	
	# 순차 처리 (이미 멀티프로세싱 환경이므로 추가 병렬화는 비효율적)
	for crop_img, h in windows_to_process:
		boxes, scores, cls, features = process_window(crop_img, h)
		if boxes is not None:
			all_boxes.append(boxes)
			all_scores.append(scores)
			all_cls.append(cls)
			all_features.append(features)
	
	end_time = time.time()
	logging.info(f"Time taken: {end_time - start_time} seconds for figma extraction (sequential in MP environment)")

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


def check_interaction_overlay(match: MatchResult, web_navigator: WebNavigator, figma_raw: List[Dict], web_img: Image.Image, interaction: Dict) -> bool:
	return_matches = []
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

		overlay_figma_img = get_img_by_id(interaction['interactionType']['destinationId'], figma_raw)

		# 스크린샷을 이미지 변수로 저장
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

				# Compare the resized images
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
				margin = (0.2 + 0.5) / 2 # this is an average of the margin_pos and margin_neg hyperparameters
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
			web_navigator.driver.close()
			web_navigator.driver.switch_to.window(web_navigator.driver.window_handles[0])

	return return_matches

def check_interaction_navigate(match: MatchResult, web_navigator: WebNavigator, figma_raw: List[Dict], web_img: Image.Image, interaction: Dict) -> bool:
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

def match_interaction(matcher: LegacyElementExtractor, matches: List[MatchResult], web_navigator: WebNavigator, web_img: Image.Image, interactions: List[Dict], figma_tree: TreeNode, figma_raw: List[Dict]) -> List[BaseMappingInfo]:
	return_matches = []
	for match in matches:
		interaction = get_interaction_by_id(match.figma.id, interactions)
		print("##", interaction['interactionType']['navigation'], "##")
		if interaction['interactionType']['navigation'] == 'NAVIGATE':
			return_matches.extend(check_interaction_navigate(match, web_navigator, figma_raw, web_img, interaction))
		elif interaction['interactionType']['navigation'] == 'OVERLAY':
			return_matches.extend(check_interaction_overlay(match, web_navigator, figma_raw, web_img, interaction))

	return return_matches
def process_matches(matcher: LegacyElementExtractor, matches: List[MatchResult], web_navigator: WebNavigator, interactions: List[Dict], figma_tree: TreeNode, figma_raw: List[Dict]) -> List[BaseMappingInfo]:
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

def convert_raw_to_tree(figma_tree: Dict, root_image: Image.Image, level: int = 0) -> TreeNode:
	"""
	Deprecated: TreeManager.from_figma_tree()를 사용하십시오.

	현재 구현은 TreeManager를 통해 트리를 구성합니다.
	"""
	manager = TreeManager.from_figma_tree(figma_tree)
	return manager.root
	
def get_frame_by_name(figma_tree: Dict, name: str) -> Optional[TreeNode]:
	if figma_tree['data']['name'] == name:
		return figma_tree
	for child in figma_tree['children']:
		result = get_frame_by_name(child, name)
		if result is not None:
			return result
	return None


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
				iou = LegacyElementExtractor.calculate_iou(node_rect, extracted.box)
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


def extract_elements_worker(args):
	"""스레드 안전한 요소 추출 워커"""
	image, target_height, task_name = args
	
	try:
		logging.info(f"🔥 Starting {task_name} elements extraction...")
		start_time = time.time()
		
		# 각 워커마다 새로운 matcher 인스턴스 생성
		local_matcher = LegacyElementExtractor(resize_size=(736, 736))
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

@ray.remote
class LegacyElementExtractorActor:
	"""Ray Actor for element extraction with YOLO model"""
	
	def __init__(self):
		try:
			import torch
			import os
			
			# GPU 사용 비활성화
			torch.cuda.set_device(-1) if torch.cuda.is_available() else None
			os.environ['CUDA_VISIBLE_DEVICES'] = ''
			
			logging.info("🔧 Initializing YOLO model in Ray actor (CPU mode)...")
			self.matcher = LegacyElementExtractor(resize_size=(736, 736))
			
			# YOLO 모델을 CPU 모드로 강제 설정
			self.matcher.yolo.model.to('cpu')
			
			logging.info("✅ YOLO model initialized in Ray actor (CPU mode)")
		except Exception as e:
			logging.error(f"❌ Error initializing Ray actor: {e}")
			raise e
	
	def extract_elements_with_ocr(self, image_data, target_height, task_name, start_x=0, include_ocr=True):
		"""Ray 원격 함수로 요소 추출 및 OCR"""
		try:
			logging.info(f"🔥 Starting {task_name} elements extraction in Ray actor...")
			start_time = time.time()
			
			# 이미지 데이터를 PIL Image로 변환
			if isinstance(image_data, bytes):
				import io
				image = Image.open(io.BytesIO(image_data))
			else:
				image = image_data
			
			# 요소 추출
			extracted_elements = extract_elements(image, start_x, target_height, self.matcher)
			
			extraction_time = time.time()
			logging.info(f"✅ {task_name} elements extraction completed: {extraction_time - start_time:.2f} seconds")
			logging.info(f"{task_name} elements extracted: {len(extracted_elements)}")
			
			# OCR 처리 (요청된 경우)
			if include_ocr and extracted_elements:
				logging.info(f"🔤 Starting OCR for {task_name} elements in Ray actor...")
				ocr_start = time.time()
				
				# 각 요소에 대해 OCR 수행
				for element in extracted_elements:
					text_margin = 10
					x1, y1, x2, y2 = map(int, element.box)
					margin_box = (x1 - text_margin, y1 - text_margin, x2 + text_margin, y2 + text_margin)
					element.text = self.matcher.extract_text(image, margin_box)
				
				ocr_time = time.time()
				logging.info(f"✅ {task_name} OCR completed: {ocr_time - ocr_start:.2f} seconds")
			
			total_time = time.time()
			logging.info(f"🎯 {task_name} total processing time: {total_time - start_time:.2f} seconds")
			
			return extracted_elements
			
		except Exception as e:
			logging.error(f"❌ Error in {task_name} elements extraction/OCR: {e}")
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


def get_frame_by_name_from_raw(figma_raw: List[Dict], name: str) -> Optional[Dict]:
    for data in figma_raw:
        if data['data']['name'] == name:
            return data
    return None


def _save_match_crops(figma_img: Image.Image, web_img: Image.Image, matches: List[MatchResult], out_root: Optional[str] = None) -> str:
    """Save side-by-side crops for each match with score annotations."""
    import os
    import time
    from PIL import Image, ImageDraw, ImageFont

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

# 새로운 파이프라인 기반 매핑 함수
def mapping_v2(base_url: str, current_page: str, json_url: str, **kwargs):
	"""
	새로운 클래스 기반 파이프라인을 사용한 매핑 함수
	- 의존성 주입 패턴 적용
	- 설정 기반 파라미터 조정
	- 개선된 에러 처리
	"""
	logging.info(f"✨ Starting new pipeline mapping for base_url: {base_url} and json_url: {json_url}")

	try:
		from ..core.mapping_processor import MappingProcessor

		# 파이프라인 생성
		pipeline = create_pipeline()

		# 매칭 실행
		logging.info("🚀 Executing pipeline.match()...")
		matches = pipeline.match(
			figma_url=json_url,
			web_url=base_url,
			page_name=current_page
		)

		# 매핑 정보 생성
		logging.info("📋 Generating mapping info...")
		processor = MappingProcessor()
		mapping_infos = processor.get_mapping_info(matches)

		logging.info(f"✅ New pipeline completed: {len(mapping_infos)} mapping infos generated")
		return mapping_infos

	except Exception as e:
		logging.error(f"❌ New pipeline mapping failed: {e}")
		import traceback
		traceback.print_exc()
		# fallback to legacy mapping
		logging.info("⚠️  Falling back to legacy mapping...")
		return mapping_legacy(base_url, current_page, json_url, **kwargs)


def mapping(base_url: str, current_page: str, json_url: str, test_performance: bool = False, use_new_pipeline: bool = None):
	"""
	메인 실행 함수 - 새로운 파이프라인과 기존 코드 호환성 제공
	
	Args:
		use_new_pipeline: True면 새 파이프라인, False면 기존 코드, None이면 환경변수로 결정
	"""
	# 파이프라인 선택
	if use_new_pipeline is None:
		use_new_pipeline = str(os.environ.get('USE_NEW_PIPELINE', '0')).lower() in ('1', 'true', 'yes')
	
	if use_new_pipeline:
		logging.info("Using new class-based pipeline")
		return mapping_v2(base_url, current_page, json_url, test_performance=test_performance)
	else:
		logging.info("Using legacy mapping implementation")
		return mapping_legacy(base_url, current_page, json_url, test_performance=test_performance)


def mapping_legacy(base_url: str, current_page: str, json_url: str, test_performance: bool = False):
	"""기존 레거시 매핑 함수 (원본 코드)"""
	logging.info(f"Starting legacy mapping process for base_url: {base_url} and json_url: {json_url}")
	target_height = 720
	seed_everything(42)
	web_navigator = WebNavigator(headless=True, base_url=base_url)
	visualizer = Visualizer()

	# Ray 초기화
	if not ray.is_initialized():
		logging.info("🚀 Initializing Ray...")
		ray.init(
			num_cpus=None,  # 시스템 CPU 수 자동 감지
			ignore_reinit_error=True,
			log_to_driver=False  # Ray 로그를 줄임
		)
		logging.info("✅ Ray initialized successfully")
	
	profiler = cProfile.Profile()
	profiler.enable()

	try:
		# 1. 이미지 처리
		logging.info("Step 1: Image Processing")
		matcher = LegacyElementExtractor(resize_size=(736, 736))
		figma_raw, figma_interactions = load_figma_data(json_url)

		root_frame = get_frame_by_name_from_raw(figma_raw, current_page)
		root_image = get_img_by_id(root_frame['data']['id'], figma_raw)
		figma_tree = convert_raw_to_tree(root_frame, root_image)
		start_x = get_start_x(figma_tree)
		
		logging.info("Start Web page capturing...")
		web_img = web_navigator.capture_full_page_with_scroll(root_image, target_height)
		logging.info("finished web page capturing\n")

		logging.info("Start Figma elements extracting...")
		start_time = time.time()
		# 🚀 Ray를 사용한 분산 병렬 요소 추출
		logging.info("🔥 Using Ray for distributed parallel extraction...")

		figma_extracted, web_extracted = extract_elements_ray(root_image, web_img, target_height, start_x, include_ocr=True)
		end_time = time.time()
		logging.info(f"Time taken: {end_time - start_time} seconds for elements extraction + OCR\n")
		logging.info(f"Figma elements extracted: {len(figma_extracted)}")
		logging.info(f"Web elements extracted: {len(web_extracted)}")

		visualizer.visualize_boxes(root_image, [f.box for f in figma_extracted], "Figma elements extracted")
		visualizer.visualize_boxes(web_img, [e.box for e in web_extracted], "Web elements extracted")

		logging.info(f"Figma elements extracted: {len(figma_extracted)}")        
		fare_figma = fare_figma_extracted(figma_tree, figma_extracted, figma_interactions)
		# visualizer.visualize_boxes(root_image, [f.extracted.box for f in fare_figma], "fare")
		# return []
		logging.info("finished figma elements extracting\n")

		# OCR은 이미 multiprocessing 안에서 완료됨
		logging.info("✅ OCR already completed during multiprocessing extraction")
		figma_extracted_boxes = [f.extracted.box for f in fare_figma]
		web_extracted_boxes = [e.box for e in web_extracted]
		logging.info(f"Web elements extracted: {len(web_extracted)}")

		logging.info("Step 3: Separating Figma elements based on interactions")
		interaction_source_ids = {interaction['interactionType']['sourceId'] for interaction in figma_interactions}
		fare_figma_interaction = [f for f in fare_figma if f.id in interaction_source_ids]
		fare_figma_no_interaction = [f for f in fare_figma if f.id not in interaction_source_ids]
		logging.info(f"Found {len(fare_figma_interaction)} elements with interactions and {len(fare_figma_no_interaction)} without.")


		logging.info("Step 4a: Matching elements WITH interactions")
		sim_dict_interaction = matcher.calculate_similarity(root_image, web_img, fare_figma_interaction, web_extracted)
		matches_interaction, unmatched_figma_interaction, unmatched_web_interaction = \
			matcher.get_matches(sim_dict_interaction, fare_figma_interaction, web_extracted, 0.65)
		logging.info(f"Found {len(matches_interaction)} matches for interaction elements.")



		# visualizer.visualize_boxes(root_image, [m.figma.extracted.box for m in low_confidence_interaction], "Low confidence matches")
		# Get the web elements that have been matched
		matched_web_elements_ids = {id(match.web) for match in matches_interaction}
		web_extracted_remaining = [web_el for web_el in web_extracted if id(web_el) not in matched_web_elements_ids]
		
		logging.info(f"{len(web_extracted_remaining)} web elements remaining for second match.")
		logging.info("Step 4b: Matching elements WITHOUT interactions")
		if fare_figma_no_interaction and web_extracted_remaining:
			sim_dict_no_interaction = matcher.calculate_similarity(root_image, web_img, fare_figma_no_interaction, web_extracted_remaining)
			matches_no_interaction, unmatched_figma_no_interaction, unmatched_web_no_interaction = \
				matcher.get_matches(sim_dict_no_interaction, fare_figma_no_interaction, web_extracted_remaining, 0.65)
			logging.info(f"Found {len(matches_no_interaction)} matches for non-interaction elements.")
		else:
			matches_no_interaction = []
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
		visualizer.visualize_matches(root_image, web_img, matches, "Matching Visualization")

		# Save per-match crops with scores for debugging
		try:
			if getattr(matcher, 'debug_similarity', False):
				out_dir = _save_match_crops(root_image, web_img, matches)
				logging.info(f"Saved match crops to: {out_dir}")
		except Exception as e:
			logging.warning(f"Failed to save match crops: {e}")

		# visualizer.visualize_boxes(root_image, [m.figma.extracted.box for m in unmatched_figma], "Unmatched Figma elements")
		# visualizer.visualize_boxes(web_img, [m.web.box for m in unmatched_web], "Unmatched Web elements")
		# return []
		logging.info(f"Found {len(matches)} total matches.")
	
		logging.info("Step 5: Processing Matches")
		return_matches = []
		# 리스트를 extend로 합쳐서 중첩된 리스트 구조를 평평하게 만듦
		logging.info("Step 6: Matching Interactions")
		return_matches.extend(match_interaction(matcher, matches_interaction, web_navigator, web_img, figma_interactions, figma_tree, figma_raw))
		logging.info(f"Found {len(return_matches)} matches for interaction elements.")
		logging.info("Step 7: Processing Matches")
		return_matches.extend(process_matches(matcher, matches, web_navigator, figma_interactions, figma_tree, figma_raw))
		logging.info(f"Found {len(return_matches)} matches for non-interaction elements.")
		return return_matches

	finally:
		if web_navigator.driver is not None:
			logging.info("Closing WebDriver.")
			web_navigator.quit()
		
		# Ray 정리
		if ray.is_initialized():
			logging.info("🔧 Shutting down Ray...")
			ray.shutdown()
			logging.info("✅ Ray shutdown completed")
		
		profiler.disable()
		stats = pstats.Stats(profiler).sort_stats('cumtime')
		stats.dump_stats('profile_results.prof')
		logging.info("Profiling results saved to profile_results.prof")

# 기존 mapping 함수에 오버헤드 분석 옵션 추가

def extract_elements_ray(figma_image: Image.Image, web_image: Image.Image, target_height: int = 540, start_x: int = 0, include_ocr: bool = True) -> Tuple[List[ExtractedElement], List[ExtractedElement]]:
	"""
	Ray를 사용한 요소 추출 및 OCR (분산 처리 버전)
	"""
	if include_ocr:
		logging.info("🚀 Starting Ray distributed elements extraction + OCR...")
	else:
		logging.info("🚀 Starting Ray distributed elements extraction...")
	
	# 이미지를 bytes로 변환 (Ray 간 전달용)
	import io
	
	# Figma 이미지를 bytes로 변환
	figma_buffer = io.BytesIO()
	figma_image.save(figma_buffer, format='PNG')
	figma_data = figma_buffer.getvalue()
	
	# Web 이미지를 bytes로 변환
	web_buffer = io.BytesIO()
	web_image.save(web_buffer, format='PNG')
	web_data = web_buffer.getvalue()
	
	# Ray를 사용한 분산 처리
	start_time = time.time()
	
	# Ray actors 생성 (2개의 워커)
	figma_actor = LegacyElementExtractorActor.remote()
	web_actor = LegacyElementExtractorActor.remote()
	
	try:
		# Figma와 Web 요소 추출을 동시에 실행
		figma_future = figma_actor.extract_elements_with_ocr.remote(figma_data, target_height, "Figma", start_x, include_ocr)
		web_future = web_actor.extract_elements_with_ocr.remote(web_data, target_height, "Web", start_x, include_ocr)
		
		# 결과 수집 (Ray.get으로 결과 대기)
		figma_extracted, web_extracted = ray.get([figma_future, web_future])
		
	finally:
		# Ray actors 정리
		ray.kill(figma_actor)
		ray.kill(web_actor)
	
	end_time = time.time()
	if include_ocr:
		logging.info(f"🎯 Ray distributed elements extraction + OCR completed: {end_time - start_time:.2f} seconds")
	else:
		logging.info(f"🎯 Ray distributed elements extraction completed: {end_time - start_time:.2f} seconds")
	
	return figma_extracted, web_extracted

def extract_elements_multiprocessing_safe(figma_image: Image.Image, web_image: Image.Image, target_height: int = 540, start_x: int = 0, include_ocr: bool = True) -> Tuple[List[ExtractedElement], List[ExtractedElement]]:
	"""
	호환성을 위한 래퍼 함수 - Ray 버전을 호출
	"""
	return extract_elements_ray(figma_image, web_image, target_height, start_x, include_ocr)
