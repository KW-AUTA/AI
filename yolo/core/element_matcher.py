import numpy as np
from PIL import Image
from PIL import ImageFilter
import torch
import torch.nn.functional as F
import torchvision.ops
import torchvision.transforms as T
from torchvision.transforms.functional import pad
from ultralytics import YOLO
import tesserocr
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Any, Dict
import cv2
import os
import re
import unicodedata
from scipy.optimize import linear_sum_assignment
from .models import FigmaFare, ExtractedElement, MatchResult
from ..utils.errorChecker import ErrorChecker
from ..utils.error_list import *
# Letterbox function: resize and pad image to meet new_shape, maintaining aspect ratio.
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
	"""
	Resize and pad image to meet new_shape, maintaining aspect ratio.
	Returns padded image, resize ratio, and padding (left, top).
	"""
	# Convert PIL numpy array (H, W, C)
	import cv2
	shape = im.shape[:2]  # current shape [height, width]
	# Compute resize ratio
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	# Compute unpadded new size
	new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
	# Compute padding
	dw = new_shape[1] - new_unpad[0]
	dh = new_shape[0] - new_unpad[1]
	dw /= 2  # divide padding into 2 sides
	dh /= 2
	# Resize image
	im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
	# Compute border sizes
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	# Add border
	im_padded = cv2.copyMakeBorder(
		im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
	)
	return im_padded, r, (left, top)


# Non-Maximum Suppression for axis-aligned bounding boxes
def non_max_suppression(
	boxes: np.ndarray,
	scores: Optional[np.ndarray] = None,
	iou_threshold: float = 0.5
) -> List[int]:
	"""
	Perform Non-Maximum Suppression on axis-aligned bounding boxes using torchvision.ops.nms.
	boxes: numpy array of shape (N,4) in (x1,y1,x2,y3)
	scores: confidence scores array of shape (N,), defaults to all 1s
	returns: list of indices to keep
	"""
	if boxes.size == 0:
		return []

	# Convert numpy arrays to torch tensors
	boxes_tensor = torch.from_numpy(boxes).float()
	if scores is None:
		scores_tensor = torch.ones(len(boxes), dtype=torch.float32)
	else:
		scores_tensor = torch.from_numpy(scores).float()

	# Perform NMS
	keep_indices_tensor = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)

	# Convert result back to a list of integers
	return keep_indices_tensor.cpu().numpy().tolist()
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules.conv import Conv
from torch.nn import Conv2d

# 모델 import
from .models import (
	MatchResult, ExtractedElement
)

class ElementExtractor:
	"""요소 매칭을 수행하는 클래스"""
	def __init__(self, yolo_model_path: str = None, resize_size: Tuple[int, int] = (736, 736), debug_similarity: bool = False):
		current_dir = os.path.dirname(__file__)
		if yolo_model_path is None:
			yolo_model_path = os.path.join(current_dir, "..", "models_weights", "best.pt")
		self.yolo = YOLO(yolo_model_path, task='detect', verbose=False)
		self.resize_size = resize_size
		self.debug_similarity = debug_similarity or str(os.environ.get('SIM_DEBUG', '1')).lower() in ('1', 'true', 'yes')
		# tesserocr 설정 개선 (한글+영어 지원, 최적화된 설정)
		tessdata_dir = "/usr/local/share/tessdata"
		self.api = tesserocr.PyTessBaseAPI(path=tessdata_dir, lang='kor+eng')
		# OCR 품질 향상을 위한 설정들
		self.api.SetVariable("user_defined_dpi", "300")
		self.api.SetVariable("tessedit_char_blacklist", "")  # 블랙리스트 비우기
		self.api.SetVariable("preserve_interword_spaces", "1")  # 단어 간 공백 보존
		
		self.transform = T.Compose([
			T.Resize(resize_size),
			# T.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
			T.ToTensor(),
			# T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	def _pick_psm(self, w: int, h: int) -> int:
		"""ROI 크기/종횡비에 따라 최적화된 PSM을 선택"""
		if w == 0 or h == 0:
			return tesserocr.PSM.SINGLE_BLOCK
		aspect = w / max(1, h)
		area = w * h
		
		# 매우 작은 영역: 단일 문자나 숫자
		if area < 20 * 20:
			return tesserocr.PSM.SINGLE_CHAR
		# 작은 영역: 단어 단위
		elif area < 50 * 50:
			return tesserocr.PSM.SINGLE_WORD
		# 매우 가로로 긴 형태: 단일 라인 (버튼, 탭 등)
		elif aspect >= 4.0:
			return tesserocr.PSM.SINGLE_LINE
		# 세로로 긴 형태나 중간 크기의 가로 긴 형태: 단일 라인
		elif aspect <= 0.5 or aspect >= 2.0:
			return tesserocr.PSM.SINGLE_LINE
		# 기본: 단일 블록 처리
		else:
			return tesserocr.PSM.SINGLE_BLOCK

	def _preprocess_roi(self, pil_crop: Image.Image, scale: float = None, pad: int = 15) -> Image.Image:
		"""개선된 ROI 전처리로 OCR 인식률 향상"""
		import cv2
		import numpy as np
		
		arr = np.array(pil_crop)
		if arr.ndim == 3:
			gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
		else:
			gray = arr
		
		# 동적 스케일링 (ROI 크기에 따라 조정)
		h, w = gray.shape
		if scale is None:
			if max(h, w) < 30:
				scale = 4.0  # 매우 작은 텍스트는 4배 확대
			elif max(h, w) < 60:
				scale = 3.0  # 작은 텍스트는 3배 확대
			elif max(h, w) < 120:
				scale = 2.0  # 보통 텍스트는 2배 확대
			else:
				scale = 1.5  # 큰 텍스트는 적게 확대
		
		# 스케일링 (Lanczos 보간법으로 품질 향상)
		if scale > 1.0:
			new_w, new_h = int(w * scale), int(h * scale)
			gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
		
		# 대비 향상 (CLAHE 적용)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		enhanced = clahe.apply(gray)
		
		# 노이즈 제거 (약한 필터링)
		denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
		
		# 적응적 임계값 적용 (Otsu보다 더 나은 결과)
		binary = cv2.adaptiveThreshold(
			denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
			cv2.THRESH_BINARY, 11, 2
		)
		
		# 모폴로지 연산으로 텍스트 구조 개선
		kernel = np.ones((2, 2), np.uint8)
		binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
		
		# 여백 추가 (더 넉넉하게)
		binary = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
		
		return Image.fromarray(binary)

	def __del__(self):
		"""tesserocr API 정리"""
		if hasattr(self, 'api'):
			self.api.End()

	def extract_text(self, img: Image.Image, box: np.ndarray) -> str:
		x1, y1, x2, y2 = map(int, box)
		# ROI에 소량 패딩을 주어 자획 절단 방지
		pad = 12
		x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
		x2p, y2p = min(img.width, x2 + pad), min(img.height, y2 + pad)
		crop_img = img.crop((x1p, y1p, x2p, y2p))

		# 향상된 전처리 적용
		binary_pil = self._preprocess_roi(crop_img, pad=20)

		# ROI 크기에 따라 최적화된 PSM 선택
		roi_w, roi_h = x2p - x1p, y2p - y1p
		psm = self._pick_psm(roi_w, roi_h)
		self.api.SetPageSegMode(psm)

		# 텍스트 유형별 최적화된 설정
		try:
			# OCR 엔진 모드 설정 (LSTM이 더 정확)
			self.api.SetVariable("tessedit_ocr_engine_mode", "1")  # LSTM 모드
			
			# 작은 텍스트나 숫자의 경우 화이트리스트 적용
			if psm in (tesserocr.PSM.SINGLE_WORD, tesserocr.PSM.SINGLE_CHAR) or roi_w * roi_h < 2000:
				self.api.SetVariable("tessedit_char_whitelist", 
					"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가-힣 ,.-%₩:()[]#/@&")
			else:
				# 긴 텍스트는 화이트리스트 해제하여 유연성 확보
				self.api.SetVariable("tessedit_char_whitelist", "")
			
			# 품질 향상 설정들
			self.api.SetVariable("tessedit_enable_dict_correction", "1")  # 사전 교정
			self.api.SetVariable("tessedit_enable_bigram_correction", "1")  # 바이그램 교정
			self.api.SetVariable("classify_enable_learning", "1")  # 학습 기능
			
		except Exception as e:
			print(f"Warning: Failed to set OCR parameters: {e}")

		# OCR 실행
		self.api.SetImage(binary_pil)
		raw = self.api.GetUTF8Text() or ""

		# 향상된 후처리
		text = self._postprocess_text(raw)
		return text
	
	def _postprocess_text(self, raw_text: str) -> str:
		"""OCR 결과 후처리"""
		if not raw_text:
			return ""
		
		# 공백 정규화
		text = ' '.join(raw_text.split())
		
		# 일반적인 OCR 오류 수정
		corrections = {
			'0': ['O', 'o'],  # 숫자 0과 문자 O 혼동
			'1': ['l', 'I', '|'],  # 숫자 1과 소문자 l, 대문자 I 혼동
			'5': ['S'],  # 숫자 5와 문자 S 혼동
			'8': ['B'],  # 숫자 8과 문자 B 혼동
		}
		
		# 문맥에 따른 교정 (숫자가 많은 문자열에서)
		if any(c.isdigit() for c in text):
			for correct, wrongs in corrections.items():
				for wrong in wrongs:
					text = text.replace(wrong, correct)
		
		# 특수 문자 정리 및 허용된 문자만 유지
		allowed_chars = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가-힣 ,.-%₩:()[]#/@&+")
		cleaned = ''.join(c for c in text if c in allowed_chars or '\uAC00' <= c <= '\uD7A3')
		
		return cleaned.strip()

	def _normalize_for_match(self, s: str) -> str:
		"""텍스트 매칭용 정규화: NFKC, 소문자, 공백 정리"""
		if not s:
			return ""
		s = unicodedata.normalize('NFKC', s)
		s = ' '.join(s.lower().split())
		return s

	def _tokenize_text(self, s: str) -> List[str]:
		"""한글/영문/숫자 토큰화"""
		if not s:
			return []
		return re.findall(r"[A-Za-z0-9\uAC00-\uD7A3]+", s)

	def _extract_numbers(self, s: str) -> List[str]:
		if not s:
			return []
		return re.findall(r"\d+", s)

	def text_similarity(self, text1: str, text2: str) -> float:
		# 둘 다 비어있는 경우 유사도를 낮게 설정하여 오매칭 방지
		if not text1 and not text2:
			return 0.2
		if not text1 or not text2:
			return 0.0

		n1 = self._normalize_for_match(text1)
		n2 = self._normalize_for_match(text2)
		if n1 == n2:
			return 1.0

		# 길이 유사도(너무 다른 길이면 과매칭 방지)
		len1, len2 = len(n1), len(n2)
		len_sim = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

		# 토큰/숫자 기반 일치도
		toks1, toks2 = self._tokenize_text(n1), self._tokenize_text(n2)
		set1, set2 = set(toks1), set(toks2)
		jacc = (len(set1 & set2) / max(1, len(set1 | set2))) if (set1 or set2) else 0.0
		nums1, nums2 = set(self._extract_numbers(n1)), set(self._extract_numbers(n2))
		if nums1 and nums2 and nums1.isdisjoint(nums2):
			# 숫자 불일치 시 강한 패널티
			return 0.0

		# 긴 문장 vs 짧은 단어 과매칭 방지 규칙
		max_len = max(len1, len2)
		min_len = min(len1, len2)
		max_tok = max(len(set1), len(set2))
		min_tok = min(len(set1), len(set2))
		# 극단적 길이/토큰 차이에서 단어 하나만 겹치면 컷
		if (max_len >= 12 or max_tok >= 4) and (min_len <= 3 or min_tok <= 1):
			return 0.0

		# 길이 비율이 낮을 때 긴 쪽 토큰 커버리지 기준 필터링
		coverage_long = (len(set1 & set2) / max(1, max_tok)) if max_tok > 0 else 0.0
		if len_sim < 0.6 and coverage_long < 0.6:
			return 0.0

		# 토큰 겹침이 거의 없고 길이도 제법 다르면 컷
		if jacc < 0.2 and len_sim < 0.8:
			return 0.0

		# 문자 기반 유사도(정렬된 토큰 문자열 + 원문)
		sorted_join1 = ' '.join(sorted(toks1))
		sorted_join2 = ' '.join(sorted(toks2))
		char_sim = max(
			SequenceMatcher(None, n1, n2).ratio(),
			SequenceMatcher(None, sorted_join1, sorted_join2).ratio()
		)

		# n-gram(2gram) 겹침 정도(짧은 토큰 대비)
		def bigrams(s: str) -> set:
			return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else set()
		b1, b2 = bigrams(n1), bigrams(n2)
		ng_sim = (2 * len(b1 & b2) / max(1, len(b1) + len(b2))) if (b1 or b2) else 0.0

		# 결합: 문자/토큰/바이그램을 혼합, 길이 유사도로 조절(긴-짧 페널티 강화)
		base = 0.5 * char_sim + 0.3 * jacc + 0.2 * ng_sim
		base *= (0.4 + 0.6 * len_sim)  # 0.4~1.0 범위

		# 더 보수적인 컷
		return float(base) if base >= 0.5 else 0.0


	def detect_boxes_yolo(self,
						pil_img: Image.Image,
						conf_thresh: float = 0.05,
						max_det: int = 500,
						extract_features: bool = False):
		orig_w, orig_h = pil_img.size
		resize_size = self.resize_size  # e.g. (640,640) 혹은 (1024,1024)

		# 1) 샤픈
		pil_img = pil_img.filter(
			ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1)
		)

		# 2) 단순 리사이즈 (종횡비 유지 제거)
		img_resized = pil_img.resize(resize_size, Image.Resampling.LANCZOS)
		img_np = np.array(img_resized)  # H×W×C, RGB
		
		# BGR, float32, [0–1]
		img_input = img_np[..., ::-1].astype(np.float32) / 255.0
		img_input = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0)
		
		# 3) Hook backbone to extract feature map during inference
		feat_map = None
		feature_map_storage = []
		if extract_features:
			# Hook the C2PSA layer (index 10) for better feature extraction in YOLO11
			# C2PSA provides richer features with Parallel Spatial Attention mechanism
			backbone_modules = list(self.yolo.model.model.children())
			hook_layer = backbone_modules[10]  # C2PSA (YOLO11's enhanced attention layer)
			handle = hook_layer.register_forward_hook(
				lambda m, inp, out: feature_map_storage.append(out.detach())
			)
		# 4) Inference (remove hook after prediction)
		results = self.yolo.predict(
			source=img_input,
			conf=conf_thresh,
			iou=0.1,
			max_det=max_det,
			verbose=False
		)[0]        # 4) Remove hook and retrieve feature map
		if extract_features:
			handle.remove()
			feat_map = feature_map_storage[0]

		# 5) 박스 좌표 복원 (단순 스케일링)
		boxes = results.boxes.xyxy.cpu().numpy().copy()
		scale_x = orig_w / resize_size[1]
		scale_y = orig_h / resize_size[0]
		boxes[:, [0, 2]] *= scale_x
		boxes[:, [1, 3]] *= scale_y

		scores = results.boxes.conf.cpu().numpy()
		classes = results.boxes.cls.cpu().numpy()

		# 6) confidence 필터 → score 순 정렬 → top-k
		mask = scores >= conf_thresh
		boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
		order = scores.argsort()[::-1][:max_det]
		boxes_final = boxes[order]
		scores_final = scores[order]
		classes_final = classes[order]

		if extract_features:
			return boxes_final, scores_final, classes_final, feat_map, (orig_w, orig_h)
		else:
			return boxes_final, scores_final, classes_final

	@staticmethod
	def calculate_iou(rect_box1, rect_box2):
		# 입력 타입에 따라 좌표 추출
		if isinstance(rect_box1, dict):
			# 딕셔너리 형태: {"x": x, "y": y, "width": w, "height": h}
			x1_min, y1_min = rect_box1["x"], rect_box1["y"]
			x1_max = rect_box1["x"] + rect_box1["width"]
			y1_max = rect_box1["y"] + rect_box1["height"]
		elif isinstance(rect_box1, (list, tuple, np.ndarray)):
			# 리스트/튜플/배열 형태: [x1, y1, x2, y2]
			x1_min, y1_min, x1_max, y1_max = rect_box1
		else:
			raise TypeError(f"Unsupported type for rect_box1: {type(rect_box1)}")
		
		if isinstance(rect_box2, dict):
			# 딕셔너리 형태: {"x": x, "y": y, "width": w, "height": h}
			x2_min, y2_min = rect_box2["x"], rect_box2["y"]
			x2_max = rect_box2["x"] + rect_box2["width"]
			y2_max = rect_box2["y"] + rect_box2["height"]
		elif isinstance(rect_box2, (list, tuple, np.ndarray)):
			# 리스트/튜플/배열 형태: [x1, y1, x2, y2]
			x2_min, y2_min, x2_max, y2_max = rect_box2
		else:
			raise TypeError(f"Unsupported type for rect_box2: {type(rect_box2)}")

		# 2) 교집합 영역 좌표
		inter_x_min = max(x1_min, x2_min)
		inter_y_min = max(y1_min, y2_min)
		inter_x_max = min(x1_max, x2_max)
		inter_y_max = min(y1_max, y2_max)

		# 3) 교집합 너비·높이 (겹치지 않으면 0)
		inter_w = max(0.0, inter_x_max - inter_x_min)
		inter_h = max(0.0, inter_y_max - inter_y_min)
		inter_area = inter_w * inter_h

		# 4) 각 박스 면적
		area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
		area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

		# 5) 합집합 면적
		union_area = area1 + area2 - inter_area
		if union_area <= 0:
			return 0.0

		# 6) IoU 계산
		return inter_area / union_area

	

	def extract_features_from_map(self, full_feat_map: torch.Tensor, boxes: np.ndarray, original_img_size: Tuple[int, int]) -> torch.Tensor:
		if len(boxes) == 0:
			return torch.empty(0)

		orig_w, orig_h = original_img_size
		resized_w, resized_h = self.resize_size

		# Scale boxes from original image coordinates to resized image coordinates
		scale_x_orig_to_resized = resized_w / orig_w
		scale_y_orig_to_resized = resized_h / orig_h

		boxes_resized_coords = boxes.copy()
		boxes_resized_coords[:, [0, 2]] *= scale_x_orig_to_resized
		boxes_resized_coords[:, [1, 3]] *= scale_y_orig_to_resized

		# Convert resized image coordinates to feature map coordinates
		feat_map_h, feat_map_w = full_feat_map.shape[2:]
		scale_x_resized_to_feat = feat_map_w / resized_w
		scale_y_resized_to_feat = feat_map_h / resized_h

		boxes_feat_map_coords = boxes_resized_coords.copy()
		boxes_feat_map_coords[:, [0, 2]] *= scale_x_resized_to_feat
		boxes_feat_map_coords[:, [1, 3]] *= scale_y_resized_to_feat

		# Add batch index for roi_align
		batch_indices = torch.zeros((boxes_feat_map_coords.shape[0], 1), dtype=torch.float32)
		roi_boxes = torch.cat([batch_indices, torch.from_numpy(boxes_feat_map_coords).float()], dim=1)

		# Perform ROI Align
		pooled_features = torchvision.ops.roi_align(full_feat_map, roi_boxes, output_size=(1, 1), spatial_scale=1.0)
		
		# Flatten and normalize
		pooled_features = pooled_features.view(pooled_features.size(0), -1)
		normalized_features = torch.nn.functional.normalize(pooled_features, p=2, dim=1)
		
		return normalized_features.cpu()


	def calculate_similarity(self, img1: Image.Image, img2: Image.Image, figma_fare: List[FigmaFare], web_extracted: List[ExtractedElement]) -> Dict[str, np.ndarray]:
		# Extract features for Figma elements from img1
		boxes1 = np.array([f.extracted.box for f in figma_fare])

		# Extract features for Web elements from img2
		boxes2 = np.array([e.box for e in web_extracted])

		text_sim = self.calculate_text_similarity_matrix(figma_fare, web_extracted)
		feature_sim = self.calculate_feature_similarity_matrix(figma_fare, web_extracted)
		size_sim = self.calculate_size_similarity_matrix(figma_fare, web_extracted)
		coordinate_sim = self.calculate_coordinate_similarity_matrix(img1, img2, boxes1, boxes2)
		return {'text': text_sim, 'feature': feature_sim, 'size': size_sim, 'coordinate': coordinate_sim}

	def calculate_text_similarity_matrix(self, figma_fare: List[FigmaFare], web_extracted: List[ExtractedElement]) -> np.ndarray:
		textsA = [FigmaFare.extracted.text for FigmaFare in figma_fare]
		textsB = [extracted2.text for extracted2 in web_extracted]
		text_sim = np.zeros((len(textsA), len(textsB)))
		for i in range(len(textsA)):
			for j in range(len(textsB)):
				text_sim[i, j] = self.text_similarity(textsA[i], textsB[j])
		# Keep raw similarity to combine with other cues; threshold later at final score
		return text_sim.astype(np.float32)

	def calculate_feature_similarity_matrix(
		self,
		figma_fare: List[FigmaFare],
		web_extracted: List[ExtractedElement],
	) -> np.ndarray:
		# Handle empty inputs early
		N, M = len(figma_fare), len(web_extracted)
		if N == 0 or M == 0:
			return np.zeros((N, M), dtype=np.float32)

		with torch.no_grad():
			# Safe converter: handle torch.Tensor or np.ndarray; copy if non-writable
			def _to_float_tensor(x) -> torch.Tensor:
				if isinstance(x, torch.Tensor):
					return x.detach().to(dtype=torch.float32, copy=False)
				arr = np.asarray(x)
				if not arr.flags.writeable:
					arr = arr.copy()
				return torch.from_numpy(arr.astype(np.float32, copy=False))

			featuresA = torch.stack([
				_to_float_tensor(getattr(f.extracted, "feature"))
				for f in figma_fare
			], dim=0)
			featuresB = torch.stack([
				_to_float_tensor(getattr(e, "feature"))
				for e in web_extracted
			], dim=0)

			# Extra safety: normalize to ensure cosine via dot-product
			featuresA = torch.nn.functional.normalize(featuresA, p=2, dim=1)
			featuresB = torch.nn.functional.normalize(featuresB, p=2, dim=1)

			# Similarity matrix (cosine if normalized)
			sim_matrix = featuresA @ featuresB.T  # (N, M)
			return sim_matrix.cpu().numpy().astype(np.float32)

	def calculate_size_similarity_matrix(self, figma_fare: List[FigmaFare], web_extracted: List[ExtractedElement]) -> np.ndarray:
		boxes1 = np.array([FigmaFare.extracted.box for FigmaFare in figma_fare])
		boxes2 = np.array([extracted2.box for extracted2 in web_extracted])
		size_sim = np.zeros((len(boxes1), len(boxes2)))
		for i in range(len(boxes1)):
			for j in range(len(boxes2)):
				size_sim[i, j] = ElementExtractor.calculate_iou(boxes1[i], boxes2[j])
		return size_sim

	def calculate_coordinate_similarity_matrix(self, img1: Image.Image, img2: Image.Image, boxes1: np.ndarray, boxes2: np.ndarray, sigma: float = 0.2) -> np.ndarray:
		sim_mat = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
		for i in range(len(boxes1)):
			for j in range(len(boxes2)):
				sim_mat[i, j] = self.compute_coordinate_similarity(boxes1[i], boxes2[j], img1.size, img2.size, sigma=sigma)
		return sim_mat

	def compute_coordinate_similarity(self, box1: np.ndarray, box2: np.ndarray, size1: Tuple[int,int], size2: Tuple[int,int], sigma: float = 0.2) -> float:
		cx1, cy1 = (box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0
		cx2, cy2 = (box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0
		W1, H1 = size1
		W2, H2 = size2
		nx1, ny1 = (cx1 / W1, cy1 / H1) if W1 > 0 and H1 > 0 else (0.0, 0.0)
		nx2, ny2 = (cx2 / W2, cy2 / H2) if W2 > 0 and H2 > 0 else (0.0, 0.0)
		dist_norm = np.sqrt((nx1 - nx2)**2 + (ny1 - ny2)**2)
		return float(np.exp(- (dist_norm**2) / (2 * sigma**2)))

	def resize_and_adjust_boxes(self, img: Image.Image, boxes: np.ndarray, target_size: Tuple[int, int]) -> Tuple[Image.Image, np.ndarray]:
		orig_w, orig_h = img.size
		target_w, target_h = target_size
		resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
		if len(boxes) > 0:
			adjusted_boxes = boxes.copy()
			adjusted_boxes[:, [0, 2]] *= target_w / orig_w
			adjusted_boxes[:, [1, 3]] *= target_h / orig_h
		else:
			adjusted_boxes = boxes
		return resized_img, adjusted_boxes 
	

	def get_matches(
		self,
		sim_dict: Dict[str, np.ndarray],
		figma_elements_data: List[FigmaFare],
		web_elements_data: List[ExtractedElement],
		min_similarity: float = None,
		config: Optional['SimilarityConfig'] = None
	) -> Tuple[List[MatchResult], List[MatchResult], List[MatchResult]]:
		
		# 설정 클래스 초기화
		if config is None:
			from .models import SimilarityConfig
			config = SimilarityConfig.from_env()
		
		if min_similarity is None:
			min_similarity = config.weights.MIN_SIMILARITY
		
		weights = config.weights
		
		# 1. Weighted similarity calculation with robust row-wise normalization
		def _rowwise_quantile_normalize(mat: np.ndarray, q_low: float = weights.QUANTILE_LOW, q_high: float = weights.QUANTILE_HIGH) -> np.ndarray:
			if mat.size == 0:
				return mat.astype(np.float32)
			low = np.quantile(mat, q_low, axis=1, keepdims=True)
			high = np.quantile(mat, q_high, axis=1, keepdims=True)
			range_ = np.clip(high - low, 1e-6, None)
			mat_clipped = np.clip(mat, low, high)
			return ((mat_clipped - low) / range_).astype(np.float32)

		def _rowwise_softmax(mat: np.ndarray, tau: float = config.softmax_tau) -> np.ndarray:
			if mat.size == 0:
				return mat.astype(np.float32)
			# subtract row-wise max for numerical stability
			row_max = np.max(mat, axis=1, keepdims=True)
			exp = np.exp((mat - row_max) / max(tau, 1e-6))
			sum_exp = np.sum(exp, axis=1, keepdims=True) + 1e-9
			return (exp / sum_exp).astype(np.float32)

		# Relative (row-wise) views for competitive assignment: quantile-norm or softmax (config toggle)
		use_softmax_rel = config.use_softmax_relative
		tau = config.softmax_tau
		if use_softmax_rel:
			text_mat_rel = _rowwise_softmax(sim_dict['text'].astype(np.float32), tau)
			# feature/size/coord may have different ranges; softmax handles relative ranking
			feat_mat_rel = _rowwise_softmax(sim_dict['feature'].astype(np.float32), tau)
			size_mat_rel = _rowwise_softmax(sim_dict['size'].astype(np.float32), tau)
			coord_mat_rel = _rowwise_softmax(sim_dict['coordinate'].astype(np.float32), tau)
		else:
			text_mat_rel = _rowwise_quantile_normalize(sim_dict['text'])
			feat_mat_rel = _rowwise_quantile_normalize(sim_dict['feature'])
			size_mat_rel = _rowwise_quantile_normalize(sim_dict['size'])
			coord_mat_rel = _rowwise_quantile_normalize(sim_dict['coordinate'])

		# Absolute views keep original scales (clamp to [0,1] where needed)
		text_mat_abs = sim_dict['text'].astype(np.float32)
		feat_mat_abs = np.clip(sim_dict['feature'], 0.0, 1.0).astype(np.float32)  # cosine may be [-1,1]
		size_mat_abs = np.clip(sim_dict['size'], 0.0, 1.0).astype(np.float32)
		coord_mat_abs = np.clip(sim_dict['coordinate'], 0.0, 1.0).astype(np.float32)

		# Determine matrix size from any of the views (they share shape)
		N, M = text_mat_abs.shape
		if N == 0 or M == 0:
			return [], [
				MatchResult(figma=figma, web=None, feature_similarity=0.0, text_similarity=0.0, size_similarity=0.0, coordinate_similarity=0.0, score=0.0, errorCategories=[G_ERROR_NOT_MATCHED])
				for figma in figma_elements_data
			], [
				MatchResult(figma=None, web=web, feature_similarity=0.0, text_similarity=0.0, size_similarity=0.0, coordinate_similarity=0.0, score=0.0, errorCategories=[G_ERROR_NOT_MATCHED])
				for web in web_elements_data
			]

		# 동적 텍스트 가중치: 설정 클래스 사용
		figma_has_text = np.array([bool(getattr(f.extracted, 'text', '') or '') for f in figma_elements_data], dtype=bool)
		web_has_text = np.array([bool(getattr(w, 'text', '') or '') for w in web_elements_data], dtype=bool)
		both_have_text = np.outer(figma_has_text, web_has_text)
		w_text_base = np.where(both_have_text, weights.TEXT_WITH_BOTH, weights.TEXT_WITHOUT).astype(np.float32)
		# 텍스트 유사도가 낮을수록 텍스트 가중치를 감소
		text_scale_factor = weights.TEXT_SCALE_MIN + (weights.TEXT_SCALE_MAX - weights.TEXT_SCALE_MIN) * np.clip(text_mat_abs, 0.0, 1.0)
		w_text_scaled = (w_text_base * text_scale_factor).astype(np.float32)
		w_feat_base = np.full((N, M), weights.FEATURE_BASE, dtype=np.float32)
		w_size = np.full((N, M), weights.SIZE_BASE, dtype=np.float32)
		w_coord = np.full((N, M), weights.COORDINATE_BASE, dtype=np.float32)
		# 분모는 base 가중치 합으로 고정하여, 텍스트/피처 가중치 조절이 다른 신호 부스팅으로 이어지지 않도록 고정
		w_sum = w_text_base + w_feat_base + w_size + w_coord

		# 텍스트 요소일 때 피처 가중치 축소
		feat_scale_both = weights.FEAT_SCALE_BOTH_TEXT
		feat_scale_xor = weights.FEAT_SCALE_XOR_TEXT
		pair_has_text = np.logical_or.outer(figma_has_text, web_has_text)
		pair_both_text = both_have_text
		feat_scale = np.where(pair_both_text, feat_scale_both, np.where(pair_has_text, feat_scale_xor, 1.0)).astype(np.float32)
		# 텍스트 유사도가 낮을수록 피처 가중치도 더 줄임
		feat_text_scale = weights.FEAT_SCALE_MIN + (1.0 - weights.FEAT_SCALE_MIN) * np.clip(text_mat_abs, 0.0, 1.0)
		feat_scale *= feat_text_scale
		w_feat_scaled = (w_feat_base * feat_scale).astype(np.float32)

		# Relative combined score (for assignment)
		sim_matrix_rel = (text_mat_rel * w_text_scaled + feat_mat_rel * w_feat_scaled + size_mat_rel * w_size + coord_mat_rel * w_coord) / w_sum
		# Absolute combined score (for gating)
		sim_matrix_abs = (text_mat_abs * w_text_scaled + feat_mat_abs * w_feat_scaled + size_mat_abs * w_size + coord_mat_abs * w_coord) / w_sum

		if config.debug_mode or getattr(self, 'debug_similarity', False):
			import matplotlib.pyplot as plt
			import seaborn as sns
			debug_dir = os.path.join(os.path.dirname(__file__), 'debug_sim')
			os.makedirs(debug_dir, exist_ok=True)
			def save_heatmap(mat: np.ndarray, title: str, fname: str, vmin: float = 0.0, vmax: float = 1.0):
				plt.figure(figsize=(10, 7))
				sns.heatmap(mat, vmin=vmin, vmax=vmax, cmap="viridis")
				plt.title(title)
				plt.xlabel("Web Elements")
				plt.ylabel("Figma Elements")
				plt.tight_layout()
				plt.savefig(os.path.join(debug_dir, fname), dpi=150)
				plt.close()
			# Absolute
			save_heatmap(text_mat_abs,  'Text (abs)',  'text_abs.png')
			save_heatmap(feat_mat_abs,  'Feature (abs)',  'feature_abs.png')
			save_heatmap(size_mat_abs,  'Size (abs)',  'size_abs.png')
			save_heatmap(coord_mat_abs, 'Coord (abs)', 'coord_abs.png')
			save_heatmap(sim_matrix_abs, 'Combined (abs)', 'combined_abs.png')
			# Relative
			save_heatmap(text_mat_rel,  'Text (rel)',  'text_rel.png')
			save_heatmap(feat_mat_rel,  'Feature (rel)',  'feature_rel.png')
			save_heatmap(size_mat_rel,  'Size (rel)',  'size_rel.png')
			save_heatmap(coord_mat_rel, 'Coord (rel)', 'coord_rel.png')
			save_heatmap(sim_matrix_rel, 'Combined (rel)', 'combined_rel.png')
			# Top-K textual summary
			try:
				K = min(5, M)
				lines = []
				for i in range(N):
					order_abs = np.argsort(-sim_matrix_abs[i])[:K]
					order_rel = np.argsort(-sim_matrix_rel[i])[:K]
					lines.append(f"Figma {i} | ABS top{K}: " + ", ".join([f"j={int(j)}:{sim_matrix_abs[i,j]:.2f}" for j in order_abs]))
					lines.append(f"Figma {i} | REL top{K}: " + ", ".join([f"j={int(j)}:{sim_matrix_rel[i,j]:.2f}" for j in order_rel]))
				open(os.path.join(debug_dir, 'topk.txt'), 'w').write("\n".join(lines))
			except Exception:
				pass
		

		# # 2. Normalize to [0,1]
		# min_val, max_val = sim_matrix.min(), sim_matrix.max()
		# sim_matrix = (sim_matrix - min_val) / (max_val - min_val + 1e-8)

		# 3. Optimal assignment (Hungarian). Minimize cost = 1 - similarity
		N, M = sim_matrix_rel.shape
		matches: List[MatchResult] = []
		# Use relative matrix to find the best structure of matches
		row_ind, col_ind = linear_sum_assignment(1.0 - sim_matrix_rel)
		row_max_rel = sim_matrix_rel.max(axis=1) if N > 0 else np.array([])
		rel_keep = 0.85
		used_figma = set()
		used_web = set()
		for i, j in zip(row_ind, col_ind):
			score_rel = float(sim_matrix_rel[i, j])
			score_abs = float(sim_matrix_abs[i, j])
			# 상대 임계값과 절대 임계값을 병행
			row_thr = float(max(min_similarity, row_max_rel[i] * rel_keep))
			if score_rel < row_thr or score_abs < min_similarity:
				continue
			# 텍스트 존재 비대칭/양자 존재 시 추가 게이트
			fig_txt = getattr(figma_elements_data[i].extracted, 'text', '') or ''
			web_txt = getattr(web_elements_data[j], 'text', '') or ''
			fig_has = len(fig_txt.strip()) > 0
			web_has = len(web_txt.strip()) > 0
			text_sim = float(sim_dict['text'][i, j])
			# 환경변수로 요구 임계 설정
			try:
				req_both = float(os.environ.get('SIM_TEXT_REQ_BOTH', '0.40'))
				req_xor = float(os.environ.get('SIM_TEXT_REQ_XOR', '0.20'))
				min_len = int(os.environ.get('SIM_TEXT_MIN_LEN', '3'))
			except Exception:
				req_both, req_xor, min_len = 0.40, 0.20, 3
			if fig_has and web_has:
				if text_sim < req_both:
					continue
			elif fig_has ^ web_has:
				# 한쪽만 텍스트가 있는 경우, 긴 텍스트가 있으면 최소 유사도 요구
				if max(len(fig_txt), len(web_txt)) >= min_len and text_sim < req_xor:
					continue
			mr = MatchResult(
				figma=figma_elements_data[i],
				web=web_elements_data[j],
				feature_similarity=float(sim_dict['feature'][i, j]),
				text_similarity=float(sim_dict['text'][i, j]),
				size_similarity=float(sim_dict['size'][i, j]),
				coordinate_similarity=float(sim_dict['coordinate'][i, j]),
				score=score_abs,
				errorCategories=ErrorChecker().check_error(
					figma_elements_data[i].extracted.box,
					web_elements_data[j].box,
					figma_elements_data[i].extracted.text,
					web_elements_data[j].text
				)
			)
			matches.append(mr)
			used_figma.add(i)
			used_web.add(j)

		# 남은 것들 집합 계산
		unmatched_figma_idxs = set(range(N)) - used_figma
		unmatched_web_idxs = set(range(M)) - used_web

		# 4. (선택) IOU 기반 제외 처리 — 필요 없으면 건너뛰셔도 됩니다.
		for mr in matches:
			# figma 중 IOU 높으면 제거
			to_remove = {
				i for i in unmatched_figma_idxs
				if ElementExtractor.calculate_iou(
					mr.figma.extracted.box,
					figma_elements_data[i].extracted.box
				) > 0.5
			}
			unmatched_figma_idxs -= to_remove

			# web 중 IOU 높으면 제거
			to_remove = {
				j for j in unmatched_web_idxs
				if ElementExtractor.calculate_iou(
					mr.web.box,
					web_elements_data[j].box
				) > 0.5
			}
			unmatched_web_idxs -= to_remove

		# 5. unmatched MatchResult 생성
		unmatched_figma = [
			MatchResult(
				figma=figma_elements_data[i],
				web=None,
				feature_similarity=0.0,
				text_similarity=0.0,
				size_similarity=0.0,
				coordinate_similarity=0.0,
				score=0.0,
				errorCategories=[G_ERROR_NOT_MATCHED]
			)
			for i in sorted(unmatched_figma_idxs)
		]
		unmatched_web = [
			MatchResult(
				figma=None,
				web=web_elements_data[j],
				feature_similarity=0.0,
				text_similarity=0.0,
				size_similarity=0.0,
				coordinate_similarity=0.0,
				score=0.0,
				errorCategories=[G_ERROR_NOT_MATCHED]
			)
			for j in sorted(unmatched_web_idxs)
		]

		return matches, unmatched_figma, unmatched_web
