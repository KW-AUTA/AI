"""
Element Matcher - 레거시 요소 매칭 클래스

이 모듈은 하위 호환성을 위해 유지되는 레거시 ElementExtractor 클래스입니다.
새로운 코드에서는 extractor.py의 ElementExtractor 사용을 권장합니다.
"""

import os
import re
import unicodedata
import cv2
import numpy as np

# CRITICAL: PaddlePaddle을 PyTorch보다 먼저 import해야 함
# macOS에서 libc++ std::string 충돌 방지
_USE_PADDLE = os.environ.get('USE_PADDLEOCR', 'false').lower() in ('true', '1', 'yes')
_PADDLE_IMPORTED = False

if _USE_PADDLE:
	try:
		# Paddle을 먼저 import
		import paddle
		paddle_version = paddle.__version__

		# PaddleOCR 준비
		from paddleocr import PaddleOCR as _PaddleOCR
		import logging
		logging.getLogger('ppocr').setLevel(logging.ERROR)
		_PADDLE_IMPORTED = True
		print(f"✓ PaddlePaddle {paddle_version} pre-loaded (before PyTorch)")
	except Exception as e:
		_PADDLE_IMPORTED = False
		print(f"⚠️ PaddlePaddle pre-load failed: {e}")

# PyTorch는 PaddlePaddle 이후에 import
import torch
import torch.nn.functional as F
import torchvision.ops
import torchvision.transforms as T
from PIL import Image, ImageFilter
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
import tesserocr

from .models import FigmaFare, ExtractedElement, MatchResult
from .paddle_ocr_helper import PaddleOCRHelper
from ..utils.errorChecker import ErrorChecker
from ..utils.error_list import *


# ============================================================================
# Helper Functions
# ============================================================================

def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
	"""
	Resize and pad image to meet new_shape, maintaining aspect ratio.

	Args:
		im: Input image (numpy array)
		new_shape: Target shape (height, width)
		color: Padding color

	Returns:
		Tuple of (padded_image, resize_ratio, (left_padding, top_padding))
	"""
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


def non_max_suppression(
	boxes: np.ndarray,
	scores: Optional[np.ndarray] = None,
	iou_threshold: float = 0.5
) -> List[int]:
	"""
	Perform Non-Maximum Suppression on axis-aligned bounding boxes.

	Args:
		boxes: numpy array of shape (N,4) in (x1,y1,x2,y2) format
		scores: confidence scores array of shape (N,), defaults to all 1s
		iou_threshold: IOU threshold for NMS

	Returns:
		List of indices to keep
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


# ============================================================================
# Legacy ElementExtractor Class
# ============================================================================

class ElementExtractor:
	"""
	레거시 요소 추출 및 매칭 클래스

	주요 기능:
	- YOLO 기반 요소 검출
	- Tesseract OCR을 통한 텍스트 추출
	- 특징 추출 및 유사도 계산
	- 최적 매칭 알고리즘

	Note:
		새로운 코드에서는 extractor.py의 ElementExtractor 사용을 권장합니다.
	"""

	def __init__(
		self,
		yolo_model_path: str = None,
		resize_size: Tuple[int, int] = (736, 736),
		debug_similarity: bool = False,
		use_paddleocr: bool = False  # macOS 충돌 문제로 기본값 False
	):
		"""
		Args:
			yolo_model_path: YOLO 모델 경로
			resize_size: 이미지 리사이즈 크기
			debug_similarity: 디버그 모드 활성화
			use_paddleocr: PaddleOCR 사용 여부 (기본값: False, True이면 subprocess 기반 PaddleOCR 사용)
		"""
		# macOS 환경 설정 (PaddlePaddle + YOLO 충돌 방지)
		os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
		os.environ['OMP_NUM_THREADS'] = '1'
		os.environ['MKL_NUM_THREADS'] = '1'

		# YOLO 모델 초기화
		current_dir = os.path.dirname(__file__)
		if yolo_model_path is None:
				yolo_model_path = os.path.join(current_dir, "..", "models_weights", "best_one.pt")
		self.yolo = YOLO(yolo_model_path, task='detect', verbose=False)
		self.resize_size = resize_size
		self.debug_similarity = debug_similarity or str(os.environ.get('SIM_DEBUG', '1')).lower() in ('1', 'true', 'yes')

		# OCR 엔진 선택
		self.use_paddleocr = use_paddleocr

		if use_paddleocr:
			# PaddleOCR Helper 초기화 (subprocess 기반 - 충돌 없음)
			try:
				self.paddle_helper = PaddleOCRHelper()
				self.api = None
				self.paddle_ocr = None
				print("✓ PaddleOCR Helper 초기화 성공 (subprocess 기반)")
			except Exception as e:
				print(f"⚠️ PaddleOCR Helper 초기화 실패, Tesseract로 폴백: {e}")
				self.use_paddleocr = False
				possible_paths = [
						"/usr/share/tesseract-ocr/5/tessdata",
						"/usr/share/tessdata",
						"/usr/local/share/tessdata"
				]
				tessdata_dir = next((p for p in possible_paths if os.path.exists(p)), None)
				if tessdata_dir is None:
						raise RuntimeError("❌ Tesseract tessdata를 찾을 수 없습니다. 설치 확인 필요.")

				self.api = tesserocr.PyTessBaseAPI(path=tessdata_dir, lang='kor')
				self.api.SetVariable("user_defined_dpi", "300")
				self.api.SetVariable("tessedit_char_blacklist", "")
				self.api.SetVariable("preserve_interword_spaces", "1")
				self.paddle_helper = None
		else:
			# Tesseract OCR 설정 (한글+영어 지원)
			
			possible_paths = [
						"/usr/share/tesseract-ocr/5/tessdata",
						"/usr/share/tessdata",
						"/usr/local/share/tessdata"
			]
			tessdata_dir = next((p for p in possible_paths if os.path.exists(p)), None)
			if tessdata_dir is None:
				raise RuntimeError("❌ Tesseract tessdata를 찾을 수 없습니다. 설치 확인 필요.")

			self.api = tesserocr.PyTessBaseAPI(path=tessdata_dir, lang='kor+eng')
			self.api.SetVariable("user_defined_dpi", "300")
			self.api.SetVariable("tessedit_char_blacklist", "")
			self.api.SetVariable("preserve_interword_spaces", "1")
			self.paddle_helper = None

		# Image transform
		self.transform = T.Compose([
			T.Resize(resize_size),
			T.ToTensor(),
		])

	def __del__(self):
		"""Tesseract API 정리"""
		if hasattr(self, 'api') and self.api is not None:
			self.api.End()

	# ========================================================================
	# OCR Methods
	# ========================================================================

	def _pick_psm(self, w: int, h: int) -> int:
		"""ROI 크기/종횡비에 따라 최적화된 PSM 선택"""
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
		# 매우 가로로 긴 형태: 단일 라인
		elif aspect >= 4.0:
			return tesserocr.PSM.SINGLE_LINE
		# 세로로 긴 형태나 중간 크기의 가로 긴 형태: 단일 라인
		elif aspect <= 0.5 or aspect >= 2.0:
			return tesserocr.PSM.SINGLE_LINE
		# 기본: 단일 블록 처리
		else:
			return tesserocr.PSM.SINGLE_BLOCK

	def _preprocess_roi(
		self,
		pil_crop: Image.Image,
		scale: float = None,
		pad: int = 15
	) -> Image.Image:
		"""개선된 ROI 전처리로 OCR 인식률 향상

		어두운 배경 + 밝은 텍스트 자동 감지 및 반전 처리
		"""
		arr = np.array(pil_crop)
		if arr.ndim == 3:
			gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
		else:
			gray = arr

		# 동적 스케일링 (ROI 크기에 따라 조정)
		h, w = gray.shape
		if scale is None:
			if max(h, w) < 30:
				scale = 4.0
			elif max(h, w) < 60:
				scale = 3.0
			elif max(h, w) < 120:
				scale = 2.0
			else:
				scale = 1.5

		# 스케일링 (Lanczos 보간법)
		if scale > 1.0:
			new_w, new_h = int(w * scale), int(h * scale)
			gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

		# 대비 향상 (CLAHE)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		enhanced = clahe.apply(gray)

		# 노이즈 제거
		denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)

		# 배경 밝기 자동 감지
		# 이미지 외곽 영역의 평균 밝기로 배경 추정
		border_width = max(5, min(denoised.shape) // 10)
		top_border = denoised[:border_width, :]
		bottom_border = denoised[-border_width:, :]
		left_border = denoised[:, :border_width]
		right_border = denoised[:, -border_width:]

		background_brightness = np.mean([
			np.mean(top_border),
			np.mean(bottom_border),
			np.mean(left_border),
			np.mean(right_border)
		])

		# 배경이 어두우면 (밝기 < 128) 반전 필요
		# 어두운 배경 + 밝은 텍스트 → 밝은 배경 + 어두운 텍스트로 변환
		needs_inversion = background_brightness < 128

		if needs_inversion:
			# THRESH_BINARY_INV: 어두운 배경을 흰색으로, 밝은 텍스트를 검은색으로
			binary = cv2.adaptiveThreshold(
				denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
				cv2.THRESH_BINARY_INV, 11, 2
			)
		else:
			# THRESH_BINARY: 일반적인 경우 (밝은 배경 + 어두운 텍스트)
			binary = cv2.adaptiveThreshold(
				denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
				cv2.THRESH_BINARY, 11, 2
			)

		# 모폴로지 연산으로 텍스트 구조 개선
		if needs_inversion:
			# 어두운 배경의 경우: 외곽선 텍스트를 채우는 강력한 처리
			# 1. 강한 Dilation으로 외곽선 채우기
			kernel_dilate = np.ones((4, 4), np.uint8)
			binary = cv2.dilate(binary, kernel_dilate, iterations=2)

			# 2. Closing으로 남은 구멍 메우기
			kernel_close = np.ones((3, 3), np.uint8)
			binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

			# 3. Erosion으로 너무 두꺼워진 글자 다시 얇게 (약하게)
			kernel_erode = np.ones((2, 2), np.uint8)
			binary = cv2.erode(binary, kernel_erode, iterations=1)
		else:
			# 일반적인 경우: 기본 Closing만
			kernel_close = np.ones((2, 2), np.uint8)
			binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

		# 여백 추가
		binary = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

		return Image.fromarray(binary)

	def extract_text(self, img: Image.Image, box: np.ndarray, image_path: str = None) -> str:
		"""이미지에서 텍스트 추출

		Args:
			img: 이미지 (PIL Image)
			box: 박스 좌표 [x1, y1, x2, y2]
			image_path: 원본 이미지 파일 경로 (PaddleOCR 사용 시 필요)

		Returns:
			추출된 텍스트
		"""
		if self.use_paddleocr and self.paddle_helper:
			# PaddleOCR Helper 사용 (subprocess 기반)
			if image_path is None:
				# 임시 파일로 저장
				import tempfile
				with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
					img.save(f.name)
					image_path = f.name
					result = self.paddle_helper.extract_text_single(image_path, box)
					os.unlink(image_path)
					return self._postprocess_text(result)
			else:
				result = self.paddle_helper.extract_text_single(image_path, box)
				return self._postprocess_text(result)
		else:
			# Tesseract 사용 (레거시)
			x1, y1, x2, y2 = map(int, box)
			pad = 12
			x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
			x2p, y2p = min(img.width, x2 + pad), min(img.height, y2 + pad)
			crop_img = img.crop((x1p, y1p, x2p, y2p))
			return self._extract_text_tesseract(crop_img, x2p - x1p, y2p - y1p)

	def extract_text_batch(self, image_path: str, boxes: List[np.ndarray]) -> List[str]:
		"""여러 박스에서 텍스트 일괄 추출 (PaddleOCR 사용 시 효율적)

		Args:
			image_path: 이미지 파일 경로
			boxes: 박스 좌표 리스트 [[x1,y1,x2,y2], ...]

		Returns:
			추출된 텍스트 리스트
		"""
		if self.use_paddleocr and self.paddle_helper:
			# PaddleOCR Helper 사용 (batch 처리)
			raw_texts = self.paddle_helper.extract_text_batch(image_path, boxes)
			return [self._postprocess_text(t) for t in raw_texts]
		else:
			# Tesseract 사용 (개별 처리)
			img = Image.open(image_path)
			results = []
			for box in boxes:
				x1, y1, x2, y2 = map(int, box)
				pad = 12
				x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
				x2p, y2p = min(img.width, x2 + pad), min(img.height, y2 + pad)
				crop_img = img.crop((x1p, y1p, x2p, y2p))
				text = self._extract_text_tesseract(crop_img, x2p - x1p, y2p - y1p)
				results.append(text)
			return results

	def _extract_text_tesseract(self, crop_img: Image.Image, roi_w: int, roi_h: int) -> str:
		"""Tesseract를 사용한 텍스트 추출 (레거시)"""
		# 전처리 적용
		binary_pil = self._preprocess_roi(crop_img, pad=20)

		# PSM 선택
		psm = self._pick_psm(roi_w, roi_h)
		self.api.SetPageSegMode(psm)

		# OCR 설정
		try:
			self.api.SetVariable("tessedit_ocr_engine_mode", "1")  # LSTM 모드

			# 작은 텍스트나 숫자의 경우 화이트리스트 적용
			if psm in (tesserocr.PSM.SINGLE_WORD, tesserocr.PSM.SINGLE_CHAR) or roi_w * roi_h < 2000:
				self.api.SetVariable("tessedit_char_whitelist",
					"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가-힣 ,.-%₩:()[]#/@&")
			else:
				self.api.SetVariable("tessedit_char_whitelist", "")

			# 품질 향상 설정
			self.api.SetVariable("tessedit_enable_dict_correction", "1")
			self.api.SetVariable("tessedit_enable_bigram_correction", "1")
			self.api.SetVariable("classify_enable_learning", "1")
		except Exception as e:
			print(f"Warning: Failed to set OCR parameters: {e}")

		# OCR 실행
		self.api.SetImage(binary_pil)
		raw = self.api.GetUTF8Text() or ""

		# 후처리
		text = self._postprocess_text(raw)
		return text

	def _postprocess_text(self, raw_text: str) -> str:
		"""OCR 결과 후처리"""
		if not raw_text:
			return ""

		# 공백 정규화
		text = ' '.join(raw_text.split())

		# PaddleOCR은 이미 정확하므로 교정 로직 비활성화
		# Tesseract는 여전히 필요하지만, 전체 문자열이 아닌 개별 단어에 적용해야 함
		# 현재는 비활성화하여 "Product" → "Pr0duct" 같은 오교정 방지

		# 허용된 문자만 유지
		allowed_chars = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가-힣 ,.-%₩:()[]#/@&+=")
		cleaned = ''.join(c for c in text if c in allowed_chars or '\uAC00' <= c <= '\uD7A3')

		return cleaned.strip()

	# ========================================================================
	# Text Similarity Methods
	# ========================================================================

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
		"""숫자 추출"""
		if not s:
			return []
		return re.findall(r"\d+", s)

	def text_similarity(self, text1: str, text2: str) -> float:
		"""텍스트 유사도 계산"""
		# 둘 다 비어있는 경우
		if not text1 and not text2:
			return 0.2
		if not text1 or not text2:
			return 0.0

		n1 = self._normalize_for_match(text1)
		n2 = self._normalize_for_match(text2)
		if n1 == n2:
			return 1.0

		# 길이 유사도
		len1, len2 = len(n1), len(n2)
		len_sim = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

		# 토큰 기반 일치도
		toks1, toks2 = self._tokenize_text(n1), self._tokenize_text(n2)
		set1, set2 = set(toks1), set(toks2)
		jacc = (len(set1 & set2) / max(1, len(set1 | set2))) if (set1 or set2) else 0.0

		# 숫자 불일치 체크
		nums1, nums2 = set(self._extract_numbers(n1)), set(self._extract_numbers(n2))
		if nums1 and nums2 and nums1.isdisjoint(nums2):
			return 0.0

		# 극단적 길이/토큰 차이 필터링
		max_len = max(len1, len2)
		min_len = min(len1, len2)
		max_tok = max(len(set1), len(set2))
		min_tok = min(len(set1), len(set2))

		if (max_len >= 12 or max_tok >= 4) and (min_len <= 3 or min_tok <= 1):
			return 0.0

		# 커버리지 기반 필터링
		coverage_long = (len(set1 & set2) / max(1, max_tok)) if max_tok > 0 else 0.0
		if len_sim < 0.6 and coverage_long < 0.6:
			return 0.0

		if jacc < 0.2 and len_sim < 0.8:
			return 0.0

		# 문자 기반 유사도
		sorted_join1 = ' '.join(sorted(toks1))
		sorted_join2 = ' '.join(sorted(toks2))
		char_sim = max(
			SequenceMatcher(None, n1, n2).ratio(),
			SequenceMatcher(None, sorted_join1, sorted_join2).ratio()
		)

		# n-gram 겹침
		def bigrams(s: str) -> set:
			return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else set()

		b1, b2 = bigrams(n1), bigrams(n2)
		ng_sim = (2 * len(b1 & b2) / max(1, len(b1) + len(b2))) if (b1 or b2) else 0.0

		# 최종 점수
		base = 0.5 * char_sim + 0.3 * jacc + 0.2 * ng_sim
		base *= (0.4 + 0.6 * len_sim)

		return float(base) if base >= 0.5 else 0.0

	# ========================================================================
	# YOLO Detection Methods
	# ========================================================================

	def detect_boxes_yolo(
		self,
		pil_img: Image.Image,
		conf_thresh: float = 0.05,
		max_det: int = 500,
		extract_features: bool = False
	):
		"""YOLO를 사용한 박스 검출"""
		orig_w, orig_h = pil_img.size
		resize_size = self.resize_size

		# 샤프닝
		pil_img = pil_img.filter(
			ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1)
		)

		# 리사이즈
		img_resized = pil_img.resize(resize_size, Image.Resampling.LANCZOS)
		img_np = np.array(img_resized)

		# BGR, float32, [0-1]
		img_input = img_np[..., ::-1].astype(np.float32) / 255.0
		img_input = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0)

			# Multi-scale feature map 추출을 위한 hook
		feat_maps = []  # 여러 레이어의 특징 맵 저장
		handles = []  # hook 핸들 저장

		if extract_features:
			backbone_modules = list(self.yolo.model.model.children())

			# 여러 스테이지에서 특징 추출
			# Layer 6: 초기 특징 (에지, 텍스처) - 아이콘/Vector 구분에 유용
			# Layer 8: 중간 특징 (패턴, 형태)
			# Layer 10: 고수준 특징 (의미론적) - 텍스트 요소에 유용
			target_layers = [6, 8, 10]

			for layer_idx in target_layers:
				hook_layer = backbone_modules[layer_idx]
				handle = hook_layer.register_forward_hook(
					lambda m, inp, out, idx=layer_idx: feat_maps.append((idx, out.detach()))
				)
				handles.append(handle)

		# Inference
		results = self.yolo.predict(
			source=img_input,
			conf=conf_thresh,
			iou=0.1,
			max_det=max_det,
			verbose=False
		)[0]

			# Hook 제거 및 multi-scale feature maps 정리
		if extract_features:
			for handle in handles:
				handle.remove()

			# 레이어 인덱스순으로 정렬
			feat_maps.sort(key=lambda x: x[0])
			# feat_map은 (layer_idx, tensor) 튜플의 리스트
			feat_map = feat_maps  # [(6, tensor), (8, tensor), (10, tensor)]

		# 박스 좌표 복원
		boxes = results.boxes.xyxy.cpu().numpy().copy()
		scale_x = orig_w / resize_size[1]
		scale_y = orig_h / resize_size[0]
		boxes[:, [0, 2]] *= scale_x
		boxes[:, [1, 3]] *= scale_y

		scores = results.boxes.conf.cpu().numpy()
		classes = results.boxes.cls.cpu().numpy()

		# Confidence 필터 및 정렬
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
	def calculate_iou(rect_box1, rect_box2) -> float:
		"""두 박스 간의 IOU 계산"""
		# 입력 타입에 따라 좌표 추출
		if isinstance(rect_box1, dict):
			x1_min, y1_min = rect_box1["x"], rect_box1["y"]
			x1_max = rect_box1["x"] + rect_box1["width"]
			y1_max = rect_box1["y"] + rect_box1["height"]
		elif isinstance(rect_box1, (list, tuple, np.ndarray)):
			x1_min, y1_min, x1_max, y1_max = rect_box1
		else:
			raise TypeError(f"Unsupported type for rect_box1: {type(rect_box1)}")

		if isinstance(rect_box2, dict):
			x2_min, y2_min = rect_box2["x"], rect_box2["y"]
			x2_max = rect_box2["x"] + rect_box2["width"]
			y2_max = rect_box2["y"] + rect_box2["height"]
		elif isinstance(rect_box2, (list, tuple, np.ndarray)):
			x2_min, y2_min, x2_max, y2_max = rect_box2
		else:
			raise TypeError(f"Unsupported type for rect_box2: {type(rect_box2)}")

		# 교집합 영역 좌표
		inter_x_min = max(x1_min, x2_min)
		inter_y_min = max(y1_min, y2_min)
		inter_x_max = min(x1_max, x2_max)
		inter_y_max = min(y1_max, y2_max)

		# 교집합 면적
		inter_w = max(0.0, inter_x_max - inter_x_min)
		inter_h = max(0.0, inter_y_max - inter_y_min)
		inter_area = inter_w * inter_h

		# 각 박스 면적
		area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
		area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

		# 합집합 면적
		union_area = area1 + area2 - inter_area
		if union_area <= 0:
			return 0.0

		return inter_area / union_area

	# ========================================================================
	# Feature Extraction Methods
	# ========================================================================

	def extract_features_from_map(
		self,
		feat_maps: list,  # [(layer_idx, tensor), ...] 형식
		boxes: np.ndarray,
		original_img_size: Tuple[int, int]
	) -> torch.Tensor:
		"""Multi-scale feature maps에서 박스 영역의 특징 추출 및 결합"""
		if len(boxes) == 0:
			return torch.empty(0)

		# 레거시 호환성: 단일 텐서가 전달된 경우
		if isinstance(feat_maps, torch.Tensor):
			feat_maps = [(10, feat_maps)]  # 기존 단일 레이어로 처리

		orig_w, orig_h = original_img_size
		resized_w, resized_h = self.resize_size

		# 박스 좌표를 리사이즈된 이미지 좌표로 변환
		scale_x_orig_to_resized = resized_w / orig_w
		scale_y_orig_to_resized = resized_h / orig_h

		boxes_resized_coords = boxes.copy()
		boxes_resized_coords[:, [0, 2]] *= scale_x_orig_to_resized
		boxes_resized_coords[:, [1, 3]] *= scale_y_orig_to_resized

		# 각 레이어에서 특징 추출
		multi_scale_features = []

		for layer_idx, full_feat_map in feat_maps:
			# Feature map 좌표로 변환
			feat_map_h, feat_map_w = full_feat_map.shape[2:]
			scale_x_resized_to_feat = feat_map_w / resized_w
			scale_y_resized_to_feat = feat_map_h / resized_h

			boxes_feat_map_coords = boxes_resized_coords.copy()
			boxes_feat_map_coords[:, [0, 2]] *= scale_x_resized_to_feat
			boxes_feat_map_coords[:, [1, 3]] *= scale_y_resized_to_feat

			# ROI Align을 위한 배치 인덱스 추가
			batch_indices = torch.zeros((boxes_feat_map_coords.shape[0], 1), dtype=torch.float32)
			roi_boxes = torch.cat([batch_indices, torch.from_numpy(boxes_feat_map_coords).float()], dim=1)

			# ROI Align 수행
			pooled_features = torchvision.ops.roi_align(
				full_feat_map, roi_boxes, output_size=(1, 1), spatial_scale=1.0
			)

			# Flatten
			pooled_features = pooled_features.view(pooled_features.size(0), -1)
			multi_scale_features.append(pooled_features)

		# Multi-scale 특징 결합
		# 옵션 1: Concatenation (모든 스케일을 이어붙임)
		combined_features = torch.cat(multi_scale_features, dim=1)

		# 옵션 2: 차원 축소 후 정규화 (메모리 효율적)
		# 각 스케일을 같은 차원으로 projection한 후 평균
		# target_dim = 512
		# projected = []
		# for feat in multi_scale_features:
		#     if feat.shape[1] != target_dim:
		#         proj = torch.nn.functional.adaptive_avg_pool1d(
		#             feat.unsqueeze(1), target_dim
		#         ).squeeze(1)
		#         projected.append(proj)
		#     else:
		#         projected.append(feat)
		# combined_features = torch.stack(projected).mean(dim=0)

		# 정규화
		normalized_features = torch.nn.functional.normalize(combined_features, p=2, dim=1)

		return normalized_features.cpu()

	# ========================================================================
	# Similarity Calculation Methods
	# ========================================================================

	def calculate_similarity(
		self,
		img1: Image.Image,
		img2: Image.Image,
		figma_fare: List[FigmaFare],
		web_extracted: List[ExtractedElement]
	) -> Dict[str, np.ndarray]:
		"""모든 유사도 계산"""
		boxes1 = np.array([f.extracted.box for f in figma_fare])
		boxes2 = np.array([e.box for e in web_extracted])

		text_sim = self.calculate_text_similarity_matrix(figma_fare, web_extracted)
		feature_sim = self.calculate_feature_similarity_matrix(figma_fare, web_extracted)
		size_sim = self.calculate_size_similarity_matrix(figma_fare, web_extracted)
		coordinate_sim = self.calculate_coordinate_similarity_matrix(img1, img2, boxes1, boxes2)

		return {
			'text': text_sim,
			'feature': feature_sim,
			'size': size_sim,
			'coordinate': coordinate_sim
		}

	def calculate_text_similarity_matrix(
		self,
		figma_fare: List[FigmaFare],
		web_extracted: List[ExtractedElement]
	) -> np.ndarray:
		"""텍스트 유사도 행렬 계산"""
		textsA = [f.extracted.text for f in figma_fare]
		textsB = [e.text for e in web_extracted]
		text_sim = np.zeros((len(textsA), len(textsB)))

		for i in range(len(textsA)):
			for j in range(len(textsB)):
				text_sim[i, j] = self.text_similarity(textsA[i], textsB[j])

		return text_sim.astype(np.float32)

	def calculate_feature_similarity_matrix(
		self,
		figma_fare: List[FigmaFare],
		web_extracted: List[ExtractedElement]
	) -> np.ndarray:
		"""특징 유사도 행렬 계산"""
		N, M = len(figma_fare), len(web_extracted)
		if N == 0 or M == 0:
			return np.zeros((N, M), dtype=np.float32)

		with torch.no_grad():
			# 안전한 텐서 변환
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

			# 정규화
			featuresA = torch.nn.functional.normalize(featuresA, p=2, dim=1)
			featuresB = torch.nn.functional.normalize(featuresB, p=2, dim=1)

			# 유사도 행렬 (cosine similarity)
			sim_matrix = featuresA @ featuresB.T
			return sim_matrix.cpu().numpy().astype(np.float32)

	def calculate_size_similarity_matrix(
		self,
		figma_fare: List[FigmaFare],
		web_extracted: List[ExtractedElement]
	) -> np.ndarray:
		"""크기 유사도 행렬 계산 (IOU 기반)"""
		boxes1 = np.array([f.extracted.box for f in figma_fare])
		boxes2 = np.array([e.box for e in web_extracted])
		size_sim = np.zeros((len(boxes1), len(boxes2)))

		for i in range(len(boxes1)):
			for j in range(len(boxes2)):
				size_sim[i, j] = ElementExtractor.calculate_iou(boxes1[i], boxes2[j])

		return size_sim

	def calculate_coordinate_similarity_matrix(
		self,
		img1: Image.Image,
		img2: Image.Image,
		boxes1: np.ndarray,
		boxes2: np.ndarray,
		sigma: float = 0.2
	) -> np.ndarray:
		"""좌표 유사도 행렬 계산"""
		sim_mat = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

		for i in range(len(boxes1)):
			for j in range(len(boxes2)):
				sim_mat[i, j] = self.compute_coordinate_similarity(
					boxes1[i], boxes2[j], img1.size, img2.size, sigma=sigma
				)

		return sim_mat

	def compute_coordinate_similarity(
		self,
		box1: np.ndarray,
		box2: np.ndarray,
		size1: Tuple[int, int],
		size2: Tuple[int, int],
		sigma: float = 0.2
	) -> float:
		"""두 박스 간의 좌표 유사도 계산"""
		cx1, cy1 = (box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0
		cx2, cy2 = (box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0
		W1, H1 = size1
		W2, H2 = size2
		nx1, ny1 = (cx1 / W1, cy1 / H1) if W1 > 0 and H1 > 0 else (0.0, 0.0)
		nx2, ny2 = (cx2 / W2, cy2 / H2) if W2 > 0 and H2 > 0 else (0.0, 0.0)
		dist_norm = np.sqrt((nx1 - nx2)**2 + (ny1 - ny2)**2)
		return float(np.exp(- (dist_norm**2) / (2 * sigma**2)))

	def resize_and_adjust_boxes(
		self,
		img: Image.Image,
		boxes: np.ndarray,
		target_size: Tuple[int, int]
	) -> Tuple[Image.Image, np.ndarray]:
		"""이미지 리사이즈 및 박스 좌표 조정"""
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

	# ========================================================================
	# Matching Methods
	# ========================================================================

	def get_matches(
		self,
		sim_dict: Dict[str, np.ndarray],
		figma_elements_data: List[FigmaFare],
		web_elements_data: List[ExtractedElement],
		min_similarity: float = None,
		config: Optional['SimilarityConfig'] = None
	) -> Tuple[List[MatchResult], List[MatchResult], List[MatchResult]]:
		"""최적 매칭 수행 (Hungarian Algorithm)"""

		# 설정 초기화
		if config is None:
			from .models import SimilarityConfig
			config = SimilarityConfig.from_env()

		if min_similarity is None:
			min_similarity = config.weights.MIN_SIMILARITY

		weights = config.weights

		# 정규화 함수들
		def _rowwise_quantile_normalize(
			mat: np.ndarray,
			q_low: float = weights.QUANTILE_LOW,
			q_high: float = weights.QUANTILE_HIGH
		) -> np.ndarray:
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
			row_max = np.max(mat, axis=1, keepdims=True)
			exp = np.exp((mat - row_max) / max(tau, 1e-6))
			sum_exp = np.sum(exp, axis=1, keepdims=True) + 1e-9
			return (exp / sum_exp).astype(np.float32)

		# Relative views (경쟁적 할당을 위한)
		use_softmax_rel = config.use_softmax_relative
		tau = config.softmax_tau

		if use_softmax_rel:
			text_mat_rel = _rowwise_softmax(sim_dict['text'].astype(np.float32), tau)
			feat_mat_rel = _rowwise_softmax(sim_dict['feature'].astype(np.float32), tau)
			size_mat_rel = _rowwise_softmax(sim_dict['size'].astype(np.float32), tau)
			coord_mat_rel = _rowwise_softmax(sim_dict['coordinate'].astype(np.float32), tau)
		else:
			text_mat_rel = _rowwise_quantile_normalize(sim_dict['text'])
			feat_mat_rel = _rowwise_quantile_normalize(sim_dict['feature'])
			size_mat_rel = _rowwise_quantile_normalize(sim_dict['size'])
			coord_mat_rel = _rowwise_quantile_normalize(sim_dict['coordinate'])

		# Absolute views
		text_mat_abs = sim_dict['text'].astype(np.float32)
		feat_mat_abs = np.clip(sim_dict['feature'], 0.0, 1.0).astype(np.float32)
		size_mat_abs = np.clip(sim_dict['size'], 0.0, 1.0).astype(np.float32)
		coord_mat_abs = np.clip(sim_dict['coordinate'], 0.0, 1.0).astype(np.float32)

		# 행렬 크기 확인
		N, M = text_mat_abs.shape
		if N == 0 or M == 0:
			return [], [
				MatchResult(figma=figma, web=None, feature_similarity=0.0, text_similarity=0.0,
					size_similarity=0.0, coordinate_similarity=0.0, score=0.0,
					errorCategories=[G_ERROR_NOT_MATCHED])
				for figma in figma_elements_data
			], [
				MatchResult(figma=None, web=web, feature_similarity=0.0, text_similarity=0.0,
					size_similarity=0.0, coordinate_similarity=0.0, score=0.0,
					errorCategories=[G_ERROR_NOT_MATCHED])
				for web in web_elements_data
			]

		# 동적 텍스트 가중치
		figma_has_text = np.array([bool(getattr(f.extracted, 'text', '') or '') for f in figma_elements_data], dtype=bool)
		web_has_text = np.array([bool(getattr(w, 'text', '') or '') for w in web_elements_data], dtype=bool)
		both_have_text = np.outer(figma_has_text, web_has_text)

		w_text_base = np.where(both_have_text, weights.TEXT_WITH_BOTH, weights.TEXT_WITHOUT).astype(np.float32)
		text_scale_factor = weights.TEXT_SCALE_MIN + (weights.TEXT_SCALE_MAX - weights.TEXT_SCALE_MIN) * np.clip(text_mat_abs, 0.0, 1.0)
		w_text_scaled = (w_text_base * text_scale_factor).astype(np.float32)

		w_feat_base = np.full((N, M), weights.FEATURE_BASE, dtype=np.float32)
		w_size = np.full((N, M), weights.SIZE_BASE, dtype=np.float32)
		w_coord = np.full((N, M), weights.COORDINATE_BASE, dtype=np.float32)
		w_sum = w_text_base + w_feat_base + w_size + w_coord

		# 피처 가중치 조정
		feat_scale_both = weights.FEAT_SCALE_BOTH_TEXT
		feat_scale_xor = weights.FEAT_SCALE_XOR_TEXT
		pair_has_text = np.logical_or.outer(figma_has_text, web_has_text)
		pair_both_text = both_have_text
		feat_scale = np.where(pair_both_text, feat_scale_both, np.where(pair_has_text, feat_scale_xor, 1.0)).astype(np.float32)
		feat_text_scale = weights.FEAT_SCALE_MIN + (1.0 - weights.FEAT_SCALE_MIN) * np.clip(text_mat_abs, 0.0, 1.0)
		feat_scale *= feat_text_scale
		w_feat_scaled = (w_feat_base * feat_scale).astype(np.float32)

		# 종합 유사도 행렬
		sim_matrix_rel = (text_mat_rel * w_text_scaled + feat_mat_rel * w_feat_scaled + size_mat_rel * w_size + coord_mat_rel * w_coord) / w_sum
		sim_matrix_abs = (text_mat_abs * w_text_scaled + feat_mat_abs * w_feat_scaled + size_mat_abs * w_size + coord_mat_abs * w_coord) / w_sum

		# 디버그 모드
		if config.debug_mode or getattr(self, 'debug_similarity', False):
			self._save_debug_similarity_plots(
				text_mat_abs, feat_mat_abs, size_mat_abs, coord_mat_abs, sim_matrix_abs,
				text_mat_rel, feat_mat_rel, size_mat_rel, coord_mat_rel, sim_matrix_rel,
				N, M
			)

		# Hungarian Algorithm
		matches: List[MatchResult] = []
		row_ind, col_ind = linear_sum_assignment(1.0 - sim_matrix_rel)
		row_max_rel = sim_matrix_rel.max(axis=1) if N > 0 else np.array([])
		rel_keep = 0.85
		used_figma = set()
		used_web = set()

		for i, j in zip(row_ind, col_ind):
			score_rel = float(sim_matrix_rel[i, j])
			score_abs = float(sim_matrix_abs[i, j])

			# 임계값 필터링
			row_thr = float(max(min_similarity, row_max_rel[i] * rel_keep))
			if score_rel < row_thr or score_abs < min_similarity:
				continue

			# 텍스트 게이팅
			fig_txt = getattr(figma_elements_data[i].extracted, 'text', '') or ''
			web_txt = getattr(web_elements_data[j], 'text', '') or ''
			fig_has = len(fig_txt.strip()) > 0
			web_has = len(web_txt.strip()) > 0
			text_sim = float(sim_dict['text'][i, j])

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

		# Unmatched 요소들
		unmatched_figma_idxs = set(range(N)) - used_figma
		unmatched_web_idxs = set(range(M)) - used_web

		# IOU 기반 제외 처리
		for mr in matches:
			# Figma 중 IOU 높은 것 제거
			to_remove = {
				i for i in unmatched_figma_idxs
				if ElementExtractor.calculate_iou(
					mr.figma.extracted.box,
					figma_elements_data[i].extracted.box
				) > 0.5
			}
			unmatched_figma_idxs -= to_remove

			# Web 중 IOU 높은 것 제거
			to_remove = {
				j for j in unmatched_web_idxs
				if ElementExtractor.calculate_iou(
					mr.web.box,
					web_elements_data[j].box
				) > 0.5
			}
			unmatched_web_idxs -= to_remove

		# Unmatched MatchResult 생성
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

	def _save_debug_similarity_plots(
		self,
		text_abs, feat_abs, size_abs, coord_abs, combined_abs,
		text_rel, feat_rel, size_rel, coord_rel, combined_rel,
		N, M
	):
		"""디버그용 유사도 히트맵 저장"""
		try:
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
			save_heatmap(text_abs, 'Text (abs)', 'text_abs.png')
			save_heatmap(feat_abs, 'Feature (abs)', 'feature_abs.png')
			save_heatmap(size_abs, 'Size (abs)', 'size_abs.png')
			save_heatmap(coord_abs, 'Coord (abs)', 'coord_abs.png')
			save_heatmap(combined_abs, 'Combined (abs)', 'combined_abs.png')

			# Relative
			save_heatmap(text_rel, 'Text (rel)', 'text_rel.png')
			save_heatmap(feat_rel, 'Feature (rel)', 'feature_rel.png')
			save_heatmap(size_rel, 'Size (rel)', 'size_rel.png')
			save_heatmap(coord_rel, 'Coord (rel)', 'coord_rel.png')
			save_heatmap(combined_rel, 'Combined (rel)', 'combined_rel.png')

			# Top-K summary
			try:
				K = min(5, M)
				lines = []
				for i in range(N):
					order_abs = np.argsort(-combined_abs[i])[:K]
					order_rel = np.argsort(-combined_rel[i])[:K]
					lines.append(f"Figma {i} | ABS top{K}: " + ", ".join([f"j={int(j)}:{combined_abs[i,j]:.2f}" for j in order_abs]))
					lines.append(f"Figma {i} | REL top{K}: " + ", ".join([f"j={int(j)}:{combined_rel[i,j]:.2f}" for j in order_rel]))
				open(os.path.join(debug_dir, 'topk.txt'), 'w').write("\n".join(lines))
			except Exception:
				pass
		except Exception as e:
			print(f"Warning: Failed to save debug plots: {e}")
