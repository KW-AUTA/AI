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
		# PaddlePaddle 로드 성공 (로깅은 나중에 logger로)
		pass
	except Exception as e:
		_PADDLE_IMPORTED = False
		# PaddlePaddle 로드 실패 (로깅은 나중에 logger로)
		pass

# PyTorch는 PaddlePaddle 이후에 import
import torch
import torch.nn.functional as F
import torchvision.ops
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageEnhance
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
import tesserocr

from .models import FigmaFare, ExtractedElement, MatchResult
from .paddle_ocr_helper import PaddleOCRHelper
from ..utils.errorChecker import ErrorChecker
from ..utils.error_list import *

# Logger 설정 (색상 + 박스 + 모듈명)
import logging
import sys

class ColoredFormatter(logging.Formatter):
	"""색상과 박스 문자를 사용한 로그 포매터"""

	# ANSI 색상 코드
	COLORS = {
		'DEBUG': '\033[36m',    # Cyan
		'INFO': '\033[32m',     # Green
		'WARNING': '\033[33m',  # Yellow
		'ERROR': '\033[31m',    # Red
		'CRITICAL': '\033[35m', # Magenta
	}
	RESET = '\033[0m'
	BOLD = '\033[1m'

	# 박스 문자
	ICONS = {
		'DEBUG': '🔍',
		'INFO': '✓',
		'WARNING': '⚠',
		'ERROR': '✗',
		'CRITICAL': '🔥',
	}

	def format(self, record):
		levelname = record.levelname
		color = self.COLORS.get(levelname, '')
		icon = self.ICONS.get(levelname, '•')
		module = record.name.split('.')[-1]  # 모듈명만 (예: element_matcher)

		# 형식: [아이콘 레벨] 모듈 | 메시지
		log_fmt = f"{color}{self.BOLD}[{icon} {levelname:8s}]{self.RESET} {color}{module:20s}{self.RESET} │ {record.getMessage()}"
		return log_fmt

logger = logging.getLogger(__name__)
if not logger.handlers:
	handler = logging.StreamHandler(sys.stdout)
	handler.setFormatter(ColoredFormatter())
	logger.addHandler(handler)
	logger.setLevel(logging.INFO)
	logger.propagate = False  # 중복 방지

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

		logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
		logger.info("ElementMatcher initialized")
		logger.info(f"yolo_model_path: {yolo_model_path}")
		logger.info(f"resize_size: {resize_size}")
		logger.info(f"debug_similarity: {debug_similarity}")
		logger.info(f"use_paddleocr: {use_paddleocr}")
		logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
		
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
				logger.info("PaddleOCR Helper initialized (subprocess)")
			except Exception as e:
				logger.warning(f"PaddleOCR Helper init failed, fallback to Tesseract: {e}")
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
			logger.warning(f"Failed to set OCR parameters: {e}")

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

	# 클래스 레벨에서 실행 ID 관리
	_debug_run_id = None

	@classmethod
	def _get_or_create_run_id(cls):
		"""실행 ID를 가져오거나 생성 (동일 실행 내에서 공유)"""
		if cls._debug_run_id is None:
			import time
			cls._debug_run_id = time.strftime("run_%Y%m%d_%H%M%S")
		return cls._debug_run_id

	def _apply_preprocessing(
		self,
		pil_img: Image.Image,
		mode: str,
		save_debug: bool = False,
		debug_dir: str = None
	) -> Image.Image:
		"""
		전처리 적용

		Args:
			pil_img: 입력 이미지
			mode: 전처리 모드
				- "default": 기본 샤프닝만
				- "clahe": CLAHE 대비 향상
				- "bilateral": Bilateral 필터 (엣지 보존 노이즈 제거)
				- "clahe_bilateral": CLAHE + Bilateral 조합
				- "gamma": Gamma 보정
				- "adaptive": Adaptive Histogram Equalization
				- "all": 모든 방법 적용

		Returns:
			전처리된 이미지
		"""
		# 전처리 시작 로그
		logger.info(f"Applying preprocessing mode: '{mode}'")

		if save_debug and debug_dir:
			with open(os.path.join(debug_dir, 'preprocess.log'), 'a') as f:
				f.write(f"Applying preprocessing mode: '{mode}'\n")

		img_np = np.array(pil_img)
		step = 2

		if mode == "default":
			# 기본: 샤프닝만
			result = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1))
			if save_debug and debug_dir:
				result.save(os.path.join(debug_dir, f'{step}_sharpened.png'))
				logger.debug(f"Saved: {step}_sharpened.png")
			return result

		elif mode == "clahe":
			# CLAHE (대비 향상)
			if len(img_np.shape) == 3:
				gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
			else:
				gray = img_np

			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			enhanced = clahe.apply(gray)

			if save_debug and debug_dir:
				Image.fromarray(enhanced).save(os.path.join(debug_dir, f'{step}_clahe.png'))
				logger.debug(f"Saved: {step}_clahe.png")

			# 다시 RGB로 변환
			if len(img_np.shape) == 3:
				enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
			else:
				enhanced_rgb = enhanced

			# 샤프닝 추가
			result = Image.fromarray(enhanced_rgb).filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))
			if save_debug and debug_dir:
				result.save(os.path.join(debug_dir, f'{step+1}_clahe_sharpened.png'))
				logger.debug(f"Saved: {step+1}_clahe_sharpened.png")

			return result

		elif mode == "bilateral":
			# Bilateral 필터 (엣지 보존)
			bilateral = cv2.bilateralFilter(img_np, 9, 75, 75)

			if save_debug and debug_dir:
				Image.fromarray(bilateral).save(os.path.join(debug_dir, f'{step}_bilateral.png'))
				logger.debug(f"Saved: {step}_bilateral.png")

			# 샤프닝 추가
			result = Image.fromarray(bilateral).filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))
			if save_debug and debug_dir:
				result.save(os.path.join(debug_dir, f'{step+1}_bilateral_sharpened.png'))
				logger.debug(f"Saved: {step+1}_bilateral_sharpened.png")

			return result

		elif mode == "clahe_bilateral":
			# CLAHE + Bilateral 조합 (최고 품질)
			if len(img_np.shape) == 3:
				gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
			else:
				gray = img_np

			# 1. CLAHE
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			enhanced = clahe.apply(gray)

			if save_debug and debug_dir:
				Image.fromarray(enhanced).save(os.path.join(debug_dir, f'{step}_clahe.png'))
				logger.debug(f"Saved: {step}_clahe.png")
				step += 1

			# 2. Bilateral
			if len(img_np.shape) == 3:
				enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
			else:
				enhanced_rgb = enhanced

			bilateral = cv2.bilateralFilter(enhanced_rgb, 9, 75, 75)

			if save_debug and debug_dir:
				Image.fromarray(bilateral).save(os.path.join(debug_dir, f'{step}_bilateral.png'))
				logger.debug(f"Saved: {step}_bilateral.png")
				step += 1

			# 3. 샤프닝
			result = Image.fromarray(bilateral).filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))
			if save_debug and debug_dir:
				result.save(os.path.join(debug_dir, f'{step}_sharpened.png'))
				logger.debug(f"Saved: {step}_sharpened.png")

			return result

		elif mode == "gamma":
			# Gamma 보정
			gamma = 1.2
			inv_gamma = 1.0 / gamma
			table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
			gamma_corrected = cv2.LUT(img_np, table)

			if save_debug and debug_dir:
				Image.fromarray(gamma_corrected).save(os.path.join(debug_dir, f'{step}_gamma.png'))
				logger.debug(f"Saved: {step}_gamma.png")
				step += 1

			# 샤프닝 추가
			result = Image.fromarray(gamma_corrected).filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))
			if save_debug and debug_dir:
				result.save(os.path.join(debug_dir, f'{step}_gamma_sharpened.png'))
				logger.debug(f"Saved: {step}_gamma_sharpened.png")

			return result

		elif mode == "adaptive":
			# Adaptive Histogram Equalization
			if len(img_np.shape) == 3:
				# YUV 변환 후 Y 채널에만 적용
				yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
				yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
				enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
			else:
				enhanced = cv2.equalizeHist(img_np)

			if save_debug and debug_dir:
				Image.fromarray(enhanced).save(os.path.join(debug_dir, f'{step}_adaptive.png'))
				logger.debug(f"Saved: {step}_adaptive.png")
				step += 1

			# 샤프닝 추가
			result = Image.fromarray(enhanced).filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))
			if save_debug and debug_dir:
				result.save(os.path.join(debug_dir, f'{step}_adaptive_sharpened.png'))
				logger.debug(f"Saved: {step}_adaptive_sharpened.png")

			return result

		elif mode == "all":
			# 모든 전처리 방법 적용 (디버그용)
			logger.info("Applying all preprocessing methods for comparison")

			if len(img_np.shape) == 3:
				gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
			else:
				gray = img_np

			# 1. Default (샤프닝만)
			default_result = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1))
			if save_debug and debug_dir:
				default_result.save(os.path.join(debug_dir, f'{step}_default_sharpened.png'))
				logger.debug(f"Saved: {step}_default_sharpened.png")
				step += 1

			# 2. CLAHE
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			clahe_result = clahe.apply(gray)
			if save_debug and debug_dir:
				Image.fromarray(clahe_result).save(os.path.join(debug_dir, f'{step}_clahe.png'))
				logger.debug(f"Saved: {step}_clahe.png")
				step += 1

			# 3. Bilateral
			bilateral_result = cv2.bilateralFilter(img_np, 9, 75, 75)
			if save_debug and debug_dir:
				Image.fromarray(bilateral_result).save(os.path.join(debug_dir, f'{step}_bilateral.png'))
				logger.debug(f"Saved: {step}_bilateral.png")
				step += 1

			# 4. CLAHE + Bilateral (최종 사용)
			clahe_rgb = cv2.cvtColor(clahe_result, cv2.COLOR_GRAY2RGB) if len(img_np.shape) == 3 else clahe_result
			clahe_bilateral = cv2.bilateralFilter(clahe_rgb, 9, 75, 75)
			if save_debug and debug_dir:
				Image.fromarray(clahe_bilateral).save(os.path.join(debug_dir, f'{step}_clahe_bilateral.png'))
				logger.debug(f"Saved: {step}_clahe_bilateral.png")
				step += 1

			# 5. Gamma
			gamma = 1.2
			inv_gamma = 1.0 / gamma
			table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
			gamma_result = cv2.LUT(img_np, table)
			if save_debug and debug_dir:
				Image.fromarray(gamma_result).save(os.path.join(debug_dir, f'{step}_gamma.png'))
				logger.debug(f"Saved: {step}_gamma.png")
				step += 1

			# 6. Adaptive
			if len(img_np.shape) == 3:
				yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
				yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
				adaptive_result = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
			else:
				adaptive_result = cv2.equalizeHist(img_np)
			if save_debug and debug_dir:
				Image.fromarray(adaptive_result).save(os.path.join(debug_dir, f'{step}_adaptive.png'))
				logger.debug(f"Saved: {step}_adaptive.png")

			logger.info("All preprocessing methods saved")
			logger.info("Using Adaptive for detection")

			# 최종: Adaptive 사용 (RGB 버전)
			return Image.fromarray(adaptive_result)

		else:
			# 알 수 없는 모드: 기본값 사용
			logger.warning(f"Unknown preprocess mode '{mode}', using default")
			return pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1))

	def detect_boxes_yolo(
		self,
		pil_img: Image.Image,
		conf_thresh: float = 0.05,
		max_det: int = 1000,
		extract_features: bool = False,
		save_preprocessing: bool = None,
		save_path: str = None,
		window_id: str = None,
		image_type: str = None
	):
		"""YOLO를 사용한 박스 검출

		Args:
			pil_img: 입력 이미지
			conf_thresh: Confidence threshold
			max_det: 최대 검출 개수
			extract_features: 특징 추출 여부
			save_preprocessing: 전처리 과정 저장 여부 (None이면 환경변수 확인)
			save_path: 저장 경로 (None이면 자동 생성)
			window_id: 윈도우 식별자 (예: "window_1_large", "window_2_small")
			image_type: 이미지 타입 (예: "figma", "web")
		"""
		orig_w, orig_h = pil_img.size
		resize_size = self.resize_size

		# 환경변수로 디버그 모드 확인
		if save_preprocessing is None:
			save_preprocessing = os.environ.get('DEBUG_PREPROCESSING', 'false').lower() in ('true', '1', 'yes')

		# 전처리 과정 저장을 위한 디버그 모드
		if save_preprocessing:
			# 실행 ID로 묶인 폴더 구조: debug_preprocessing/run_20250101_120000/figma_window_001_h0000/
			run_id = self._get_or_create_run_id()
			base_debug_dir = save_path or os.path.join(os.path.dirname(__file__), 'debug_preprocessing', run_id)

			# 윈도우별 폴더 생성 (이미지 타입 포함)
			if window_id:
				if image_type:
					debug_dir = os.path.join(base_debug_dir, f"{image_type}_{window_id}")
				else:
					debug_dir = os.path.join(base_debug_dir, window_id)
			else:
				import time
				timestamp = time.strftime("%H%M%S_%f")
				type_prefix = f"{image_type}_" if image_type else ""
				debug_dir = os.path.join(base_debug_dir, f"{type_prefix}window_{timestamp}")

			os.makedirs(debug_dir, exist_ok=True)

			# 원본 저장
			pil_img.save(os.path.join(debug_dir, '1_original.png'))
			logger.debug(f"Saved original image to {debug_dir}/1_original.png")

		# 전처리 모드 확인 (환경변수)
		preprocess_mode = os.environ.get('PREPROCESS_MODE', 'default').lower()

		# 로깅 (깔끔하게 정리)
		import sys

		# 로그 메시지 구성
		log_lines = [
			"",
			"╔" + "═" * 78 + "╗",
			"║ 🔍 YOLO DETECTION START" + " " * 53 + "║",
			"╠" + "═" * 78 + "╣",
			f"║ 📐 Image Size   : {orig_w:4d} x {orig_h:4d}" + " " * (78 - len(f"║ 📐 Image Size   : {orig_w:4d} x {orig_h:4d}")) + "║",
			f"║ 🎨 Preprocess   : {preprocess_mode:<20}" + " " * (78 - len(f"║ 🎨 Preprocess   : {preprocess_mode:<20}")) + "║",
			f"║ 🐛 Debug Mode   : {str(save_preprocessing):<20}" + " " * (78 - len(f"║ 🐛 Debug Mode   : {str(save_preprocessing):<20}")) + "║",
		]

		if window_id:
			log_lines.append(f"║ 🪟 Window ID    : {window_id:<20}" + " " * (78 - len(f"║ 🪟 Window ID    : {window_id:<20}")) + "║")
		if image_type:
			log_lines.append(f"║ 🏷️  Image Type   : {image_type:<20}" + " " * (78 - len(f"║ 🏷️  Image Type   : {image_type:<20}")) + "║")

		log_lines.append("╚" + "═" * 78 + "╝")

		# 출력
		log_text = "\n".join(log_lines)
		sys.stdout.write(log_text + "\n")
		sys.stdout.flush()
		logger.info(f"YOLO Detection - Size:{orig_w}x{orig_h}, Mode:{preprocess_mode}, Type:{image_type or 'N/A'}")

		# 파일로도 저장 (디버그용)
		if save_preprocessing and debug_dir:
			log_file = os.path.join(debug_dir, 'preprocess.log')
			with open(log_file, 'a') as f:
				f.write(log_text + "\n")

		pil_img_processed = self._apply_preprocessing(
			pil_img,
			preprocess_mode,
			save_preprocessing,
			debug_dir if save_preprocessing else None
		)

		# 완료 로그
		logger.info(f"Preprocessing completed: {preprocess_mode}")

		if save_preprocessing and debug_dir:
			with open(os.path.join(debug_dir, 'preprocess.log'), 'a') as f:
				f.write(f"Preprocessing completed: {preprocess_mode}\n")

		# 리사이즈
		img_resized = pil_img_processed.resize(resize_size, Image.Resampling.LANCZOS)

		if save_preprocessing:
			img_resized.save(os.path.join(debug_dir, '9_final_resized.png'))
			logger.debug(f"Saved final resized image to {debug_dir}/9_final_resized.png")

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
		config: Optional['SimilarityConfig'] = None,
		iou_threshold: float = 0.5
	) -> Tuple[List[MatchResult], List[MatchResult], List[MatchResult]]:
		"""
		Performs element matching between Figma and Web elements using similarity scores.

		Args:
			sim_dict: Dictionary containing similarity matrices for different features
			figma_elements_data: List of Figma elements
			web_elements_data: List of Web elements
			min_similarity: Minimum similarity threshold for matching (default: 0.8)
			config: Optional similarity configuration (currently unused)
			iou_threshold: IOU threshold for filtering overlapping unmatched elements (default: 0.5)

		Returns:
			Tuple of (matched, unmatched_figma, unmatched_web) MatchResult lists
		"""
		# 설정 초기화
		if min_similarity is None:
			# 환경변수로 threshold 조정 가능
			import os
			min_similarity = float(os.environ.get('MATCH_THRESHOLD', '0.45'))
			logger.info(f"Using min_similarity threshold: {min_similarity} (set MATCH_THRESHOLD env to override)")

		# Absolute views (단순화)
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

		# 1. Calculate weighted similarity matrix
		sim_matrix = (
			text_mat_abs * 0.35 +
			feat_mat_abs * 0.35 +
			size_mat_abs * 0.15 +
			coord_mat_abs * 0.15
		)

		# 1.5. Pre-filter: Apply masks for invalid matching candidates
		# This prevents invalid matches from being considered in the first place

		# Check which elements have text (vectorized, with type safety)
		def has_text_safe(text):
			if isinstance(text, bool):
				return False
			if text is None:
				return False
			return bool(str(text).strip())

		figma_has_text = np.array([has_text_safe(f.extracted.text) for f in figma_elements_data])
		web_has_text = np.array([has_text_safe(w.text) for w in web_elements_data])

		# Create mask: True where text existence matches
		# Broadcasting: (N, 1) == (1, M) -> (N, M)
		text_match_mask = figma_has_text[:, np.newaxis] == web_has_text[np.newaxis, :]

		# Apply mask: set mismatched pairs to 0
		filtered_count = np.sum(~text_match_mask)
		sim_matrix[~text_match_mask] = 0.0

		if filtered_count > 0:
			logger.info(f"Pre-filtered {filtered_count} invalid candidates (text/non-text mismatch)")
			logger.debug(f"  Figma with text: {np.sum(figma_has_text)}/{N}, Web with text: {np.sum(web_has_text)}/{M}")

		# 2. Greedy matching for high-confidence pairs
		matches: List[MatchResult] = []
		unmatched_figma_idxs = set(range(N))
		unmatched_web_idxs = set(range(M))
		temp_matrix = sim_matrix.copy()

		# Debug: Log similarity matrix statistics
		logger.info(f"Similarity matrix stats - Max: {sim_matrix.max():.3f}, Mean: {sim_matrix.mean():.3f}, Min: {sim_matrix.min():.3f}")
		logger.info(f"Matching threshold: {min_similarity}")
		logger.info(f"Scores above threshold: {np.sum(sim_matrix >= min_similarity)}")

		while True:
			# Find the highest similarity pair
			i, j = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
			score = temp_matrix[i, j]

			# Stop if similarity is below threshold or filtered out (0.0)
			if score <= 0.0:
				logger.info(f"Stopped matching - No more valid candidates (best score: {score:.3f})")
				break
			if score < min_similarity:
				logger.info(f"Stopped matching - Best remaining score {score:.3f} < threshold {min_similarity}")
				break

			logger.info(f"Matched - Figma[{i}]: '{figma_elements_data[i].extracted.text}' <-> Web[{j}]: '{web_elements_data[j].text}' | Score: {score:.3f} (text: {sim_dict['text'][i,j]:.2f}, feat: {sim_dict['feature'][i,j]:.2f}, size: {sim_dict['size'][i,j]:.2f}, coord: {sim_dict['coordinate'][i,j]:.2f})")

			# Create match result
			mr = MatchResult(
				figma=figma_elements_data[i],
				web=web_elements_data[j],
				feature_similarity=float(sim_dict['feature'][i, j]),
				text_similarity=float(sim_dict['text'][i, j]),
				size_similarity=float(sim_dict['size'][i, j]),
				coordinate_similarity=float(sim_dict['coordinate'][i, j]),
				score=float(score),
				errorCategories=ErrorChecker().check_error(
					figma_elements_data[i].extracted.box,
					web_elements_data[j].box,
					figma_elements_data[i].extracted.text,
					web_elements_data[j].text
				)
			)
			matches.append(mr)

			# Remove matched indices
			unmatched_figma_idxs.discard(i)
			unmatched_web_idxs.discard(j)

			# Mark as matched in temp matrix
			temp_matrix[i, :] = -np.inf
			temp_matrix[:, j] = -np.inf

		logger.info(f"Initial matching complete - Matched: {len(matches)}, Unmatched Figma: {len(unmatched_figma_idxs)}, Unmatched Web: {len(unmatched_web_idxs)}")
		logger.debug(f"Unmatched Figma indices: {unmatched_figma_idxs}")
		logger.debug(f"Unmatched Web indices: {unmatched_web_idxs}")

		# 3. Apply NMS: Remove unmatched elements with high IOU overlap with matched elements
		# This filters out elements that are likely duplicates or part of matched elements
		removed_figma_count = 0
		removed_web_count = 0

		for mr in matches:
			# Remove figma elements with high IOU overlap
			to_remove_figma = {
				i for i in unmatched_figma_idxs
				if ElementExtractor.calculate_iou(
					mr.figma.extracted.box,
					figma_elements_data[i].extracted.box
				) > iou_threshold
			}
			removed_figma_count += len(to_remove_figma)
			unmatched_figma_idxs -= to_remove_figma

			# Remove web elements with high IOU overlap
			to_remove_web = {
				j for j in unmatched_web_idxs
				if ElementExtractor.calculate_iou(
					mr.web.box,
					web_elements_data[j].box
				) > iou_threshold
			}
			removed_web_count += len(to_remove_web)
			unmatched_web_idxs -= to_remove_web

		if removed_figma_count > 0 or removed_web_count > 0:
			logger.info(f"NMS filtering - Removed {removed_figma_count} Figma and {removed_web_count} Web overlapping elements")

		# 4. Create MatchResult objects for unmatched elements
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

		# Final summary
		total_figma = len(figma_elements_data)
		total_web = len(web_elements_data)
		match_rate_figma = len(matches) / total_figma * 100 if total_figma > 0 else 0
		match_rate_web = len(matches) / total_web * 100 if total_web > 0 else 0

		logger.info("=" * 80)
		logger.info(f"MATCHING SUMMARY")
		logger.info(f"  Total Figma elements: {total_figma}")
		logger.info(f"  Total Web elements: {total_web}")
		logger.info(f"  Matched pairs: {len(matches)} ({match_rate_figma:.1f}% of Figma, {match_rate_web:.1f}% of Web)")
		logger.info(f"  Unmatched Figma: {len(unmatched_figma)}")
		logger.info(f"  Unmatched Web: {len(unmatched_web)}")
		logger.info("=" * 80)

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
			logger.warning(f"Failed to save debug plots: {e}")
