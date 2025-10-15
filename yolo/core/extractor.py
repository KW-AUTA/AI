"""
개선된 ElementExtractor 클래스
- 의존성 주입 패턴 적용
- 설정 관리 개선
- 리소스 관리 개선
- 단일 책임 원칙 적용
"""

import os
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Tuple, Optional, Protocol
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO
import tesserocr

from .models import ExtractedElement, SimilarityConfig


@dataclass(frozen=True)
class ExtractorConfig:
    """요소 추출기 설정"""
    yolo_model_path: Optional[str] = None
    resize_size: Tuple[int, int] = (736, 736)
    debug_mode: bool = False
    
    # OCR 설정
    tesseract_data_dir: str = "/usr/local/share/tessdata"
    ocr_languages: str = "kor+eng"
    ocr_dpi: str = "300"
    
    # 이미지 전처리 설정
    scale_factors: Tuple[float, float, float, float] = (4.0, 3.0, 2.0, 1.5)
    scale_thresholds: Tuple[int, int, int] = (30, 60, 120)
    
    @classmethod
    def from_env(cls) -> 'ExtractorConfig':
        """환경변수에서 설정 로드"""
        def get_bool_env(key: str, default: bool) -> bool:
            return str(os.environ.get(key, str(default))).lower() in ('1', 'true', 'yes')
            
        def get_tuple_env(key: str, default: tuple, dtype=int) -> tuple:
            try:
                values = os.environ.get(key, "").split(",")
                return tuple(dtype(v.strip()) for v in values if v.strip())
            except:
                return default
        
        return cls(
            yolo_model_path=os.environ.get('YOLO_MODEL_PATH'),
            resize_size=get_tuple_env('YOLO_RESIZE_SIZE', (736, 736)),
            debug_mode=get_bool_env('EXTRACTOR_DEBUG', False),
            tesseract_data_dir=os.environ.get('TESSDATA_DIR', "/usr/local/share/tessdata"),
            ocr_languages=os.environ.get('OCR_LANGUAGES', "kor+eng"),
            ocr_dpi=os.environ.get('OCR_DPI', "300")
        )


class OCRProcessor(Protocol):
    """OCR 처리기 인터페이스"""
    def extract_text(self, image: Image.Image, bbox: np.ndarray) -> str: ...
    def cleanup(self) -> None: ...


class TesseractOCR:
    """Tesseract OCR 구현체"""
    
    def __init__(self, config: ExtractorConfig):
        self.config = config
        self._api = None
        
    @property
    def api(self) -> tesserocr.PyTessBaseAPI:
        """지연 로딩으로 OCR API 초기화"""
        if self._api is None:
            self._api = tesserocr.PyTessBaseAPI(
                path=self.config.tesseract_data_dir, 
                lang=self.config.ocr_languages
            )
            self._setup_api()
        return self._api
    
    def _setup_api(self) -> None:
        """OCR API 최적화 설정"""
        self.api.SetVariable("user_defined_dpi", self.config.ocr_dpi)
        self.api.SetVariable("tessedit_char_blacklist", "")
        self.api.SetVariable("preserve_interword_spaces", "1")
    
    def extract_text(self, image: Image.Image, bbox: np.ndarray) -> str:
        """이미지에서 텍스트 추출"""
        try:
            # ROI 추출
            x1, y1, x2, y2 = bbox.astype(int)
            roi = image.crop((x1, y1, x2, y2))
            
            # 전처리
            processed_roi = self._preprocess_roi(roi, x2-x1, y2-y1)
            
            # PSM 설정
            psm = self._pick_psm(x2-x1, y2-y1)
            self.api.SetPageSegMode(psm)
            
            # OCR 실행
            self.api.SetImage(processed_roi)
            text = self.api.GetUTF8Text().strip()
            
            return self._clean_text(text)
            
        except Exception as e:
            logging.warning(f"OCR extraction failed: {e}")
            return ""
    
    def _pick_psm(self, w: int, h: int) -> int:
        """ROI 크기/종횡비에 따라 최적화된 PSM 선택"""
        if w == 0 or h == 0:
            return tesserocr.PSM.SINGLE_BLOCK
            
        aspect = w / max(1, h)
        area = w * h
        
        if area < 400:  # 20x20
            return tesserocr.PSM.SINGLE_CHAR
        elif area < 2500:  # 50x50
            return tesserocr.PSM.SINGLE_WORD
        elif aspect >= 4.0 or (aspect <= 0.5 or aspect >= 2.0):
            return tesserocr.PSM.SINGLE_LINE
        else:
            return tesserocr.PSM.SINGLE_BLOCK
    
    def _preprocess_roi(self, pil_crop: Image.Image, w: int, h: int) -> Image.Image:
        """ROI 전처리"""
        arr = np.array(pil_crop)
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        
        # 동적 스케일링
        max_dim = max(h, w)
        thresholds = self.config.scale_thresholds
        scales = self.config.scale_factors
        
        if max_dim < thresholds[0]:
            scale = scales[0]
        elif max_dim < thresholds[1]:
            scale = scales[1]
        elif max_dim < thresholds[2]:
            scale = scales[2]
        else:
            scale = scales[3]
        
        # 스케일링
        if scale > 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 노이즈 제거
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # 적응적 임계값
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 모폴로지 연산
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 여백 추가
        binary = cv2.copyMakeBorder(binary, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
        
        return Image.fromarray(binary)
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        import unicodedata
        import re
        
        # 유니코드 정규화
        text = unicodedata.normalize('NFKC', text)
        
        # 불필요한 문자 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def cleanup(self) -> None:
        """리소스 정리"""
        if self._api is not None:
            self._api.End()
            self._api = None


class YOLODetector:
    """YOLO 객체 검출기"""
    
    def __init__(self, config: ExtractorConfig):
        self.config = config
        self._model = None
        self._transform = None
    
    @property
    def model(self) -> YOLO:
        """지연 로딩으로 YOLO 모델 초기화"""
        if self._model is None:
            model_path = self._get_model_path()
            self._model = YOLO(model_path, task='detect', verbose=False)
        return self._model
    
    @property
    def transform(self) -> T.Compose:
        """이미지 변환 파이프라인"""
        if self._transform is None:
            self._transform = T.Compose([
                T.Resize(self.config.resize_size),
                T.ToTensor(),
            ])
        return self._transform
    
    def _get_model_path(self) -> str:
        """모델 파일 경로 결정"""
        if self.config.yolo_model_path:
            return self.config.yolo_model_path
        
        # 기본 모델 경로
        current_dir = Path(__file__).parent
        default_path = current_dir.parent / "models_weights" / "best.pt"
        
        if not default_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {default_path}")
        
        return str(default_path)
    
    def detect_elements(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """이미지에서 요소 검출
        
        Returns:
            boxes: (N, 4) 바운딩 박스 [x1, y1, x2, y2]
            features: (N, D) 특징 벡터
            classes: (N,) 클래스 ID
        """
        # YOLO 추론
        results = self.model(image)
        
        if not results or len(results) == 0:
            return np.array([]), np.array([]), np.array([])
        
        result = results[0]
        
        # 바운딩 박스 추출
        if result.boxes is None:
            return np.array([]), np.array([]), np.array([])
        
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # 특징 벡터 추출 (YOLO 백본에서)
        features = self._extract_features(image, boxes)
        
        return boxes, features, classes
    
    def _extract_features(self, image: Image.Image, boxes: np.ndarray) -> np.ndarray:
        """바운딩 박스에서 특징 벡터 추출"""
        if len(boxes) == 0:
            return np.array([])
        
        features = []
        image_np = np.array(image)
        
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            
            # ROI 추출
            roi = image_np[y1:y2, x1:x2]
            
            if roi.size == 0:
                features.append(np.zeros(512))  # 기본 특징 크기
                continue
            
            # 간단한 특징 추출 (히스토그램 기반)
            feature = self._compute_histogram_features(roi)
            features.append(feature)
        
        return np.array(features)
    
    def _compute_histogram_features(self, roi: np.ndarray) -> np.ndarray:
        """ROI에서 히스토그램 기반 특징 추출"""
        if len(roi.shape) == 3:
            # RGB 히스토그램
            hist_r = cv2.calcHist([roi], [0], None, [64], [0, 256])
            hist_g = cv2.calcHist([roi], [1], None, [64], [0, 256])
            hist_b = cv2.calcHist([roi], [2], None, [64], [0, 256])
            feature = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        else:
            # 그레이스케일 히스토그램
            hist = cv2.calcHist([roi], [0], None, [192], [0, 256])
            feature = hist.flatten()
        
        # 정규화
        feature = feature / (feature.sum() + 1e-8)
        
        # 고정 크기로 패딩 또는 잘라내기
        if len(feature) > 512:
            feature = feature[:512]
        elif len(feature) < 512:
            feature = np.pad(feature, (0, 512 - len(feature)))
        
        return feature


class ElementExtractor:
    """개선된 요소 추출기 - 의존성 주입 및 설정 관리"""
    
    def __init__(
        self, 
        config: Optional[ExtractorConfig] = None,
        detector: Optional[YOLODetector] = None,
        ocr_processor: Optional[OCRProcessor] = None
    ):
        self.config = config or ExtractorConfig.from_env()
        self.detector = detector or YOLODetector(self.config)
        self.ocr_processor = ocr_processor or TesseractOCR(self.config)
        
        self.logger = logging.getLogger(__name__)
    
    def extract_elements(
        self, 
        image: Image.Image, 
        include_ocr: bool = True
    ) -> List[ExtractedElement]:
        """이미지에서 요소 추출"""
        try:
            # YOLO 검출
            boxes, features, classes = self.detector.detect_elements(image)
            
            if len(boxes) == 0:
                return []
            
            elements = []
            
            for i, (box, feature, cls) in enumerate(zip(boxes, features, classes)):
                # OCR 텍스트 추출
                text = ""
                if include_ocr:
                    text = self.ocr_processor.extract_text(image, box)
                
                # ExtractedElement 생성
                element = ExtractedElement(
                    box=box,
                    feature=torch.from_numpy(feature),
                    text=text,
                    cls=int(cls)
                )
                elements.append(element)
            
            self.logger.info(f"Extracted {len(elements)} elements")
            return elements
            
        except Exception as e:
            self.logger.error(f"Element extraction failed: {e}")
            return []
    
    def __enter__(self) -> 'ElementExtractor':
        """컨텍스트 매니저 지원"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """리소스 정리"""
        self.cleanup()
    
    def cleanup(self) -> None:
        """리소스 정리"""
        if hasattr(self.ocr_processor, 'cleanup'):
            self.ocr_processor.cleanup()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 팩토리 함수
def create_extractor(
    config: Optional[ExtractorConfig] = None,
    use_mock_ocr: bool = False
) -> ElementExtractor:
    """ExtractorConfig를 기반으로 ElementExtractor 생성"""
    config = config or ExtractorConfig.from_env()
    
    # Mock OCR for testing
    if use_mock_ocr:
        class MockOCR:
            def extract_text(self, image: Image.Image, bbox: np.ndarray) -> str:
                return "mock_text"
            def cleanup(self) -> None:
                pass
        
        return ElementExtractor(config=config, ocr_processor=MockOCR())
    
    return ElementExtractor(config=config)