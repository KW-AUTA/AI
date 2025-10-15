"""
Mapping Processor - 클래스 기반 매핑 처리

이 모듈은 mapping.py의 함수들을 클래스 기반으로 리팩토링한 버전입니다.
의존성 주입 패턴과 단일 책임 원칙을 적용했습니다.
"""

import logging
import time
import random
import numpy as np
import torch
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from PIL import Image

from ..core.models import ExtractedElement, FigmaFare, MatchResult
from ..core.element_matcher import ElementExtractor as LegacyElementExtractor
from ..utils.error_list import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass(frozen=True)
class MappingConfig:
    """매핑 처리 설정"""
    random_seed: int = 42
    iou_threshold: float = 0.5
    speed_mode: str = "balanced"  # "fast", "balanced", "accurate"
    target_height: int = 720
    enable_profiling: bool = False

    @classmethod
    def from_env(cls) -> 'MappingConfig':
        """환경변수에서 설정 로드"""
        import os
        return cls(
            random_seed=int(os.environ.get('RANDOM_SEED', '42')),
            iou_threshold=float(os.environ.get('IOU_THRESHOLD', '0.5')),
            speed_mode=os.environ.get('SPEED_MODE', 'balanced'),
            target_height=int(os.environ.get('TARGET_HEIGHT', '720')),
            enable_profiling=os.environ.get('ENABLE_PROFILING', 'false').lower() == 'true'
        )


class TimeManager:
    """시간 측정 및 성능 프로파일링 관리자"""

    def __init__(self, enable_logging: bool = True):
        """
        Args:
            enable_logging: 로깅 활성화 여부
        """
        self.enable_logging = enable_logging
        self.timings: Dict[str, float] = {}

    @contextmanager
    def time_check(self, name: str):
        """시간 측정 컨텍스트 매니저

        Args:
            name: 측정할 작업의 이름
        """
        start_time = time.time()
        yield
        end_time = time.time()
        elapsed = end_time - start_time
        self.timings[name] = elapsed

        if self.enable_logging:
            logging.info(f"{name}: {elapsed:.2f} seconds")

    def timecheck_decorator(self, func: Callable) -> Callable:
        """시간 측정 데코레이터

        Args:
            func: 측정할 함수

        Returns:
            래핑된 함수
        """
        def wrapper(*args, **kwargs):
            with self.time_check(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def get_timing(self, name: str) -> Optional[float]:
        """특정 작업의 측정 시간 반환

        Args:
            name: 작업 이름

        Returns:
            측정 시간 (초) 또는 None
        """
        return self.timings.get(name)

    def get_all_timings(self) -> Dict[str, float]:
        """모든 측정 시간 반환"""
        return self.timings.copy()

    def reset(self):
        """측정 데이터 초기화"""
        self.timings.clear()


class SeedManager:
    """랜덤 시드 관리자"""

    @staticmethod
    def seed_everything(seed: int = 42):
        """모든 랜덤 시드 설정

        Args:
            seed: 시드 값
        """
        random.seed(seed)
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info(f"Random seed set to {seed}")


class MatchCategorizer:
    """매칭 결과 카테고리 분류기"""

    @staticmethod
    def categorize_match(match: MatchResult) -> List[str]:
        """매칭된 요소들을 카테고리별로 분류

        Args:
            match: 매칭 결과

        Returns:
            카테고리 리스트
        """
        category = []

        # 좌표 오차 검사
        if abs(match.figma.extracted.box[0] - match.web.box[0]) > 10:
            category.append(G_ERROR_COORDINATE_X)
        if abs(match.figma.extracted.box[1] - match.web.box[1]) > 10:
            category.append(G_ERROR_COORDINATE_Y)

        # 크기 오차 검사
        w_f = match.figma.extracted.box[2] - match.figma.extracted.box[0]
        h_f = match.figma.extracted.box[3] - match.figma.extracted.box[1]
        w_w = match.web.box[2] - match.web.box[0]
        h_w = match.web.box[3] - match.web.box[1]

        if abs(w_f - w_w) > 10 or abs(h_f - h_w) > 10:
            category.append(G_ERROR_SIZE)

        # 정상인 경우
        if len(category) == 0:
            category.append(NORMAL)

        logging.debug(f"Categorized match: {category}")
        return category


class ElementExtractorHelper:
    """요소 추출 헬퍼 클래스"""

    def __init__(self, matcher: LegacyElementExtractor):
        """
        Args:
            matcher: ElementExtractor 인스턴스
        """
        self.matcher = matcher

    def extract_texts(
        self,
        img: Image.Image,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[str]:
        """기존 순차 처리 방식으로 텍스트 추출

        Args:
            img: 이미지
            boxes: 박스 리스트

        Returns:
            추출된 텍스트 리스트
        """
        texts = []
        for box in boxes:
            margin_box = (box[0], box[1], box[2], box[3])
            text = self.matcher.extract_text(img, margin_box)
            texts.append(text)
        return texts

    @staticmethod
    def is_window_empty(
        crop_img: Image.Image,
        threshold: float = 0.95
    ) -> bool:
        """윈도우가 대부분 비어있는지 확인 (흰색 배경 등)

        Args:
            crop_img: 크롭된 이미지
            threshold: 흰색 비율 임계값

        Returns:
            비어있으면 True
        """
        # 이미지를 grayscale로 변환
        gray = crop_img.convert('L')
        pixels = np.array(gray)
        # 흰색 픽셀 비율 계산 (240 이상을 흰색으로 간주)
        white_ratio = np.sum(pixels > 240) / pixels.size
        return white_ratio > threshold

    def extract_elements(
        self,
        img: Image.Image,
        start_x: int,
        windowing_height: int,
        iou_threshold: float = 0.5,
        speed_mode: str = "balanced"
    ) -> List[ExtractedElement]:
        """이미지에서 요소 추출 (최적화된 슬라이딩 윈도우 버전)

        Args:
            img: 입력 이미지
            start_x: 시작 x 좌표
            windowing_height: 윈도우 높이
            iou_threshold: IOU 임계값
            speed_mode: 속도 모드 ("fast", "balanced", "accurate")

        Returns:
            추출된 요소 리스트
        """
        logging.info(f"Extracting elements from image with height {img.height} using {speed_mode} mode...")

        # speed_mode에 따른 최적화 설정
        if speed_mode == "fast":
            if img.height > 2000:
                optimized_window_height = int(windowing_height * 2.0)
                overlap_ratio = 0.9
            else:
                optimized_window_height = int(windowing_height * 1.5)
                overlap_ratio = 0.85
            empty_threshold = 0.9
        elif speed_mode == "balanced":
            if img.height > 3000:
                optimized_window_height = int(windowing_height * 1.5)
                overlap_ratio = 0.85
            elif img.height > 2000:
                optimized_window_height = int(windowing_height * 1.2)
                overlap_ratio = 0.8
            else:
                optimized_window_height = windowing_height
                overlap_ratio = 0.75
            empty_threshold = 0.95
        else:  # "accurate"
            optimized_window_height = windowing_height
            overlap_ratio = 0.75
            empty_threshold = 0.98

        # 윈도우 처리
        windows_to_process = []
        current_height = 0
        start_x = 0
        skipped_windows = 0

        while current_height < img.height:
            crop_img = img.crop((
                start_x,
                current_height,
                img.width,
                min(current_height + optimized_window_height, img.height)
            ))

            # 빈 윈도우 체크
            if crop_img.height > 100 and self.is_window_empty(crop_img, empty_threshold):
                skipped_windows += 1
                current_height += optimized_window_height * overlap_ratio
                continue

            windows_to_process.append((crop_img, current_height))
            current_height += optimized_window_height * overlap_ratio

        if skipped_windows > 0:
            logging.info(f"Skipped {skipped_windows} empty windows for optimization")

        # 각 창에 대해 순차적으로 탐지 및 특징 추출
        all_boxes = []
        all_scores = []
        all_cls = []
        all_features = []

        logging.info(f"Processing {len(windows_to_process)} windows sequentially...")
        start_time = time.time()

        for crop_img, h in windows_to_process:
            boxes, scores, cls, feat_map, original_img_size = self.matcher.detect_boxes_yolo(
                crop_img,
                extract_features=True
            )

            if len(boxes) == 0:
                continue

            # 특징 추출
            features = self.matcher.extract_features_from_map(feat_map, boxes, original_img_size)

            # Y 좌표 조정
            boxes[:, 1] += h
            boxes[:, 3] += h

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_cls.append(cls)
            all_features.append(features)

        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds for extraction")

        if not all_boxes:
            logging.info("No elements were extracted.")
            return []

        # 모든 결과를 하나로 합침
        final_boxes = np.vstack(all_boxes)
        final_scores = np.concatenate(all_scores)
        final_cls = np.concatenate(all_cls)
        final_features = np.vstack(all_features)

        # ExtractedElement 객체 생성
        extracted_elements = []
        for i, box in enumerate(final_boxes):
            feature = final_features[i]
            cls = final_cls[i]
            extracted_elements.append(
                ExtractedElement(box=box, feature=feature, text=None, cls=cls)
            )

        logging.info(f"Extracted and filtered {len(extracted_elements)} elements.")
        return extracted_elements


class MappingProcessor:
    """
    매핑 처리 메인 클래스

    주요 기능:
    - 요소 추출 및 매칭
    - 카테고리 분류
    - 매핑 정보 생성
    """

    def __init__(
        self,
        config: Optional[MappingConfig] = None,
        time_manager: Optional[TimeManager] = None
    ):
        """
        Args:
            config: 매핑 설정
            time_manager: 시간 관리자
        """
        self.config = config or MappingConfig.from_env()
        self.time_manager = time_manager or TimeManager(enable_logging=True)
        self.categorizer = MatchCategorizer()

        # 랜덤 시드 설정
        SeedManager.seed_everything(self.config.random_seed)

    def create_extractor_helper(
        self,
        matcher: LegacyElementExtractor
    ) -> ElementExtractorHelper:
        """ElementExtractorHelper 인스턴스 생성

        Args:
            matcher: ElementExtractor 인스턴스

        Returns:
            ElementExtractorHelper 인스턴스
        """
        return ElementExtractorHelper(matcher)

    def get_mapping_info(self, matches: List[MatchResult]) -> List:
        """매칭 결과를 매핑 정보로 변환

        Args:
            matches: 매칭 결과 리스트

        Returns:
            매핑 정보 리스트
        """
        from routes.dto.response import BaseMappingInfo

        mapping_infos = []
        for match in matches:
            # 카테고리 분류
            match.errorCategories = self.categorizer.categorize_match(match)

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

    @classmethod
    def create(
        cls,
        config: Optional[MappingConfig] = None
    ) -> 'MappingProcessor':
        """팩토리 메서드: MappingProcessor 인스턴스 생성

        Args:
            config: 설정 객체

        Returns:
            MappingProcessor 인스턴스
        """
        return cls(config)


# 편의를 위한 전역 인스턴스
_default_processor = None


def get_default_processor() -> MappingProcessor:
    """기본 MappingProcessor 인스턴스 반환"""
    global _default_processor
    if _default_processor is None:
        _default_processor = MappingProcessor.create()
    return _default_processor
