"""
UIMatchingPipeline - 전체 UI 매칭 프로세스 통합 관리
- 파사드 패턴으로 복잡한 프로세스 단순화
- 의존성 주입을 통한 유연한 구성
- 단계별 결과 추적 및 로깅
- 에러 처리 및 복구 메커니즘
"""

import time
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from enum import Enum

import numpy as np
from PIL import Image

from .extractor import ElementExtractor, ExtractorConfig, create_extractor
from .matcher import SimilarityMatcher, create_similarity_matcher
from .models import ExtractedElement, FigmaFare, MatchResult, SimilarityConfig
from ..figma.figma import FigmaDataLoader
from ..web.web_navigator import WebNavigator
from ..visualization.visualizer import Visualizer
from ..utils.errorChecker import ErrorChecker


class ProcessingStage(Enum):
    """처리 단계"""
    INITIALIZATION = "initialization"
    FIGMA_EXTRACTION = "figma_extraction"
    WEB_EXTRACTION = "web_extraction"
    SIMILARITY_CALCULATION = "similarity_calculation"
    MATCHING = "matching"
    RESULT_PROCESSING = "result_processing"
    VISUALIZATION = "visualization"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    matches: List[MatchResult] = field(default_factory=list)
    unmatched_figma: List[MatchResult] = field(default_factory=list)
    unmatched_web: List[MatchResult] = field(default_factory=list)
    
    figma_elements: List[FigmaFare] = field(default_factory=list)
    web_elements: List[ExtractedElement] = field(default_factory=list)
    
    processing_time: Dict[str, float] = field(default_factory=dict)
    stage: ProcessingStage = ProcessingStage.INITIALIZATION
    
    error_message: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_time(self) -> float:
        """총 처리 시간"""
        return sum(self.processing_time.values())
    
    @property
    def success_rate(self) -> float:
        """매칭 성공률"""
        total_figma = len(self.figma_elements)
        if total_figma == 0:
            return 0.0
        return len(self.matches) / total_figma
    
    def summary(self) -> Dict[str, Any]:
        """결과 요약"""
        return {
            'stage': self.stage.value,
            'total_figma_elements': len(self.figma_elements),
            'total_web_elements': len(self.web_elements),
            'matched_pairs': len(self.matches),
            'unmatched_figma': len(self.unmatched_figma),
            'unmatched_web': len(self.unmatched_web),
            'success_rate': self.success_rate,
            'total_processing_time': self.total_time,
            'error': self.error_message
        }


@dataclass(frozen=True)
class PipelineConfig:
    """파이프라인 설정"""
    extractor_config: ExtractorConfig = field(default_factory=ExtractorConfig)
    similarity_config: SimilarityConfig = field(default_factory=SimilarityConfig)
    
    # 처리 옵션
    include_ocr: bool = True
    min_similarity_threshold: float = 0.7
    enable_visualization: bool = False
    enable_error_checking: bool = True
    
    # 성능 옵션
    max_elements_per_stage: int = 1000
    timeout_seconds: int = 300
    
    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        """환경변수에서 설정 로드"""
        def get_bool_env(key: str, default: bool) -> bool:
            import os
            return str(os.environ.get(key, str(default))).lower() in ('1', 'true', 'yes')
        
        def get_float_env(key: str, default: float) -> float:
            import os
            try:
                return float(os.environ.get(key, str(default)))
            except (ValueError, TypeError):
                return default
        
        def get_int_env(key: str, default: int) -> int:
            import os
            try:
                return int(os.environ.get(key, str(default)))
            except (ValueError, TypeError):
                return default
        
        return cls(
            extractor_config=ExtractorConfig.from_env(),
            similarity_config=SimilarityConfig.from_env(),
            include_ocr=get_bool_env('PIPELINE_INCLUDE_OCR', True),
            min_similarity_threshold=get_float_env('PIPELINE_MIN_SIMILARITY', 0.7),
            enable_visualization=get_bool_env('PIPELINE_ENABLE_VIZ', False),
            enable_error_checking=get_bool_env('PIPELINE_ERROR_CHECK', True),
            max_elements_per_stage=get_int_env('PIPELINE_MAX_ELEMENTS', 1000),
            timeout_seconds=get_int_env('PIPELINE_TIMEOUT', 300)
        )


class UIMatchingPipeline:
    """UI 매칭 파이프라인 - 전체 프로세스 통합 관리"""
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        extractor: Optional[ElementExtractor] = None,
        matcher: Optional[SimilarityMatcher] = None,
        visualizer: Optional[Visualizer] = None,
        error_checker: Optional[ErrorChecker] = None
    ):
        self.config = config or PipelineConfig.from_env()
        
        # 의존성 주입
        self.extractor = extractor or create_extractor(self.config.extractor_config)
        self.matcher = matcher or create_similarity_matcher(self.config.similarity_config)
        self.visualizer = visualizer or Visualizer() if self.config.enable_visualization else None
        self.error_checker = error_checker or ErrorChecker() if self.config.enable_error_checking else None
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 상태 추적
        self._current_result = PipelineResult()
    
    @contextmanager
    def _track_stage(self, stage: ProcessingStage):
        """처리 단계 추적"""
        start_time = time.time()
        self._current_result.stage = stage
        
        try:
            self.logger.info(f"Starting stage: {stage.value}")
            yield
            
        except Exception as e:
            self.logger.error(f"Stage {stage.value} failed: {e}")
            self._current_result.stage = ProcessingStage.FAILED
            self._current_result.error_message = str(e)
            raise
            
        finally:
            elapsed = time.time() - start_time
            self._current_result.processing_time[stage.value] = elapsed
            self.logger.info(f"Stage {stage.value} completed in {elapsed:.2f}s")
    
    def process_figma_and_web(
        self,
        figma_data_or_path: Union[str, dict, Path],
        web_image_or_url: Union[str, Image.Image, Path]
    ) -> PipelineResult:
        """Figma 데이터와 웹 이미지를 받아 전체 매칭 프로세스 실행"""
        
        self._current_result = PipelineResult()
        
        try:
            with self._track_stage(ProcessingStage.INITIALIZATION):
                self._validate_inputs(figma_data_or_path, web_image_or_url)
            
            # Figma 요소 추출
            with self._track_stage(ProcessingStage.FIGMA_EXTRACTION):
                figma_elements = self._extract_figma_elements(figma_data_or_path)
                self._current_result.figma_elements = figma_elements
            
            # 웹 요소 추출
            with self._track_stage(ProcessingStage.WEB_EXTRACTION):
                web_image = self._prepare_web_image(web_image_or_url)
                web_elements = self._extract_web_elements(web_image)
                self._current_result.web_elements = web_elements
            
            # 매칭 수행
            with self._track_stage(ProcessingStage.MATCHING):
                matches, unmatched_figma, unmatched_web = self._perform_matching(
                    figma_elements, web_elements
                )
                self._current_result.matches = matches
                self._current_result.unmatched_figma = unmatched_figma
                self._current_result.unmatched_web = unmatched_web
            
            # 결과 후처리
            with self._track_stage(ProcessingStage.RESULT_PROCESSING):
                self._post_process_results()
            
            # 시각화 (옵션)
            if self.config.enable_visualization and self.visualizer:
                with self._track_stage(ProcessingStage.VISUALIZATION):
                    self._create_visualizations(web_image)
            
            self._current_result.stage = ProcessingStage.COMPLETED
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self._current_result.stage = ProcessingStage.FAILED
            self._current_result.error_message = str(e)
        
        return self._current_result
    
    def process_from_mapping_data(
        self,
        current_url: str,
        current_page: str,
        figma_url: str
    ) -> List[Any]:
        """기존 mapping 함수 호환 인터페이스"""
        try:
            # 웹 네비게이션 및 스크린샷
            with WebNavigator() as navigator:
                navigator.navigate(current_url)
                web_image = navigator.capture_screenshot()
            
            # Figma 데이터 로드
            figma_loader = FigmaDataLoader(figma_url)
            figma_data = figma_loader.load_data()
            
            # 파이프라인 실행
            result = self.process_figma_and_web(figma_data, web_image)
            
            if result.stage == ProcessingStage.COMPLETED:
                # 기존 반환 형식에 맞게 변환
                return self._convert_to_legacy_format(result)
            else:
                self.logger.error(f"Pipeline failed: {result.error_message}")
                return []
                
        except Exception as e:
            self.logger.error(f"Mapping process failed: {e}")
            return []
    
    def _validate_inputs(self, figma_data: Any, web_image: Any) -> None:
        """입력 데이터 검증"""
        if figma_data is None:
            raise ValueError("Figma data cannot be None")
        
        if web_image is None:
            raise ValueError("Web image cannot be None")
        
        # 추가 검증 로직...
    
    def _extract_figma_elements(self, figma_data_or_path: Union[str, dict, Path]) -> List[FigmaFare]:
        """Figma 요소 추출"""
        # Figma 데이터 로딩
        if isinstance(figma_data_or_path, (str, Path)):
            loader = FigmaDataLoader(str(figma_data_or_path))
            figma_data = loader.load_data()
        else:
            figma_data = figma_data_or_path
        
        # TODO: Figma 데이터에서 FigmaFare 객체 생성 로직 구현
        # 현재는 빈 리스트 반환
        figma_elements = []
        
        self.logger.info(f"Extracted {len(figma_elements)} Figma elements")
        return figma_elements
    
    def _prepare_web_image(self, web_image_or_url: Union[str, Image.Image, Path]) -> Image.Image:
        """웹 이미지 준비"""
        if isinstance(web_image_or_url, Image.Image):
            return web_image_or_url
        
        elif isinstance(web_image_or_url, (str, Path)):
            # URL인지 파일 경로인지 확인
            if str(web_image_or_url).startswith(('http://', 'https://')):
                # URL에서 스크린샷 캡처
                with WebNavigator() as navigator:
                    navigator.navigate(str(web_image_or_url))
                    return navigator.capture_screenshot()
            else:
                # 파일에서 이미지 로드
                return Image.open(web_image_or_url)
        
        else:
            raise ValueError(f"Unsupported web image type: {type(web_image_or_url)}")
    
    def _extract_web_elements(self, web_image: Image.Image) -> List[ExtractedElement]:
        """웹 요소 추출"""
        elements = self.extractor.extract_elements(
            web_image, 
            include_ocr=self.config.include_ocr
        )
        
        # 요소 수 제한
        if len(elements) > self.config.max_elements_per_stage:
            self.logger.warning(
                f"Too many elements ({len(elements)}), limiting to {self.config.max_elements_per_stage}"
            )
            elements = elements[:self.config.max_elements_per_stage]
        
        self.logger.info(f"Extracted {len(elements)} web elements")
        return elements
    
    def _perform_matching(
        self,
        figma_elements: List[FigmaFare],
        web_elements: List[ExtractedElement]
    ) -> Tuple[List[MatchResult], List[MatchResult], List[MatchResult]]:
        """매칭 수행"""
        matches, unmatched_figma, unmatched_web = self.matcher.find_matches(
            figma_elements,
            web_elements,
            min_similarity=self.config.min_similarity_threshold
        )
        
        self.logger.info(
            f"Matching completed: {len(matches)} matches, "
            f"{len(unmatched_figma)} unmatched Figma, "
            f"{len(unmatched_web)} unmatched web elements"
        )
        
        return matches, unmatched_figma, unmatched_web
    
    def _post_process_results(self) -> None:
        """결과 후처리"""
        # 에러 체크
        if self.error_checker and self.config.enable_error_checking:
            for match in self._current_result.matches:
                match.errorCategories = self.error_checker.check_match(match)

        # 통계 정보 추가
        self._current_result.debug_info.update({
            'total_processing_time': self._current_result.total_time,
            'success_rate': self._current_result.success_rate,
            'processing_breakdown': self._current_result.processing_time
        })
    
    def _create_visualizations(self, web_image: Image.Image) -> None:
        """시각화 생성"""
        if not self.visualizer:
            return
        
        try:
            # 매칭 결과 시각화
            matched_web_boxes = [match.web.box for match in self._current_result.matches if match.web]
            unmatched_web_boxes = [match.web.box for match in self._current_result.unmatched_web if match.web]
            
            self.visualizer.visualize_boxes(
                web_image, 
                np.array(matched_web_boxes), 
                "Matched Web Elements"
            )
            
            if unmatched_web_boxes:
                self.visualizer.visualize_boxes(
                    web_image, 
                    np.array(unmatched_web_boxes), 
                    "Unmatched Web Elements"
                )
            
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
    
    def _convert_to_legacy_format(self, result: PipelineResult) -> List[Any]:
        """기존 반환 형식으로 변환"""
        # TODO: 기존 mapping 함수의 반환 형식에 맞게 변환
        return []
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """파이프라인 성능 메트릭"""
        return {
            'current_stage': self._current_result.stage.value,
            'processing_times': self._current_result.processing_time,
            'total_time': self._current_result.total_time,
            'success_rate': self._current_result.success_rate,
            'element_counts': {
                'figma': len(self._current_result.figma_elements),
                'web': len(self._current_result.web_elements),
                'matches': len(self._current_result.matches)
            }
        }
    
    def __enter__(self) -> 'UIMatchingPipeline':
        """컨텍스트 매니저 지원"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """리소스 정리"""
        if hasattr(self.extractor, 'cleanup'):
            self.extractor.cleanup()


# 팩토리 함수들
def create_pipeline(
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> UIMatchingPipeline:
    """UIMatchingPipeline 생성"""
    return UIMatchingPipeline(config=config, **kwargs)


def create_default_pipeline() -> UIMatchingPipeline:
    """기본 설정으로 파이프라인 생성"""
    config = PipelineConfig.from_env()
    return UIMatchingPipeline(config=config)


# 편의 함수
def quick_match(
    figma_data: Union[str, dict, Path],
    web_image: Union[str, Image.Image, Path],
    min_similarity: float = 0.7,
    enable_debug: bool = False
) -> PipelineResult:
    """빠른 매칭 실행"""
    config = PipelineConfig(
        min_similarity_threshold=min_similarity,
        enable_visualization=enable_debug,
        similarity_config=SimilarityConfig(debug_mode=enable_debug)
    )
    
    with create_pipeline(config) as pipeline:
        return pipeline.process_figma_and_web(figma_data, web_image)
