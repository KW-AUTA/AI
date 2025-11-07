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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .extractor import ElementExtractor, ExtractorConfig, create_extractor
from .matcher import SimilarityMatcher, create_similarity_matcher
from .models import ExtractedElement, FigmaFare, MatchResult, SimilarityConfig
from ..figma.figma import FigmaProcessor, FigmaProcessorConfig, get_frame_by_name_from_raw, get_img_by_id
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
        error_checker: Optional[ErrorChecker] = None,
        figma_processor: Optional[FigmaProcessor] = None
    ):
        self.config = config or PipelineConfig.from_env()
        
        # 의존성 주입
        self.extractor = extractor or create_extractor(self.config.extractor_config)
        self.matcher = matcher or create_similarity_matcher(self.config.similarity_config)
        self.visualizer = visualizer or Visualizer() if self.config.enable_visualization else None
        self.error_checker = error_checker or ErrorChecker() if self.config.enable_error_checking else None
        self.figma_processor = figma_processor or FigmaProcessor(FigmaProcessorConfig())
        
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
                # figma_data_or_path가 dict인 경우 current_page를 포함할 수 있도록 처리
                current_page = None
                figma_interactions = None

                if isinstance(figma_data_or_path, dict):
                    if '_current_page' in figma_data_or_path:
                        current_page = figma_data_or_path.pop('_current_page')
                    # interactions 정보 추출 (매칭에 사용)
                    figma_interactions = figma_data_or_path.get('interactions', [])

                figma_elements = self._extract_figma_elements(figma_data_or_path, current_page)
                self._current_result.figma_elements = figma_elements

            # 웹 요소 추출
            with self._track_stage(ProcessingStage.WEB_EXTRACTION):
                web_image = self._prepare_web_image(web_image_or_url)
                web_elements = self._extract_web_elements(web_image)
                self._current_result.web_elements = web_elements

            # 매칭 수행 (인터랙션 정보 전달)
            with self._track_stage(ProcessingStage.MATCHING):
                matches, unmatched_figma, unmatched_web = self._perform_matching(
                    figma_elements, web_elements, figma_interactions
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
        """기존 mapping 함수 호환 인터페이스 - 완전한 객체 지향 구현"""
        try:
            from ..web.web_navigator import WebNavigatorConfig

            # Figma 문서 로드 (객체 기반)
            figma_document = self.figma_processor.load_document(figma_url)

            # 프레임 찾기 (객체 메서드 사용)
            frame = figma_document.get_frame_by_name(current_page)
            if frame is None:
                raise ValueError(
                    f"Frame '{current_page}' not found. "
                    f"Available frames: {figma_document.frame_names}"
                )

            # 프레임에서 이미지 가져오기
            root_image = frame.img
            if not frame.has_image:
                raise ValueError(f"Frame '{current_page}' has no image")

            # 인터랙션 정보
            figma_interactions = figma_document.interactions

            # current_page를 figma_data에 추가 (process_figma_and_web에서 사용)
            figma_data_with_page = figma_document.raw_data.copy()
            figma_data_with_page['_current_page'] = current_page

            # 웹 네비게이터 생성 (인터랙션 테스트에 필요)
            nav_config = WebNavigatorConfig(headless=True, base_url=current_url)
            web_navigator = WebNavigator(config=nav_config)

            try:
                web_navigator.navigate(current_url)
                web_image = web_navigator.capture_full_page_with_scroll(root_image, 720)

                # 파이프라인 실행
                result = self.process_figma_and_web(figma_data_with_page, web_image)

                if result.stage != ProcessingStage.COMPLETED:
                    self.logger.error(f"Pipeline failed: {result.error_message}")
                    return []

                # 인터랙션/일반 요소 분리
                interaction_source_ids = {
                    interaction['interactionType']['sourceId']
                    for interaction in figma_interactions
                }

                matches_interaction = [
                    m for m in result.matches
                    if m.figma and m.figma.id in interaction_source_ids
                ]
                matches_no_interaction = [
                    m for m in result.matches
                    if m.figma and m.figma.id not in interaction_source_ids
                ]

                self.logger.info(f"Found {len(matches_interaction)} matches with interactions")
                self.logger.info(f"Found {len(matches_no_interaction)} matches without interactions")

                # 결과 변환
                return_matches = []

                # 인터랙션 테스트 (capture_full_page_with_scroll이 이미 scroll_to_top 호출함)
                if matches_interaction and figma_interactions:
                    # 인터랙션 테스트 (객체 지향 방식)
                    interaction_results = self._test_interactions(
                        matches_interaction,
                        figma_interactions,
                        web_navigator,
                        figma_document
                    )
                    return_matches.extend(interaction_results)

                # 일반 요소 처리
                general_results = self._convert_to_legacy_format(result)
                return_matches.extend(general_results)

                return return_matches

            finally:
                if web_navigator and web_navigator.driver is not None:
                    web_navigator.quit()

        except Exception as e:
            self.logger.error(f"Mapping process failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _validate_inputs(self, figma_data: Any, web_image: Any) -> None:
        """입력 데이터 검증"""
        if figma_data is None:
            raise ValueError("Figma data cannot be None")
        
        if web_image is None:
            raise ValueError("Web image cannot be None")
        
        # 추가 검증 로직...
    
    def _extract_figma_elements(
        self,
        figma_data_or_path: Union[str, dict, Path],
        current_page: Optional[str] = None
    ) -> List[FigmaFare]:
        """Figma 요소 추출 - YOLO로 검출 후 트리와 매칭"""
        from .element_matcher import ElementExtractor as LegacyElementExtractor
        from ..utils.tree_loader import TreeManager
        from .mapping import extract_elements, fare_figma_extracted, convert_raw_to_tree, get_start_x

        # Figma 데이터 로딩
        if isinstance(figma_data_or_path, dict):
            figma_data = figma_data_or_path
        else:
            document = self.figma_processor.load_document(figma_data_or_path)
            figma_data = document.raw_data

        # tree 추출
        figma_tree_list = figma_data.get('tree', [])
        figma_interactions = figma_data.get('interactions', [])

        # current_page가 지정되지 않으면 첫 번째 프레임 사용
        if current_page:
            root_frame = get_frame_by_name_from_raw(figma_tree_list, current_page)
            if root_frame is None:
                raise ValueError(f"Frame '{current_page}' not found in Figma data")
        else:
            root_frame = figma_tree_list[0] if figma_tree_list else None
            if root_frame is None:
                raise ValueError("No frames found in Figma data")

        # 루트 이미지 가져오기 (FigmaFrame 객체 사용하면 자동으로 RGB 변환됨)
        root_image = get_img_by_id(root_frame['data']['id'], figma_tree_list)
        if root_image is None:
            raise ValueError("Failed to load Figma root image")

        # 이미지가 RGB가 아니면 변환 (안전장치)
        if root_image.mode == 'RGBA':
            background = Image.new('RGB', root_image.size, (255, 255, 255))
            background.paste(root_image, mask=root_image.split()[3])
            root_image = background
        elif root_image.mode != 'RGB':
            root_image = root_image.convert('RGB')

        # Figma 트리 구조 생성
        figma_tree_node = convert_raw_to_tree(root_frame, root_image)
        start_x = get_start_x(figma_tree_node)

        # YOLO로 요소 추출
        matcher = LegacyElementExtractor(resize_size=(736, 736))
        target_height = 720
        figma_extracted = extract_elements(
            root_image,
            start_x,
            target_height,
            matcher,
            speed_mode="balanced"
        )

        self.logger.info(f"Extracted {len(figma_extracted)} raw elements from Figma image")

        # Figma 트리와 YOLO 검출 결과 매칭
        figma_elements = fare_figma_extracted(
            figma_tree_node,
            figma_extracted,
            figma_interactions
        )

        self.logger.info(f"Matched {len(figma_elements)} Figma elements with tree nodes")
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
                    return navigator.capture_full_page()
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
        web_elements: List[ExtractedElement],
        figma_interactions: Optional[List[Dict]] = None
    ) -> Tuple[List[MatchResult], List[MatchResult], List[ExtractedElement]]:
        """매칭 수행 - 인터랙션 우선순위 적용"""

        # 인터랙션 정보가 있으면 분리하여 우선 매칭
        if figma_interactions:
            interaction_source_ids = {
                interaction['interactionType']['sourceId']
                for interaction in figma_interactions
            }

            # 인터랙션 있는 요소와 없는 요소 분리
            figma_with_interaction = [
                f for f in figma_elements
                if f.id in interaction_source_ids
            ]
            figma_without_interaction = [
                f for f in figma_elements
                if f.id not in interaction_source_ids
            ]

            self.logger.info(
                f"Found {len(figma_with_interaction)} figma elements with interactions, "
                f"{len(figma_without_interaction)} without"
            )

            # Step 1: 인터랙션 있는 요소 먼저 매칭
            matches_interaction, unmatched_figma_interaction, unmatched_web_interaction = \
                self.matcher.find_matches(
                    figma_with_interaction,
                    web_elements,
                    min_similarity=self.config.min_similarity_threshold
                )

            self.logger.info(f"Matched {len(matches_interaction)} interaction elements")

            # Step 2: 남은 웹 요소로 인터랙션 없는 요소 매칭
            matched_web_ids = {id(match.web) for match in matches_interaction}
            remaining_web_elements = [
                web_el for web_el in web_elements
                if id(web_el) not in matched_web_ids
            ]

            self.logger.info(f"{len(remaining_web_elements)} web elements remaining for second match")

            if figma_without_interaction and remaining_web_elements:
                matches_no_interaction, unmatched_figma_no_interaction, unmatched_web_no_interaction = \
                    self.matcher.find_matches(
                        figma_without_interaction,
                        remaining_web_elements,
                        min_similarity=self.config.min_similarity_threshold
                    )
                self.logger.info(f"Matched {len(matches_no_interaction)} non-interaction elements")
            else:
                matches_no_interaction = []
                unmatched_figma_no_interaction = figma_without_interaction
                unmatched_web_no_interaction = remaining_web_elements

            # 결과 통합
            all_matches = matches_interaction + matches_no_interaction

            # unmatched 교집합 계산
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

        else:
            # 인터랙션 정보가 없으면 일반 매칭
            all_matches, unmatched_figma, unmatched_web = self.matcher.find_matches(
                figma_elements,
                web_elements,
                min_similarity=self.config.min_similarity_threshold
            )

        self.logger.info(
            f"Matching completed: {len(all_matches)} matches, "
            f"{len(unmatched_figma)} unmatched Figma, "
            f"{len(unmatched_web)} unmatched web elements"
        )

        return all_matches, unmatched_figma, unmatched_web
    
    def _post_process_results(self) -> None:
        """결과 후처리"""
        # 에러 체크
        if self.error_checker and self.config.enable_error_checking:
            self.logger.info(f"Running error checker on {len(self._current_result.matches)} matches")
            for match in self._current_result.matches:
                errors = self.error_checker.check_match(match)
                match.errorCategories = errors
                self.logger.debug(f"Match {match.figma.name if match.figma else 'Unknown'}: errors={errors}")
        else:
            self.logger.warning("Error checker is disabled or not available")

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
    
    def _test_interactions(
        self,
        matches_with_interaction: List[MatchResult],
        interactions: List[Dict],
        web_navigator: WebNavigator,
        figma_document: 'FigmaDocument'
    ) -> List[Any]:
        """인터랙션 테스트 (객체 지향 방식)"""
        from routes.dto.response import InteractionMappingInfo, RoutingMappingInfo
        from .interaction_tester import InteractionTester

        # Figma 이미지 딕셔너리 생성
        figma_images = {frame.id: frame.img for frame in figma_document.frames if frame.has_image}

        # Figma ID-to-name 매핑 딕셔너리 생성
        figma_id_to_name = {frame.id: frame.name for frame in figma_document.frames}

        # InteractionTester 생성
        tester = InteractionTester(web_navigator, figma_images, figma_id_to_name)

        # 인터랙션별 매핑 생성 (빠른 검색용)
        interaction_by_source = {
            interaction['interactionType']['sourceId']: interaction
            for interaction in interactions
        }

        results = []

        # 각 매칭에 대해 인터랙션 테스트
        for match in matches_with_interaction:
            interaction = interaction_by_source.get(match.figma.id)
            if not interaction:
                continue

            self.logger.info(f"Testing interaction for {match.figma.name}: {interaction['interactionType']['navigation']}")

            # 인터랙션 테스트 실행
            test_results = tester.test_interaction(match, interaction)

            # 결과를 레거시 형식으로 변환
            for test_result in test_results:
                # detail_info 디버깅
                detail_info_value = test_result.detail_info if test_result.detail_info else None
                self.logger.info(f"Interaction {test_result.component_name}: detail_info='{test_result.detail_info}' -> {detail_info_value}")

                if test_result.interaction_type in ['NAVIGATE']:
                    results.append(RoutingMappingInfo(
                        type="ROUTING",
                        componentName=test_result.component_name,
                        destinationFigmaPage=test_result.destination_page,
                        destinationUrl=test_result.destination_url,
                        actualUrl=test_result.destination_url,
                        failReason=test_result.fail_reason,
                        isSuccess=test_result.is_success,
                        detailInfo=detail_info_value
                    ))
                else:  # OVERLAY, BACK
                    results.append(InteractionMappingInfo(
                        type="INTERACTION",
                        componentName=test_result.component_name,
                        expectedAction=test_result.expected_action,
                        actualAction=test_result.actual_action,
                        failReason=test_result.fail_reason,
                        isSuccess=test_result.is_success,
                        detailInfo=detail_info_value
                    ))

        return results

    def _convert_to_legacy_format(self, result: PipelineResult) -> List[Any]:
        """기존 반환 형식으로 변환"""
        from routes.dto.response import GeneralMappingInfo, BaseMappingInfo
        from ..utils.error_list import NORMAL

        return_matches = []

        # 모든 매칭 결과를 GeneralMappingInfo로 변환
        for match in result.matches:
            comp_name = match.figma.name if match.figma else "Unknown"

            # 디버깅: errorCategories 확인
            self.logger.info(f"Processing match {comp_name}: errorCategories={match.errorCategories}")

            # 상세 정보 생성 (테스트: 모든 경우에 생성)
            detail_info = None
            if self.error_checker:
                detail_str = self.error_checker.get_detail_info(match)
                # 빈 문자열이 아닐 때만 설정
                detail_info = detail_str if detail_str else None
                self.logger.info(f"Detail info for {comp_name}: '{detail_str}' -> {detail_info}")
            else:
                self.logger.info(f"No error_checker available for {comp_name}")

            return_matches.append(GeneralMappingInfo(
                type="GENERAL",
                componentName=comp_name,
                failReason=", ".join(match.errorCategories) if match.errorCategories and match.errorCategories != [NORMAL] else "",
                isSuccess=match.errorCategories == [NORMAL] if match.errorCategories else True,
                detailInfo=detail_info
            ))

        return return_matches
    
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
