"""
새로운 클래스 기반 파이프라인 사용 예제
"""

import logging
from pathlib import Path
import sys

# yolo 패키지 import
sys.path.append(str(Path(__file__).parent.parent.parent))

from yolo import (
    ElementExtractor, SimilarityMatcher, UIMatchingPipeline,
    create_extractor, create_similarity_matcher, create_pipeline, quick_match
)
from yolo.core.models import ExtractorConfig, SimilarityConfig, PipelineConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """기본 사용법 예제"""
    logger.info("=== Example 1: Basic Usage ===")
    
    # 1. 개별 컴포넌트 생성
    extractor = create_extractor()
    matcher = create_similarity_matcher()
    
    # 2. 통합 파이프라인 생성
    pipeline = create_pipeline()
    
    logger.info("Components created successfully!")


def example_2_custom_config():
    """커스텀 설정 사용 예제"""
    logger.info("=== Example 2: Custom Configuration ===")
    
    # 커스텀 설정 생성
    extractor_config = ExtractorConfig(
        resize_size=(640, 640),
        debug_mode=True
    )
    
    similarity_config = SimilarityConfig(
        use_softmax_relative=True,
        debug_mode=True
    )
    
    pipeline_config = PipelineConfig(
        extractor_config=extractor_config,
        similarity_config=similarity_config,
        min_similarity_threshold=0.8,
        enable_visualization=True
    )
    
    # 설정을 사용한 파이프라인 생성
    pipeline = create_pipeline(pipeline_config)
    
    logger.info(f"Custom pipeline created with similarity threshold: {pipeline_config.min_similarity_threshold}")


def example_3_environment_config():
    """환경변수 기반 설정 예제"""
    logger.info("=== Example 3: Environment-based Configuration ===")
    
    import os
    
    # 환경변수 설정
    os.environ['SIM_DEBUG'] = '1'
    os.environ['SIM_MIN_SIMILARITY'] = '0.75'
    os.environ['YOLO_RESIZE_SIZE'] = '800,600'
    os.environ['PIPELINE_ENABLE_VIZ'] = '1'
    
    # 환경변수에서 설정 로드
    config = PipelineConfig.from_env()
    pipeline = create_pipeline(config)
    
    logger.info("Pipeline created from environment variables")
    logger.info(f"Debug mode: {config.similarity_config.debug_mode}")
    logger.info(f"Min similarity: {config.min_similarity_threshold}")


def example_4_dependency_injection():
    """의존성 주입 예제"""
    logger.info("=== Example 4: Dependency Injection ===")
    
    from yolo.core.extractor import TesseractOCR, YOLODetector
    from yolo.core.matcher import TextSimilarityStrategy, FeatureSimilarityStrategy
    
    # 커스텀 컴포넌트 생성
    config = ExtractorConfig()
    
    # OCR 처리기와 YOLO 검출기 생성
    ocr_processor = TesseractOCR(config)
    detector = YOLODetector(config)
    
    # 의존성 주입으로 ElementExtractor 생성
    extractor = ElementExtractor(
        config=config,
        detector=detector,
        ocr_processor=ocr_processor
    )
    
    # 유사도 전략들
    text_strategy = TextSimilarityStrategy()
    feature_strategy = FeatureSimilarityStrategy()
    
    # 의존성 주입으로 SimilarityMatcher 생성
    matcher = SimilarityMatcher(
        text_strategy=text_strategy,
        feature_strategy=feature_strategy
    )
    
    # 통합 파이프라인
    pipeline = UIMatchingPipeline(
        extractor=extractor,
        matcher=matcher
    )
    
    logger.info("Pipeline created with custom dependency injection")


def example_5_context_manager():
    """컨텍스트 매니저 사용 예제"""
    logger.info("=== Example 5: Context Manager Usage ===")
    
    # 자동 리소스 정리를 위한 컨텍스트 매니저 사용
    with create_extractor() as extractor:
        logger.info("Extractor created with automatic cleanup")
        # extractor 사용...
    
    with create_pipeline() as pipeline:
        logger.info("Pipeline created with automatic cleanup")
        # pipeline 사용...
    
    logger.info("Resources automatically cleaned up")


def example_6_legacy_compatibility():
    """레거시 호환성 예제"""
    logger.info("=== Example 6: Legacy Compatibility ===")
    
    from yolo import LegacyElementExtractor
    from yolo.core.mapping import mapping, mapping_v2, mapping_legacy
    
    # 레거시 ElementExtractor (기존 코드)
    legacy_extractor = LegacyElementExtractor()
    logger.info("Legacy extractor created")
    
    # 매핑 함수 호환성 테스트
    logger.info("Available mapping functions:")
    logger.info("- mapping(): 새 파이프라인/레거시 자동 선택")
    logger.info("- mapping_v2(): 새 파이프라인 강제 사용")
    logger.info("- mapping_legacy(): 레거시 구현 강제 사용")


def example_7_quick_match():
    """빠른 매칭 함수 예제"""
    logger.info("=== Example 7: Quick Match Function ===")
    
    # 간단한 API로 빠른 매칭
    # result = quick_match(
    #     figma_data="path/to/figma.json",
    #     web_image="https://example.com",
    #     min_similarity=0.7,
    #     enable_debug=True
    # )
    
    logger.info("Quick match function ready for use")
    logger.info("Usage: quick_match(figma_data, web_image, min_similarity=0.7)")


if __name__ == "__main__":
    logger.info("🚀 YOLO 새로운 클래스 기반 파이프라인 예제")
    logger.info("=" * 60)
    
    try:
        example_1_basic_usage()
        example_2_custom_config()
        example_3_environment_config()
        example_4_dependency_injection()
        example_5_context_manager()
        example_6_legacy_compatibility()
        example_7_quick_match()
        
        logger.info("=" * 60)
        logger.info("✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()