"""
YOLO 기반 요소 매칭 패키지

이 패키지는 Figma 디자인과 웹 페이지 간의 요소 매칭을 수행합니다.
"""

from .core.models import (
    # Enum 클래스들
    
    # 데이터 클래스들
    UrlInteraction,
    FigmaElement,
    WebElement,
    MatchResult
)

from .core.element_matcher import ElementExtractor as LegacyElementExtractor
from .core.extractor import ElementExtractor, create_extractor
from .core.matcher import SimilarityMatcher, create_similarity_matcher
from .core.pipeline import UIMatchingPipeline, create_pipeline, quick_match
from .utils.tree_loader import TreeNode
from .visualization.visualize_interaction import FigmaVisualizer
from .utils.errorChecker import ErrorChecker
from .figma.figma import FigmaDataLoader, FigmaDocument, FigmaFrame, FigmaElementTree, FigmaBox, FigmaBoxPair
from .utils.figma_utility import FigmaUtilityManager, FigmaUtilityConfig
from .utils.error_list import *
__all__ = [
    # Enum 클래스들

    # 데이터 클래스들
    'UrlInteraction',
    'FigmaElement',
    'WebElement',
    'MatchResult',

    # 새로운 클래스들 (권장)
    'ElementExtractor',
    'SimilarityMatcher',
    'UIMatchingPipeline',
    'FigmaUtilityManager',

    # 설정 클래스들
    'FigmaUtilityConfig',

    # 팩토리 함수들
    'create_extractor',
    'create_similarity_matcher',
    'create_pipeline',
    'quick_match',

    # 레거시 클래스들 (하위 호환성)
    'LegacyElementExtractor',

    # 기타 유틸리티 클래스들
    'TreeNode',
    'FigmaVisualizer',
    'ErrorChecker',
    'FigmaDataLoader',
    'FigmaDocument',
    'FigmaFrame',
    'FigmaElementTree',
    'FigmaBox',
    'FigmaBoxPair',
]

__version__ = "1.0.0" 