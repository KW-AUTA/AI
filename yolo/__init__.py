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

from .core.element_matcher import ElementExtractor
from .utils.tree_loader import TreeNode
from .visualization.visualize_interaction import FigmaVisualizer
from .utils.errorChecker import ErrorChecker
from .figma.figma import FigmaDataLoader, FigmaDocument, FigmaFrame, FigmaElementTree, FigmaBox, FigmaBoxPair
from .utils.error_list import *
__all__ = [
    # Enum 클래스들
        
    # 데이터 클래스들
    'UrlInteraction',
    'FigmaElement',
    'WebElement',
    'MatchResult',
    
    # 메인 클래스들
    'ElementExtractor',
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