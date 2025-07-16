"""
YOLO 기반 요소 매칭 패키지

이 패키지는 Figma 디자인과 웹 페이지 간의 요소 매칭을 수행합니다.
"""

from .models import (
    # Enum 클래스들
    InteractionType,
    TriggerType,
    NavigationType,
    PositionType,
    
    # 데이터 클래스들
    Vector,
    Overlay,
    NodeInteraction,
    UrlInteraction,
    Interaction,
    FigmaElement,
    WebElement,
    MatchResult
)

from .element_matcher import ElementExtractor
from .tree_loader import TreeNode
from .visualize_interaction import FigmaVisualizer

__all__ = [
    # Enum 클래스들
    'InteractionType',
    'TriggerType', 
    'NavigationType',
    'PositionType',
    
    # 데이터 클래스들
    'Vector',
    'Overlay',
    'NodeInteraction',
    'UrlInteraction',
    'Interaction',
    'FigmaElement',
    'WebElement',
    'MatchResult',
    
    # 메인 클래스들
    'ElementExtractor',
    'TreeNode',
    'FigmaVisualizer',
]

__version__ = "1.0.0" 