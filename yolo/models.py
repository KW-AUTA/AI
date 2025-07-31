from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
from enum import Enum
import torch
import numpy as np

# Interaction 관련 Enum들
class InteractionType(Enum):
    NODE = "NODE"
    BACK = "BACK"
    CLOSE = "CLOSE"
    URL = "URL"

class TriggerType(Enum):
    ON_CLICK = "ON_CLICK"
    ON_HOVER = "ON_HOVER"
    ON_PRESS = "ON_PRESS"
    ON_DRAG = "ON_DRAG"

class NavigationType(Enum):
    NAVIGATE = "NAVIGATE"
    SWAP = "SWAP"
    OVERLAY = "OVERLAY"
    SCROLL_TO = "SCROLL_TO"
    CHANGE_TO = "CHANGE_TO"

class PositionType(Enum):
    CENTER = "CENTER"
    TOP_LEFT = "TOP_LEFT"
    TOP_CENTER = "TOP_CENTER"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_CENTER = "BOTTOM_CENTER"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"
    MANUAL = "MANUAL"

# Interaction 관련 데이터 클래스들
@dataclass
class Vector:
    """2D 벡터를 나타내는 클래스"""
    x: float
    y: float

@dataclass
class Overlay:
    """오버레이 설정을 나타내는 클래스"""
    positionType: PositionType
    position: Optional[Vector] = None

@dataclass
class NodeInteraction:
    """노드 간 상호작용을 나타내는 클래스"""
    navigation: NavigationType
    sourceId: str
    destinationId: str
    overlay: Optional[Overlay] = None

@dataclass
class UrlInteraction:
    """URL 상호작용을 나타내는 클래스"""
    url: str
    target: str = "_blank"  # "_blank", "_self", "_parent", "_top"

@dataclass
class Interaction:
    """상호작용을 나타내는 메인 클래스"""
    type: InteractionType
    trigger: TriggerType
    interactionType: Union[NodeInteraction, UrlInteraction]


@dataclass(kw_only=True, frozen=True, slots=True)
class FigmaElement:
    id: str
    name: str
    absolute_position: Dict[str, float]
    relative_position: Dict[str, float]
    absolute_render_position: Dict[str, float]
    relative_render_position: Dict[str, float]

@dataclass(kw_only=True, frozen=True, slots=True)
class WebElement:
    xpath: str

@dataclass(kw_only=True, frozen=False, slots=True)
class ExtractedElement:
    box:        np.ndarray
    feature:    torch.Tensor
    text:       str
    cls:        int
    
    def __eq__(self, other):
        """커스텀 비교 메서드: numpy 배열과 torch 텐서를 안전하게 비교"""
        if not isinstance(other, ExtractedElement):
            return False
        
        # feature 비교: 타입이 다를 수 있으므로 안전하게 처리
        feature_equal = False
        try:
            if isinstance(self.feature, torch.Tensor) and isinstance(other.feature, torch.Tensor):
                feature_equal = torch.equal(self.feature, other.feature)
            elif isinstance(self.feature, np.ndarray) and isinstance(other.feature, np.ndarray):
                feature_equal = np.array_equal(self.feature, other.feature)
            elif isinstance(self.feature, torch.Tensor) and isinstance(other.feature, np.ndarray):
                feature_equal = torch.equal(self.feature, torch.from_numpy(other.feature))
            elif isinstance(self.feature, np.ndarray) and isinstance(other.feature, torch.Tensor):
                feature_equal = torch.equal(torch.from_numpy(self.feature), other.feature)
            else:
                feature_equal = False
        except:
            feature_equal = False
        
        return (
            np.array_equal(self.box, other.box) and
            feature_equal and
            self.text == other.text and
            self.cls == other.cls
        )
    
    def __hash__(self):
        """해시 메서드 구현"""
        return hash((
            tuple(self.box.flatten()),
            self.text,
            self.cls
        ))

@dataclass(kw_only=True, frozen=True, slots=True)
class ReturnMatch:
    componentName: ExtractedElement
    destinationFigmaPage: str
    destinationUrl: str
    actualUrl: str
    failReason: str
    isSuccess: bool
    isRouting: bool

@dataclass(kw_only=True, frozen=True, slots=True)
class FigmaFare:
    id: str
    name: str
    box: Tuple[float, float, float, float]
    extracted: ExtractedElement
    
    def __hash__(self):
        """해시 메서드 구현"""
        return hash((self.id, self.name, self.box))


@dataclass(kw_only=True, frozen=True, slots=True)
class WebFare:
    id: str
    xpath: str
    box: Tuple[float, float, float, float]
    extracted: ExtractedElement

@dataclass(kw_only=True, frozen=True, slots=True)
class MatchResult:
    figma: FigmaFare | None 
    web:   ExtractedElement | None
    feature_similarity:    float
    text_similarity:       float
    size_similarity:       float
    coordinate_similarity: float
    score:                 float
    errorCategories:       Optional[List[str]]
    

# 사용 예시
if __name__ == "__main__":
    # Interaction 객체 생성 예시
    # URL 상호작용
    url_interaction = UrlInteraction(url="https://example.com")
    url_interaction_obj = Interaction(
        type=InteractionType.URL,
        trigger=TriggerType.ON_CLICK,
        interactionType=url_interaction
    )
    
    # 노드 상호작용
    node_interaction = NodeInteraction(
        navigation=NavigationType.NAVIGATE,
        sourceId="button_1",
        destinationId="page_2"
    )
    node_interaction_obj = Interaction(
        type=InteractionType.NODE,
        trigger=TriggerType.ON_CLICK,
        interactionType=node_interaction
    )
    
    # 오버레이가 있는 노드 상호작용
    overlay = Overlay(
        positionType=PositionType.CENTER,
        position=Vector(x=100, y=200)
    )
    overlay_interaction = NodeInteraction(
        navigation=NavigationType.OVERLAY,
        sourceId="modal_trigger",
        destinationId="modal_content",
        overlay=overlay
    )
    overlay_interaction_obj = Interaction(
        type=InteractionType.NODE,
        trigger=TriggerType.ON_CLICK,
        interactionType=overlay_interaction
    )
    
    print("Interaction 객체 생성 완료!")
    print(f"URL Interaction: {url_interaction_obj.type.value} -> {url_interaction_obj.interactionType.url}")
    print(f"Node Interaction: {node_interaction_obj.type.value} -> {node_interaction_obj.interactionType.destinationId}")
    print(f"Overlay Interaction: {overlay_interaction_obj.type.value} -> {overlay_interaction_obj.interactionType.navigation.value}") 