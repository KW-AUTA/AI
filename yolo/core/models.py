from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
from enum import Enum
import torch
import numpy as np


@dataclass
class UrlInteraction:
    """URL 상호작용을 나타내는 클래스"""
    url: str
    target: str = "_blank"  # "_blank", "_self", "_parent", "_top"


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


@dataclass(frozen=True)
class SimilarityWeights:
    """유사도 계산을 위한 가중치 설정"""
    # 기본 가중치
    TEXT_WITH_BOTH: float = 0.3      # 양쪽 모두 텍스트가 있을 때
    TEXT_WITHOUT: float = 0.05       # 텍스트가 없을 때
    FEATURE_BASE: float = 0.4        # 피처 유사도 기본 가중치
    SIZE_BASE: float = 0.15          # 크기 유사도 가중치
    COORDINATE_BASE: float = 0.15    # 좌표 유사도 가중치
    
    # 텍스트 가중치 스케일링
    TEXT_SCALE_MIN: float = 0.2      # 텍스트 유사도가 낮을 때 최소 스케일
    TEXT_SCALE_MAX: float = 1.0      # 텍스트 유사도가 높을 때 최대 스케일
    
    # 피처 가중치 스케일링 (텍스트 요소일 때)
    FEAT_SCALE_BOTH_TEXT: float = 0.6   # 양쪽 모두 텍스트가 있을 때
    FEAT_SCALE_XOR_TEXT: float = 0.8    # 한쪽만 텍스트가 있을 때
    FEAT_SCALE_MIN: float = 0.5         # 최소 피처 스케일
    
    # 정규화 파라미터
    QUANTILE_LOW: float = 0.05       # Row-wise quantile normalization 하한
    QUANTILE_HIGH: float = 0.95      # Row-wise quantile normalization 상한
    SOFTMAX_TAU_DEFAULT: float = 0.25  # Softmax temperature 기본값
    
    # 임계값
    MIN_SIMILARITY: float = 0.7      # 최소 유사도 임계값


@dataclass(frozen=True) 
class SimilarityConfig:
    """유사도 매칭 설정"""
    weights: SimilarityWeights = field(default_factory=SimilarityWeights)
    use_softmax_relative: bool = False  # 상대적 정규화에 softmax 사용 여부
    softmax_tau: float = 0.25          # Softmax temperature
    debug_mode: bool = False           # 디버그 모드 (히트맵 저장 등)
    
    @classmethod
    def from_env(cls) -> 'SimilarityConfig':
        """환경변수에서 설정을 로드"""
        import os
        
        # 환경변수에서 값 읽기 (안전한 변환 포함)
        def get_float_env(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, str(default)))
            except (ValueError, TypeError):
                return default
                
        def get_bool_env(key: str, default: bool) -> bool:
            return str(os.environ.get(key, str(default))).lower() in ('1', 'true', 'yes')
        
        return cls(
            weights=SimilarityWeights(
                TEXT_WITH_BOTH=get_float_env('SIM_TEXT_WITH_BOTH', 0.3),
                TEXT_WITHOUT=get_float_env('SIM_TEXT_WITHOUT', 0.05),
                FEATURE_BASE=get_float_env('SIM_FEATURE_BASE', 0.4),
                SIZE_BASE=get_float_env('SIM_SIZE_BASE', 0.15),
                COORDINATE_BASE=get_float_env('SIM_COORDINATE_BASE', 0.15),
                FEAT_SCALE_BOTH_TEXT=get_float_env('SIM_FEAT_TEXT_BOTH', 0.6),
                FEAT_SCALE_XOR_TEXT=get_float_env('SIM_FEAT_TEXT_XOR', 0.8),
                MIN_SIMILARITY=get_float_env('SIM_MIN_SIMILARITY', 0.7)
            ),
            use_softmax_relative=get_bool_env('SIM_REL_SOFTMAX', False),
            softmax_tau=get_float_env('SIM_REL_TAU', 0.25),
            debug_mode=get_bool_env('SIM_DEBUG', False)
        )
    

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