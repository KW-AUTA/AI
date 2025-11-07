from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field


class BaseMappingInfo(BaseModel):
    type: str
    componentName: Optional[str]
    isSuccess: bool
    failReason: Optional[str]


class RoutingMappingInfo(BaseMappingInfo):
    type: Literal["ROUTING"]
    destinationFigmaPage: Optional[str]
    destinationUrl: Optional[str]
    actualUrl: Optional[str]
    detailInfo: Optional[str] = None  # 상세 실패 정보 (URL 차이 등)


class InteractionMappingInfo(BaseMappingInfo):
    type: Literal["INTERACTION"]
    expectedAction: Optional[str]
    actualAction: Optional[str]
    detailInfo: Optional[str] = None  # 상세 에러 정보 (이미지 유사도 등)


class GeneralMappingInfo(BaseMappingInfo):
    type: Literal["GENERAL"]
    detailInfo: Optional[str] = None  # 상세 실패 정보 (픽셀 차이, 크기 차이 등)


MappingInfo = Union[RoutingMappingInfo, InteractionMappingInfo, GeneralMappingInfo]


class MappingResponse(BaseModel):
    mappings: List[MappingInfo]


class EvaluationItem(BaseModel):
    frameSummary: str | None = None
    highlightImageUrl: str | None = None
    
class UITestResponse(BaseModel):
    usabilityScore: int
    evaluations: list[EvaluationItem]