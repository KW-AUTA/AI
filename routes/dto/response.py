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


class InteractionMappingInfo(BaseMappingInfo):
    type: Literal["INTERACTION"]
    expectedAction: Optional[str]
    actualAction: Optional[str]


class GeneralMappingInfo(BaseMappingInfo):
    type: Literal["GENERAL"]


MappingInfo = Union[RoutingMappingInfo, InteractionMappingInfo, GeneralMappingInfo]


class MappingResponse(BaseModel):
    mappings: List[MappingInfo]
