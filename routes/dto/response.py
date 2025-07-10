from pydantic import BaseModel
from typing import List, Optional

class MappingInfo(BaseModel):
    componentName: Optional[str]
    destinationFigmaPage: Optional[str]
    destinationUrl: Optional[str]
    actualUrl: Optional[str]
    failReason: Optional[str]
    isSuccess: bool
    isRouting: bool

class MappingResponse(BaseModel):
    mappings: List[MappingInfo]

class InterActionInfo(BaseModel):
    componentName: Optional[str]
    expectedAction: Optional[str]
    actualAction: Optional[str]
    failReason: Optional[str]
    isSuccess: bool

class InterActionResponse(BaseModel):
    interactions: List[InterActionInfo]