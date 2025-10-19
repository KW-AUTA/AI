from pydantic import BaseModel

class MappingRequest(BaseModel):
    currentUrl: str
    currentPage: str
    figmaUrl: str

class InteractionRequest(BaseModel):
    currentUrl: str
    currentPage: str
    figmaUrl: str

class UITestRequest(BaseModel):
    figmaJsonUrl: str