from pydantic import BaseModel

class MappingRequest(BaseModel):
    currentUrl: str
    currentPage: str
    figmaUrl: str
