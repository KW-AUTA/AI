from fastapi import APIRouter
from routes.dto.request.mapping_request import MappingRequest
from routes.dto.response.mapping_response import MappingResponse, MappingInfo
from typing import List
from service.component_test import execute_component_mapping_test

router = APIRouter()
figma_cache = {}

@router.post("/mapping", response_model=MappingResponse)
async def execute_routing(
        request: MappingRequest
):
    current_url = request.currentUrl
    current_page = request.currentPage
    figma_url = request.figmaUrl

    try:
        mapping_infos: List[MappingInfo] = await execute_component_mapping_test(current_url, current_page, figma_url)

        return MappingResponse(mappings=mapping_infos)

    except Exception as e:
        return {"error": f"테스트 중 오류가 발생했습니다.: {str(e)}"}
