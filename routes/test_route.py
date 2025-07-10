from fastapi import APIRouter
from routes.dto.request import MappingRequest, InteractionRequest
from routes.dto.response import MappingResponse, MappingInfo, InterActionResponse, InterActionInfo
from typing import List
from service.component_test import execute_component_mapping_test
from service.interaction_test import execute_interaction_test

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

@router.post("/interaction", response_model=InterActionResponse)
async def execute_routing(
        request: InteractionRequest
):
    current_url = request.currentUrl
    current_page = request.currentPage
    figma_url = request.figmaUrl

    try:
        interaction_infos: List[InterActionInfo] = await execute_interaction_test(current_url, current_page, figma_url)

        return InterActionResponse(interactions=interaction_infos)

    except Exception as e:
        return {"error": f"테스트 중 오류가 발생했습니다.: {str(e)}"}