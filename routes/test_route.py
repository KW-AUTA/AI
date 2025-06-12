from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from db.deps import get_db
from routes.dto.request.mapping_request import MappingRequest
from routes.dto.response.mapping_response import MappingResponse, MappingInfo
from typing import List
from service.component_test import execute_component_mapping_test
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# prefix 제거하고 루트 경로로 설정
router = APIRouter(prefix="")  # 또는 그냥 router = APIRouter()
figma_cache = {}

@router.post("/mapping", response_model=MappingResponse)
async def execute_routing(
        request: MappingRequest,
        db: AsyncSession = Depends(get_db)
):
    try:
        logger.debug(f"Request received - URL: {request.currentUrl}, Page: {request.currentPage}, Project ID: {request.projectId}")
        
        mapping_response = await execute_component_mapping_test(
            current_url=request.currentUrl,
            current_page=request.currentPage,
            project_id=request.projectId,
            db=db
        )
        
        logger.debug(f"Mapping result: {mapping_response}")
        return mapping_response

    except Exception as e:
        logger.error(f"Error in execute_routing: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"테스트 중 오류가 발생했습니다: {str(e)}"
        )
