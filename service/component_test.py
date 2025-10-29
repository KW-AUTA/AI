from yolo.core.mapping import mapping
from service.resource_manager import resource_manager
import logging

logger = logging.getLogger(__name__)


async def execute_component_mapping_test(current_url: str, current_page: str, figma_url: str):
    """
    컴포넌트 매핑 테스트 실행 - 리소스 매니저를 사용하여 순차 처리
    """
    # 순차 처리를 위한 락 획득
    with resource_manager.acquire_lock():
        try:
            logger.info(f"Processing mapping request: {figma_url}")

            # 리소스 매니저에서 WebNavigator 가져오기
            web_navigator = resource_manager.get_navigator()

            # 매핑 실행 (이제 Ray와 WebNavigator가 이미 초기화되어 있음)
            mapping_infos = mapping(current_url, current_page, figma_url, web_navigator=web_navigator)

            logger.info(f"Mapping completed successfully")
            return mapping_infos

        except Exception as e:
            logger.error(f"Error in mapping() function: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
