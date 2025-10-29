"""
리소스 매니저 - Ray 클러스터와 WebNavigator를 서버 생명주기 동안 유지
"""
import ray
from yolo.web.web_navigator import WebNavigator
from typing import Optional
import threading
import logging

logger = logging.getLogger(__name__)


class ResourceManager:
    """싱글톤 리소스 매니저 - Ray와 WebNavigator를 전역으로 관리"""

    _instance: Optional['ResourceManager'] = None
    _lock = threading.Lock()

    def __init__(self):
        self.ray_initialized = False
        self.navigator: Optional[WebNavigator] = None
        self.processing_lock = threading.Lock()  # 순차 처리를 위한 락

    @classmethod
    def get_instance(cls) -> 'ResourceManager':
        """싱글톤 인스턴스 가져오기"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialize(self):
        """서버 시작 시 리소스 초기화"""
        logger.info("Initializing resources...")

        # Ray 초기화
        if not self.ray_initialized:
            try:
                if not ray.is_initialized():
                    ray.init(
                        num_cpus=4,  # CPU 4개 모두 사용
                        ignore_reinit_error=True,
                        logging_level=logging.WARNING
                    )
                    logger.info("Ray cluster initialized")
                self.ray_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                raise

        # WebNavigator 초기화
        if self.navigator is None:
            try:
                self.navigator = WebNavigator()
                logger.info("WebNavigator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize WebNavigator: {e}")
                raise

        logger.info("All resources initialized successfully")

    def shutdown(self):
        """서버 종료 시 리소스 정리"""
        logger.info("Shutting down resources...")

        # WebNavigator 정리
        if self.navigator is not None:
            try:
                self.navigator.quit()
                logger.info("WebNavigator closed")
            except Exception as e:
                logger.warning(f"Error closing WebNavigator: {e}")
            finally:
                self.navigator = None

        # Ray 정리
        if self.ray_initialized:
            try:
                ray.shutdown()
                logger.info("Ray cluster shutdown")
            except Exception as e:
                logger.warning(f"Error shutting down Ray: {e}")
            finally:
                self.ray_initialized = False

        logger.info("All resources cleaned up")

    def get_navigator(self) -> WebNavigator:
        """WebNavigator 인스턴스 가져오기"""
        if self.navigator is None:
            raise RuntimeError("WebNavigator not initialized. Call initialize() first.")
        return self.navigator

    def acquire_lock(self):
        """순차 처리를 위한 락 획득"""
        return self.processing_lock


# 전역 인스턴스
resource_manager = ResourceManager.get_instance()
