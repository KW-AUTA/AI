"""
Distributed Processor - Ray 기반 분산 처리

이 모듈은 Ray를 사용한 분산 병렬 처리 기능을 제공합니다.
대규모 이미지 처리를 위한 최적화된 구현입니다.
"""

import logging
import time
import io
from typing import List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray is not installed. Distributed processing will not be available.")

from ..core.models import ExtractedElement
from ..core.element_matcher import ElementExtractor as LegacyElementExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass(frozen=True)
class RayConfig:
    """Ray 분산 처리 설정"""
    num_cpus: Optional[int] = None  # None이면 자동 감지
    log_to_driver: bool = False
    ignore_reinit_error: bool = True

    @classmethod
    def default(cls) -> 'RayConfig':
        """기본 설정 반환"""
        return cls()


if RAY_AVAILABLE:
    @ray.remote
    class ElementExtractorActor:
        """Ray Actor for element extraction with YOLO model"""

        def __init__(self):
            try:
                import torch
                import os

                # GPU 사용 비활성화
                if torch.cuda.is_available():
                    torch.cuda.set_device(-1)
                os.environ['CUDA_VISIBLE_DEVICES'] = ''

                logging.info("🔧 Initializing YOLO model in Ray actor (CPU mode)...")
                self.matcher = LegacyElementExtractor(resize_size=(736, 736))

                # YOLO 모델을 CPU 모드로 강제 설정
                self.matcher.yolo.model.to('cpu')

                logging.info("✅ YOLO model initialized in Ray actor (CPU mode)")
            except Exception as e:
                logging.error(f"❌ Error initializing Ray actor: {e}")
                raise e

        def extract_elements_with_ocr(
            self,
            image_data,
            target_height: int,
            task_name: str,
            start_x: int = 0,
            include_ocr: bool = True
        ):
            """Ray 원격 함수로 요소 추출 및 OCR

            Args:
                image_data: 이미지 데이터 (bytes)
                target_height: 타겟 높이
                task_name: 태스크 이름
                start_x: 시작 x 좌표
                include_ocr: OCR 포함 여부

            Returns:
                추출된 요소 리스트
            """
            try:
                logging.info(f"🔥 Starting {task_name} elements extraction in Ray actor...")
                start_time = time.time()

                # 이미지 데이터를 PIL Image로 변환
                if isinstance(image_data, bytes):
                    image = Image.open(io.BytesIO(image_data))
                else:
                    image = image_data

                # 요소 추출 (mapping.py의 extract_elements 함수 사용)
                from ..core.mapping import extract_elements
                extracted_elements = extract_elements(
                    image,
                    start_x,
                    target_height,
                    self.matcher
                )

                extraction_time = time.time()
                logging.info(
                    f"✅ {task_name} elements extraction completed: "
                    f"{extraction_time - start_time:.2f} seconds"
                )
                logging.info(f"{task_name} elements extracted: {len(extracted_elements)}")

                # OCR 처리 (요청된 경우)
                if include_ocr and extracted_elements:
                    logging.info(f"🔤 Starting OCR for {task_name} elements in Ray actor...")
                    ocr_start = time.time()

                    # 각 요소에 대해 OCR 수행
                    for element in extracted_elements:
                        text_margin = 10
                        x1, y1, x2, y2 = map(int, element.box)
                        margin_box = (
                            x1 - text_margin,
                            y1 - text_margin,
                            x2 + text_margin,
                            y2 + text_margin
                        )
                        element.text = self.matcher.extract_text(image, margin_box)

                    ocr_time = time.time()
                    logging.info(
                        f"✅ {task_name} OCR completed: "
                        f"{ocr_time - ocr_start:.2f} seconds"
                    )

                total_time = time.time()
                logging.info(
                    f"🎯 {task_name} total processing time: "
                    f"{total_time - start_time:.2f} seconds"
                )

                return extracted_elements

            except Exception as e:
                logging.error(f"❌ Error in {task_name} elements extraction/OCR: {e}")
                raise e


class DistributedProcessor:
    """
    Ray 기반 분산 처리 클래스

    주요 기능:
    - Ray 초기화 및 관리
    - 분산 병렬 요소 추출
    - Actor 생명주기 관리
    """

    def __init__(self, config: Optional[RayConfig] = None):
        """
        Args:
            config: Ray 설정
        """
        if not RAY_AVAILABLE:
            raise RuntimeError(
                "Ray is not installed. Please install it with: pip install ray"
            )

        self.config = config or RayConfig.default()
        self._initialized = False

    def initialize(self):
        """Ray 초기화"""
        if self._initialized:
            logging.info("Ray is already initialized")
            return

        if not ray.is_initialized():
            logging.info("🚀 Initializing Ray...")
            ray.init(
                num_cpus=self.config.num_cpus,
                ignore_reinit_error=self.config.ignore_reinit_error,
                log_to_driver=self.config.log_to_driver
            )
            logging.info("✅ Ray initialized successfully")

        self._initialized = True

    def shutdown(self):
        """Ray 종료"""
        if ray.is_initialized():
            logging.info("🔧 Shutting down Ray...")
            ray.shutdown()
            logging.info("✅ Ray shutdown completed")
            self._initialized = False

    def extract_elements_parallel(
        self,
        figma_image: Image.Image,
        web_image: Image.Image,
        target_height: int = 540,
        start_x: int = 0,
        include_ocr: bool = True
    ) -> Tuple[List[ExtractedElement], List[ExtractedElement]]:
        """
        Ray를 사용한 분산 병렬 요소 추출

        Args:
            figma_image: Figma 이미지
            web_image: Web 이미지
            target_height: 타겟 높이
            start_x: 시작 x 좌표
            include_ocr: OCR 포함 여부

        Returns:
            (Figma 추출 요소, Web 추출 요소) 튜플
        """
        self.initialize()

        if include_ocr:
            logging.info("🚀 Starting Ray distributed elements extraction + OCR...")
        else:
            logging.info("🚀 Starting Ray distributed elements extraction...")

        # 이미지를 bytes로 변환 (Ray 간 전달용)
        figma_buffer = io.BytesIO()
        figma_image.save(figma_buffer, format='PNG')
        figma_data = figma_buffer.getvalue()

        web_buffer = io.BytesIO()
        web_image.save(web_buffer, format='PNG')
        web_data = web_buffer.getvalue()

        # Ray를 사용한 분산 처리
        start_time = time.time()

        # Ray actors 생성 (2개의 워커)
        figma_actor = ElementExtractorActor.remote()
        web_actor = ElementExtractorActor.remote()

        try:
            # Figma와 Web 요소 추출을 동시에 실행
            figma_future = figma_actor.extract_elements_with_ocr.remote(
                figma_data, target_height, "Figma", start_x, include_ocr
            )
            web_future = web_actor.extract_elements_with_ocr.remote(
                web_data, target_height, "Web", start_x, include_ocr
            )

            # 결과 수집 (Ray.get으로 결과 대기)
            figma_extracted, web_extracted = ray.get([figma_future, web_future])

        finally:
            # Ray actors 정리
            ray.kill(figma_actor)
            ray.kill(web_actor)

        end_time = time.time()
        if include_ocr:
            logging.info(
                f"🎯 Ray distributed elements extraction + OCR completed: "
                f"{end_time - start_time:.2f} seconds"
            )
        else:
            logging.info(
                f"🎯 Ray distributed elements extraction completed: "
                f"{end_time - start_time:.2f} seconds"
            )

        return figma_extracted, web_extracted

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        # Ray는 프로세스 전역이므로 여기서 종료하지 않음
        # 필요시 명시적으로 shutdown() 호출
        pass

    @classmethod
    def create(cls, config: Optional[RayConfig] = None) -> 'DistributedProcessor':
        """팩토리 메서드: DistributedProcessor 인스턴스 생성

        Args:
            config: Ray 설정

        Returns:
            DistributedProcessor 인스턴스
        """
        return cls(config)


# 편의 함수
def create_distributed_processor(
    config: Optional[RayConfig] = None
) -> Optional[DistributedProcessor]:
    """DistributedProcessor 생성 편의 함수

    Args:
        config: Ray 설정

    Returns:
        DistributedProcessor 인스턴스 또는 None (Ray 미설치 시)
    """
    if not RAY_AVAILABLE:
        logging.warning("Ray is not available. Returning None.")
        return None

    return DistributedProcessor.create(config)


# Ray가 설치되지 않은 경우를 위한 더미 클래스
if not RAY_AVAILABLE:
    class DistributedProcessor:
        """Ray 미설치 시 더미 클래스"""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Ray is not installed. Distributed processing is not available. "
                "Please install it with: pip install ray"
            )
