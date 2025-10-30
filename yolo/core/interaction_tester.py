"""
InteractionTester - UI 인터랙션 테스트 및 검증
- 오버레이 인터랙션 테스트
- 네비게이션 인터랙션 테스트
- 화면 비교 및 유사도 검증
"""

import os
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from PIL import Image, ImageChops
import numpy as np
import torch
from torchvision import transforms
import io

from .models import MatchResult
from ..web.web_navigator import WebNavigator
from ..utils.error_list import (
    NORMAL,
    I_ERROR_NAVIGATE_NOT_OVERLAY,
    I_ERROR_OVERLAY_NOT_FOUND,
    I_ERROR_DIFFERENT_OVERLAY,
    I_ERROR_DIFFERENT_BACK
)


@dataclass
class InteractionResult:
    """인터랙션 테스트 결과"""
    component_name: str
    interaction_type: str  # 'OVERLAY', 'NAVIGATE', 'BACK'
    expected_action: str
    actual_action: str
    is_success: bool
    fail_reason: str = ""
    destination_page: str = ""
    destination_url: str = ""


class InteractionTester:
    """UI 인터랙션 테스트 클래스"""

    def __init__(self, web_navigator: WebNavigator, figma_images: Dict[str, Image.Image], figma_id_to_name: Optional[Dict[str, str]] = None):
        """
        Args:
            web_navigator: 웹 네비게이터 인스턴스
            figma_images: Figma 이미지 딕셔너리 {frame_id: Image}
            figma_id_to_name: Figma ID를 이름으로 변환하는 딕셔너리 (선택)
        """
        self.web_navigator = web_navigator
        self.figma_images = figma_images
        self.figma_id_to_name = figma_id_to_name or {}
        self.logger = logging.getLogger(__name__)

        # 화면 유사도 모델 로드
        self._similarity_model = None
        self._img_transforms = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @property
    def similarity_model(self):
        """지연 로딩으로 모델 로드"""
        if self._similarity_model is None:
            current_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(current_dir, "models_weights", "screensim-resnet-uda+web7k.torchscript")
            self._similarity_model = torch.jit.load(model_path)
        return self._similarity_model

    def test_interaction(
        self,
        match: MatchResult,
        interaction: Dict
    ) -> List[InteractionResult]:
        """인터랙션 테스트 메인 메서드"""
        navigation_type = interaction['interactionType']['navigation']

        if navigation_type == 'OVERLAY':
            return self._test_overlay_interaction(match, interaction)
        elif navigation_type == 'NAVIGATE':
            return self._test_navigate_interaction(match, interaction)
        else:
            self.logger.warning(f"Unknown interaction type: {navigation_type}")
            return []

    def _test_overlay_interaction(
        self,
        match: MatchResult,
        interaction: Dict
    ) -> List[InteractionResult]:
        """오버레이 인터랙션 테스트"""
        results = []

        # 새 탭에서 테스트
        self.web_navigator.driver.execute_script("window.open(window.location.href, '_blank');")
        self.web_navigator.driver.switch_to.window(self.web_navigator.driver.window_handles[-1])
        time.sleep(1)

        try:
            # 요소 클릭
            center_x = float(match.web.box[0] + match.web.box[2]) / 2
            center_y = float(match.web.box[1] + match.web.box[3]) / 2
            element, xpath = self.web_navigator.get_element_at_coordinate_and_xpath(center_x, center_y)

            if element is None or xpath is None:
                results.append(InteractionResult(
                    component_name=match.figma.name,
                    interaction_type='OVERLAY',
                    expected_action='OVERLAY',
                    actual_action='None',
                    is_success=False,
                    fail_reason=I_ERROR_OVERLAY_NOT_FOUND
                ))
                return results

            # 클릭 전 상태 저장
            current_url = self.web_navigator.driver.current_url
            self.web_navigator.scroll_to_y(center_y)
            before_img = Image.open(io.BytesIO(self.web_navigator.driver.get_screenshot_as_png()))

            # 클릭
            time.sleep(1)
            element.click()
            time.sleep(1)

            # URL 변경 확인
            click_url = self.web_navigator.driver.current_url
            if click_url != current_url:
                results.append(InteractionResult(
                    component_name=match.figma.name,
                    interaction_type='OVERLAY',
                    expected_action='OVERLAY',
                    actual_action='NAVIGATE',
                    is_success=False,
                    fail_reason=I_ERROR_NAVIGATE_NOT_OVERLAY
                ))
                return results

            # 오버레이 비교
            destination_id = interaction['interactionType']['destinationId']
            overlay_figma_img = self.figma_images.get(destination_id)

            if overlay_figma_img is None:
                self.logger.warning(f"Overlay Figma image not found: {destination_id}")
                results.append(InteractionResult(
                    component_name=match.figma.name,
                    interaction_type='OVERLAY',
                    expected_action='OVERLAY',
                    actual_action='None',
                    is_success=False,
                    fail_reason=I_ERROR_OVERLAY_NOT_FOUND
                ))
                return results

            # 화면 변화 확인
            after_img = Image.open(io.BytesIO(self.web_navigator.driver.get_screenshot_as_png()))
            diff_img = ImageChops.difference(before_img, after_img)

            if diff_img.getbbox() is None:
                results.append(InteractionResult(
                    component_name=match.figma.name,
                    interaction_type='OVERLAY',
                    expected_action='OVERLAY',
                    actual_action='None',
                    is_success=False,
                    fail_reason=I_ERROR_OVERLAY_NOT_FOUND
                ))
            else:
                # 변화 영역 추출 및 유사도 비교
                bbox = diff_img.getbbox()
                overlay_region = after_img.crop(bbox)

                is_similar = self._compare_images(overlay_region, overlay_figma_img)

                results.append(InteractionResult(
                    component_name=match.figma.name,
                    interaction_type='OVERLAY',
                    expected_action='OVERLAY',
                    actual_action='OVERLAY',
                    is_success=is_similar,
                    fail_reason="" if is_similar else I_ERROR_DIFFERENT_OVERLAY
                ))

        finally:
            # 탭 정리
            self.web_navigator.driver.close()
            self.web_navigator.driver.switch_to.window(self.web_navigator.driver.window_handles[0])

        return results

    def _test_navigate_interaction(
        self,
        match: MatchResult,
        interaction: Dict
    ) -> List[InteractionResult]:
        """네비게이션 인터랙션 테스트"""
        results = []

        # 원본 화면 저장
        original_web_image = Image.open(io.BytesIO(self.web_navigator.driver.get_screenshot_as_png()))

        # 요소 찾기 및 URL 가져오기
        center_x = float(match.web.box[0] + match.web.box[2]) / 2
        center_y = float(match.web.box[1] + match.web.box[3]) / 2
        element, xpath = self.web_navigator.get_element_at_coordinate_and_xpath(center_x, center_y)

        destination_id = interaction['interactionType']['destinationId']
        # ID를 이름으로 변환 (매핑이 없으면 ID 그대로 사용)
        destination_name = self.figma_id_to_name.get(destination_id, destination_id)

        if element is not None and xpath is not None:
            urls = self.web_navigator.get_url_in_new_tab(xpath)
            results.append(InteractionResult(
                component_name=match.figma.name,
                interaction_type='NAVIGATE',
                expected_action='NAVIGATE',
                actual_action='NAVIGATE',
                is_success=True,
                fail_reason=NORMAL,
                destination_page=destination_name,
                destination_url=urls
            ))
        else:
            results.append(InteractionResult(
                component_name=match.figma.name,
                interaction_type='NAVIGATE',
                expected_action='NAVIGATE',
                actual_action='None',
                is_success=True,
                fail_reason=NORMAL,
                destination_page=destination_name,
                destination_url=""
            ))

        # BACK 인터랙션 처리
        if "afterInteraction" in interaction and len(interaction['afterInteraction']) > 0:
            for after_interaction in interaction['afterInteraction']:
                if after_interaction['interactionType']['navigation'] == 'BACK':
                    self.web_navigator.driver.back()
                    back_image = Image.open(io.BytesIO(self.web_navigator.driver.get_screenshot_as_png()))
                    diff_img = ImageChops.difference(original_web_image, back_image)

                    if diff_img.getbbox() is not None:
                        results.append(InteractionResult(
                            component_name=match.figma.name,
                            interaction_type='BACK',
                            expected_action='BACK',
                            actual_action='BACK',
                            is_success=True,
                            fail_reason=NORMAL
                        ))
                    else:
                        results.append(InteractionResult(
                            component_name=match.figma.name,
                            interaction_type='BACK',
                            expected_action='BACK',
                            actual_action='BACK',
                            is_success=False,
                            fail_reason=I_ERROR_DIFFERENT_BACK
                        ))

        # 탭 정리
        self.web_navigator.driver.close()
        self.web_navigator.driver.switch_to.window(self.web_navigator.driver.window_handles[0])

        return results

    def _compare_images(self, img1: Image.Image, img2: Image.Image) -> bool:
        """화면 유사도 비교"""
        try:
            # numpy array를 PIL Image로 변환
            if isinstance(img1, np.ndarray):
                img1 = Image.fromarray(img1.astype(np.uint8))
            if isinstance(img2, np.ndarray):
                img2 = Image.fromarray(img2.astype(np.uint8))

            # 이미지 전처리
            img1_tensor = self._img_transforms(img1)
            img2_tensor = self._img_transforms(img2)

            # 임베딩 추출
            embedding1 = self.similarity_model(img1_tensor.unsqueeze(0))
            embedding2 = self.similarity_model(img2_tensor.unsqueeze(0))

            # 유사도 계산
            dist = torch.linalg.norm(embedding1 - embedding2)
            margin = (0.2 + 0.5) / 2  # margin_pos와 margin_neg의 평균

            is_similar = float(dist) < margin
            self.logger.debug(f"Image similarity distance: {float(dist):.3f}, similar: {is_similar}")

            return is_similar

        except Exception as e:
            self.logger.error(f"Image comparison failed: {e}")
            return False


def create_interaction_tester(
    web_navigator: WebNavigator,
    figma_images: Dict[str, Image.Image],
    figma_id_to_name: Optional[Dict[str, str]] = None
) -> InteractionTester:
    """InteractionTester 팩토리 함수"""
    return InteractionTester(web_navigator, figma_images, figma_id_to_name)
