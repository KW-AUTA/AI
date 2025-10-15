"""
시각화 모듈 - 매칭 결과 및 바운딩 박스 시각화

이 모듈은 YOLO 검출 결과, Figma-Web 매칭 결과를 시각화하는 기능을 제공합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..core.models import MatchResult


@dataclass(frozen=True)
class VisualizerConfig:
    """시각화 설정 클래스"""
    figure_width: int = 20
    figure_height: int = 10
    box_linewidth: int = 2
    text_fontsize: int = 8
    text_bbox_alpha: float = 0.5
    random_seed: int = 42
    show_match_numbers: bool = True
    connection_line_width: int = 2


class Visualizer:
    """
    시각화를 담당하는 클래스

    의존성 주입 패턴을 사용하여 설정을 관리합니다.
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Args:
            config: 시각화 설정. None이면 기본 설정 사용
        """
        self.config = config or VisualizerConfig()

    def visualize_boxes(
        self,
        img: Image.Image,
        boxes: np.ndarray,
        title: str = "Detected Boxes"
    ) -> None:
        """
        이미지 위에 바운딩 박스를 시각화

        Args:
            img: 입력 이미지
            boxes: 바운딩 박스 배열 (N, 4) - [x1, y1, x2, y2]
            title: 플롯 제목
        """
        plt.figure(figsize=(self.config.figure_width, self.config.figure_height))
        plt.imshow(img)

        for box in boxes:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(
                patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    edgecolor='red',
                    facecolor='none',
                    linewidth=self.config.box_linewidth
                )
            )
            plt.text(
                x1, y1 - 5,
                f"({int(x1)},{int(y1)})",
                color='white',
                fontsize=self.config.text_fontsize,
                bbox=dict(
                    facecolor='black',
                    alpha=self.config.text_bbox_alpha,
                    pad=1
                )
            )

        plt.title(title)
        plt.axis('off')
        plt.show()

    def visualize_matches(
        self,
        figma_img: Image.Image,
        web_img: Image.Image,
        matches: List[MatchResult],
        label: str = "",
        show_error_categories: bool = False
    ) -> None:
        """
        매칭 결과 시각화 (두 이미지를 이어붙이고, 매칭된 박스 중심끼리 선 연결)

        Args:
            figma_img: Figma 이미지
            web_img: 웹 이미지
            matches: 매칭 결과 리스트
            label: 시각화 레이블
            show_error_categories: True면 에러 카테고리 표시, False면 매칭 번호 표시
        """
        if not matches:
            print("\n[실패] 매칭된 요소가 없습니다.")
            return

        # 이미지 준비
        concat_img = self._prepare_concatenated_images(figma_img, web_img)
        overlay = concat_img.copy()

        # 색상 생성
        np.random.seed(self.config.random_seed)
        colors = np.random.randint(0, 255, size=(len(matches), 3)).tolist()

        # 매칭 시각화
        wF = figma_img.width
        for idx, (result, color) in enumerate(zip(matches, colors), 1):
            self._draw_single_match(
                overlay,
                result,
                color,
                idx,
                wF,
                show_error_categories
            )

        # 결과 표시
        self._show_visualization(
            overlay,
            label,
            show_error_categories,
            matches,
            len(matches)
        )

    def visualize_matches_with_error(
        self,
        figma_img: Image.Image,
        web_img: Image.Image,
        matches: List[MatchResult],
        label: str = ""
    ) -> None:
        """
        에러 카테고리와 함께 매칭 결과 시각화

        Args:
            figma_img: Figma 이미지
            web_img: 웹 이미지
            matches: 매칭 결과 리스트
            label: 시각화 레이블
        """
        self.visualize_matches(
            figma_img,
            web_img,
            matches,
            label,
            show_error_categories=True
        )

    # ============================================================================
    # Private Helper Methods
    # ============================================================================

    def _prepare_concatenated_images(
        self,
        figma_img: Image.Image,
        web_img: Image.Image
    ) -> np.ndarray:
        """
        두 이미지를 높이를 맞춰 수평으로 이어붙임

        Args:
            figma_img: Figma 이미지
            web_img: 웹 이미지

        Returns:
            이어붙인 이미지 (numpy array)
        """
        h = max(figma_img.height, web_img.height)
        wF, wW = figma_img.width, web_img.width

        figma_np = np.array(figma_img)
        web_np = np.array(web_img)

        # 높이가 작은 이미지에 패딩 추가
        if figma_img.height < h:
            pad = np.zeros((h - figma_img.height, wF, 3), dtype=figma_np.dtype)
            figma_np = np.vstack([figma_np, pad])

        if web_img.height < h:
            pad = np.zeros((h - web_img.height, wW, 3), dtype=web_np.dtype)
            web_np = np.vstack([web_np, pad])

        return np.hstack([figma_np, web_np])

    def _draw_single_match(
        self,
        overlay: np.ndarray,
        result: MatchResult,
        color: Tuple[int, int, int],
        idx: int,
        figma_width: int,
        show_error_categories: bool
    ) -> None:
        """
        단일 매칭 결과를 overlay에 그림

        Args:
            overlay: 그릴 이미지 (수정됨)
            result: 매칭 결과
            color: 사용할 색상 (BGR)
            idx: 매칭 인덱스
            figma_width: Figma 이미지 너비
            show_error_categories: 에러 카테고리 표시 여부
        """
        # Figma 박스 좌표
        x1, y1, x2, y2 = map(int, result.figma.extracted.box)
        # Web 박스 좌표
        wx1, wy1, wx2, wy2 = map(int, result.web.box)

        # 박스 그리기
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, self.config.box_linewidth)
        cv2.rectangle(
            overlay,
            (wx1 + figma_width, wy1),
            (wx2 + figma_width, wy2),
            color,
            self.config.box_linewidth
        )

        # 중심점 계산
        cxF, cyF = (x1 + x2) // 2, (y1 + y2) // 2
        cxW, cyW = (wx1 + wx2) // 2 + figma_width, (wy1 + wy2) // 2

        # 연결선 그리기
        cv2.line(
            overlay,
            (cxF, cyF),
            (cxW, cyW),
            color,
            self.config.connection_line_width
        )

        # 텍스트 표시 (매칭 번호 또는 에러 카테고리)
        if self.config.show_match_numbers and not show_error_categories:
            cv2.putText(
                overlay,
                f"#{idx}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            cv2.putText(
                overlay,
                f"#{idx}",
                (wx1 + figma_width, wy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    def _show_visualization(
        self,
        overlay: np.ndarray,
        label: str,
        show_error_categories: bool,
        matches: List[MatchResult],
        match_count: int
    ) -> None:
        """
        시각화 결과를 matplotlib으로 표시

        Args:
            overlay: 시각화할 이미지
            label: 레이블
            show_error_categories: 에러 카테고리 표시 여부
            matches: 매칭 결과 리스트
            match_count: 매칭된 요소 개수
        """
        overlay_rgb = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(self.config.figure_width, self.config.figure_height))
        plt.imshow(overlay_rgb)
        plt.axis('off')

        # 제목 설정
        if show_error_categories and matches:
            error_categories = matches[0].errorCategories if hasattr(matches[0], 'errorCategories') else []
            plt.title(f'Matching Visualization error {error_categories}')
        else:
            plt.title(f'Matching Visualization {label}')

        plt.show()
        print(f"\n[완료] 총 {match_count}개의 요소가 매칭되었습니다.")


# ============================================================================
# Factory Functions
# ============================================================================

def create_visualizer(
    config: Optional[VisualizerConfig] = None
) -> Visualizer:
    """
    Visualizer 인스턴스를 생성하는 팩토리 함수

    Args:
        config: 시각화 설정

    Returns:
        Visualizer 인스턴스
    """
    return Visualizer(config)


# ============================================================================
# Convenience Functions (하위 호환성)
# ============================================================================

_default_visualizer: Optional[Visualizer] = None


def get_default_visualizer() -> Visualizer:
    """기본 Visualizer 인스턴스 반환 (싱글톤 패턴)"""
    global _default_visualizer
    if _default_visualizer is None:
        _default_visualizer = Visualizer()
    return _default_visualizer
