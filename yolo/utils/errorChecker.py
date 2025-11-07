"""
Error checking utilities for match results.

레거시 함수 기반 구현을 ErrorManager 클래스로 통합하여
일관된 에러 평가 로직을 제공합니다.
"""

from dataclasses import dataclass, replace
from typing import List, Optional

from ..core.models import MatchResult
from .error_list import (
    G_ERROR_COORDINATE_X,
    G_ERROR_COORDINATE_Y,
    G_ERROR_NOT_MATCHED,
    G_ERROR_SIZE,
    G_ERROR_TEXT,
    G_ERROR_ASPECT_RATIO,
    G_ERROR_FEATURE_DISSIMILAR,
    G_ERROR_LOW_OVERALL_SCORE,
    G_ERROR_RELATIVE_POSITION,
    NORMAL,
    detail_coordinate_x,
    detail_coordinate_y,
    detail_size,
    detail_aspect_ratio,
    detail_text_difference,
    detail_feature_similarity,
    detail_overall_score,
    detail_relative_position,
)


@dataclass(frozen=True)
class ErrorCheckConfig:
    """에러 판별 기준"""

    coordinate_tolerance: int = 10
    size_tolerance: int = 10
    require_text_match: bool = True
    aspect_ratio_tolerance: float = 0.2  # 가로세로 비율 오차 허용 범위 (20%)
    min_feature_similarity: float = 0.3  # 최소 feature 유사도
    min_overall_score: float = 0.4  # 최소 전체 스코어 (이보다 낮으면 경고)
    relative_position_tolerance: float = 0.1  # 상대 좌표 오차 허용 범위 (10%)


class ErrorManager:
    """매칭 결과의 품질을 평가하는 클래스"""

    def __init__(self, config: Optional[ErrorCheckConfig] = None):
        self.config = config or ErrorCheckConfig()

    def _within_tolerance(self, diff: float) -> bool:
        return diff <= self.config.coordinate_tolerance

    def _size_within_tolerance(self, diff_w: float, diff_h: float) -> bool:
        return diff_w <= self.config.size_tolerance and diff_h <= self.config.size_tolerance

    def _text_matches(self, figma_text: str, web_text: str) -> bool:
        if not self.config.require_text_match:
            return True
        return figma_text == web_text

    def check(
        self,
        figma_box: List[int],
        web_box: List[int],
        figma_text: str,
        web_text: str
    ) -> List[str]:
        """박스와 텍스트 정보를 기반으로 에러 리스트 반환"""
        errors: List[str] = []

        if not self._within_tolerance(abs(figma_box[0] - web_box[0])):
            errors.append(G_ERROR_COORDINATE_X)

        if not self._within_tolerance(abs(figma_box[1] - web_box[1])):
            errors.append(G_ERROR_COORDINATE_Y)

        figma_w = abs(figma_box[2] - figma_box[0])
        figma_h = abs(figma_box[3] - figma_box[1])
        web_w = abs(web_box[2] - web_box[0])
        web_h = abs(web_box[3] - web_box[1])

        if not self._size_within_tolerance(abs(figma_w - web_w), abs(figma_h - web_h)):
            errors.append(G_ERROR_SIZE)

        if not self._text_matches(figma_text, web_text):
            errors.append(G_ERROR_TEXT)

        return errors or [NORMAL]

    def check_match(self, match: MatchResult) -> List[str]:
        """MatchResult를 기반으로 에러 리스트 반환"""
        if match.figma is None or match.web is None:
            return [G_ERROR_NOT_MATCHED]

        figma_box = match.figma.extracted.box
        web_box = match.web.box
        figma_text = getattr(match.figma.extracted, "text", "")
        web_text = getattr(match.web, "text", "")

        # 기본 에러 체크 (좌표, 크기, 텍스트)
        errors = self.check(figma_box, web_box, figma_text, web_text)

        # 정상이면 추가 에러 체크
        if errors == [NORMAL]:
            errors = []

        # 가로세로 비율 체크
        figma_w = abs(figma_box[2] - figma_box[0])
        figma_h = abs(figma_box[3] - figma_box[1])
        web_w = abs(web_box[2] - web_box[0])
        web_h = abs(web_box[3] - web_box[1])

        figma_ratio = figma_w / max(figma_h, 1)
        web_ratio = web_w / max(web_h, 1)
        ratio_diff = abs(figma_ratio - web_ratio) / max(figma_ratio, 0.01)

        if ratio_diff > self.config.aspect_ratio_tolerance:
            errors.append(G_ERROR_ASPECT_RATIO)

        # Feature 유사도 체크
        if match.feature_similarity < self.config.min_feature_similarity:
            errors.append(G_ERROR_FEATURE_DISSIMILAR)

        # 전체 스코어 체크
        if match.score < self.config.min_overall_score:
            errors.append(G_ERROR_LOW_OVERALL_SCORE)

        # 상대 좌표 체크 (중심점 기준)
        figma_cx = (figma_box[0] + figma_box[2]) / 2
        figma_cy = (figma_box[1] + figma_box[3]) / 2
        web_cx = (web_box[0] + web_box[2]) / 2
        web_cy = (web_box[1] + web_box[3]) / 2

        # 화면 크기로 정규화 (임시로 1920x1080 가정)
        screen_w, screen_h = 1920, 1080
        rel_diff_x = abs(figma_cx / screen_w - web_cx / screen_w)
        rel_diff_y = abs(figma_cy / screen_h - web_cy / screen_h)

        if rel_diff_x > self.config.relative_position_tolerance or rel_diff_y > self.config.relative_position_tolerance:
            errors.append(G_ERROR_RELATIVE_POSITION)

        return errors or [NORMAL]

    def annotate_match(self, match: MatchResult) -> MatchResult:
        """MatchResult에 에러 정보를 반영한 새 MatchResult를 반환"""
        errors = self.check_match(match)
        return replace(match, errorCategories=errors)

    def get_detail_info(self, match: MatchResult) -> str:
        """매칭 결과의 상세 정보를 문자열로 반환"""
        if match.figma is None or match.web is None:
            return "매칭되지 않은 요소입니다"

        figma_box = match.figma.extracted.box
        web_box = match.web.box

        details = []

        # X 좌표 차이
        x_diff = figma_box[0] - web_box[0]
        if abs(x_diff) > self.config.coordinate_tolerance:
            details.append(detail_coordinate_x(x_diff))

        # Y 좌표 차이
        y_diff = figma_box[1] - web_box[1]
        if abs(y_diff) > self.config.coordinate_tolerance:
            details.append(detail_coordinate_y(y_diff))

        # 크기 차이
        figma_w = abs(figma_box[2] - figma_box[0])
        figma_h = abs(figma_box[3] - figma_box[1])
        web_w = abs(web_box[2] - web_box[0])
        web_h = abs(web_box[3] - web_box[1])

        w_diff = figma_w - web_w
        h_diff = figma_h - web_h
        if abs(w_diff) > self.config.size_tolerance or abs(h_diff) > self.config.size_tolerance:
            # 큰 차이를 먼저 보여줌
            if abs(w_diff) > abs(h_diff):
                details.append(f"너비가 {abs(w_diff):.1f}px {'크게' if w_diff > 0 else '작게'} 되어 있습니다")
            else:
                details.append(f"높이가 {abs(h_diff):.1f}px {'크게' if h_diff > 0 else '작게'} 되어 있습니다")

        # 가로세로 비율 차이
        figma_ratio = figma_w / max(figma_h, 1)
        web_ratio = web_w / max(web_h, 1)
        ratio_diff = abs(figma_ratio - web_ratio) / max(figma_ratio, 0.01)
        if ratio_diff > self.config.aspect_ratio_tolerance:
            details.append(detail_aspect_ratio(figma_ratio, web_ratio))

        # 텍스트 차이
        figma_text = getattr(match.figma.extracted, "text", "")
        web_text = getattr(match.web, "text", "")
        if self.config.require_text_match and figma_text and web_text and figma_text != web_text:
            details.append(detail_text_difference(figma_text, web_text))

        # 시각적 특징 유사도
        if match.feature_similarity < self.config.min_feature_similarity:
            details.append(detail_feature_similarity(match.feature_similarity))

        # 전체 매칭 스코어
        if match.score < self.config.min_overall_score:
            details.append(detail_overall_score(match.score))

        # 상대 좌표 오차
        figma_cx = (figma_box[0] + figma_box[2]) / 2
        figma_cy = (figma_box[1] + figma_box[3]) / 2
        web_cx = (web_box[0] + web_box[2]) / 2
        web_cy = (web_box[1] + web_box[3]) / 2

        # 화면 크기로 정규화 (임시로 1920x1080 가정)
        screen_w, screen_h = 1920, 1080
        rel_diff_x = abs(figma_cx / screen_w - web_cx / screen_w)
        rel_diff_y = abs(figma_cy / screen_h - web_cy / screen_h)

        if rel_diff_x > self.config.relative_position_tolerance or rel_diff_y > self.config.relative_position_tolerance:
            details.append(detail_relative_position(rel_diff_x, rel_diff_y))

        return ", ".join(details) if details else ""


class ErrorChecker(ErrorManager):
    """
    레거시 호환용 클래스

    기존 코드에서 사용하던 메서드명을 유지하면서 내부 구현은 ErrorManager를 사용합니다.
    """

    def __init__(self, config: Optional[ErrorCheckConfig] = None):
        super().__init__(config=config)

    def check_error(
        self,
        figma_box: List[int],
        web_box: List[int],
        figma_text: str,
        web_text: str
    ) -> List[str]:
        return super().check(figma_box, web_box, figma_text, web_text)

    def check_error_by_match(self, match: MatchResult) -> List[str]:
        return super().check_match(match)
