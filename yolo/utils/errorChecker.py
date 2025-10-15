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
    NORMAL,
)


@dataclass(frozen=True)
class ErrorCheckConfig:
    """에러 판별 기준"""

    coordinate_tolerance: int = 10
    size_tolerance: int = 10
    require_text_match: bool = True


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
        return self.check(figma_box, web_box, figma_text, web_text)

    def annotate_match(self, match: MatchResult) -> MatchResult:
        """MatchResult에 에러 정보를 반영한 새 MatchResult를 반환"""
        errors = self.check_match(match)
        return replace(match, errorCategories=errors)


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
