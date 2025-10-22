"""
SimilarityMatcher - 개선된 유사도 기반 매칭 알고리즘

기존 calculate_similarity + get_matches 대신 통합된 find_matches 메서드를 제공합니다.
"""

import logging
from typing import List, Tuple
from PIL import Image
from .models import FigmaFare, ExtractedElement, MatchResult
from .element_matcher import ElementExtractor


class SimilarityMatcher:
    """개선된 유사도 기반 요소 매칭 클래스"""

    def __init__(self, element_extractor: ElementExtractor, min_similarity: float = 0.8):
        """
        Args:
            element_extractor: ElementExtractor 인스턴스 (feature extraction 및 유사도 계산용)
            min_similarity: 매칭으로 간주할 최소 유사도 임계값
        """
        self.extractor = element_extractor
        self.min_similarity = min_similarity
        logging.info(f"SimilarityMatcher initialized with min_similarity={min_similarity}")

    def find_matches(
        self,
        figma_elements: List[FigmaFare],
        web_elements: List[ExtractedElement],
        figma_img: Image.Image = None,
        web_img: Image.Image = None
    ) -> Tuple[List[MatchResult], List[MatchResult], List[MatchResult]]:
        """
        통합된 매칭 메서드 - 유사도 계산과 매칭을 한 번에 수행

        Args:
            figma_elements: Figma 요소 리스트
            web_elements: Web 요소 리스트
            figma_img: Figma 이미지 (좌표 유사도 계산용, optional)
            web_img: Web 이미지 (좌표 유사도 계산용, optional)

        Returns:
            (matches, unmatched_figma, unmatched_web) 튜플
        """
        if not figma_elements or not web_elements:
            logging.warning("Empty figma_elements or web_elements provided to find_matches")
            unmatched_figma = [
                MatchResult(
                    figma=f,
                    web=None,
                    feature_similarity=0.0,
                    text_similarity=0.0,
                    size_similarity=0.0,
                    coordinate_similarity=0.0,
                    score=0.0,
                    errorCategories="Not Matched"
                )
                for f in figma_elements
            ]
            unmatched_web = [
                MatchResult(
                    figma=None,
                    web=w,
                    feature_similarity=0.0,
                    text_similarity=0.0,
                    size_similarity=0.0,
                    coordinate_similarity=0.0,
                    score=0.0,
                    errorCategories="Not Matched"
                )
                for w in web_elements
            ]
            return [], unmatched_figma, unmatched_web

        logging.info(f"Computing similarities for {len(figma_elements)} figma elements and {len(web_elements)} web elements")

        # 1. 유사도 계산 (기존 calculate_similarity 메서드 활용)
        # 이미지가 제공되지 않은 경우 더미 이미지 생성
        if figma_img is None:
            # 첫 번째 Figma 요소의 box 크기를 기반으로 더미 이미지 생성
            figma_box = figma_elements[0].extracted.box
            figma_img = Image.new('RGB', (int(figma_box[2]), int(figma_box[3])), (255, 255, 255))
        if web_img is None:
            # 첫 번째 Web 요소의 box 크기를 기반으로 더미 이미지 생성
            web_box = web_elements[0].box
            web_img = Image.new('RGB', (int(web_box[2]), int(web_box[3])), (255, 255, 255))

        sim_dict = self.extractor.calculate_similarity(
            figma_img, web_img, figma_elements, web_elements
        )

        # 2. 매칭 수행 (기존 get_matches 메서드 활용)
        matches, unmatched_figma, unmatched_web = self.extractor.get_matches(
            sim_dict, figma_elements, web_elements, self.min_similarity
        )

        logging.info(f"Found {len(matches)} matches, {len(unmatched_figma)} unmatched figma, {len(unmatched_web)} unmatched web")

        return matches, unmatched_figma, unmatched_web


def create_similarity_matcher(min_similarity: float = 0.6) -> SimilarityMatcher:
    """
    SimilarityMatcher 인스턴스를 생성하는 팩토리 함수

    Args:
        min_similarity: 매칭으로 간주할 최소 유사도 임계값

    Returns:
        SimilarityMatcher 인스턴스
    """
    logging.info(f"Creating SimilarityMatcher with min_similarity={min_similarity}")

    # ElementExtractor 인스턴스 생성
    element_extractor = ElementExtractor(resize_size=(736, 736))

    # SimilarityMatcher 인스턴스 생성
    matcher = SimilarityMatcher(element_extractor, min_similarity)

    return matcher
