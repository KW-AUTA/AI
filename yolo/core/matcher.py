"""
SimilarityMatcher 클래스 - 유사도 계산 및 매칭 전담
- 단일 책임 원칙 적용
- 전략 패턴으로 유사도 계산 방식 분리
- 설정 기반 동적 파라미터 조정
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Protocol
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .models import ExtractedElement, FigmaFare, MatchResult, SimilarityConfig, SimilarityWeights
from ..utils.error_list import G_ERROR_NOT_MATCHED


class SimilarityStrategy(ABC):
    """유사도 계산 전략 인터페이스"""
    
    @abstractmethod
    def calculate_similarity(
        self, 
        figma_elements: List[FigmaFare], 
        web_elements: List[ExtractedElement]
    ) -> np.ndarray:
        """유사도 행렬 계산"""
        pass


class TextSimilarityStrategy(SimilarityStrategy):
    """텍스트 유사도 계산 전략"""
    
    def calculate_similarity(
        self, 
        figma_elements: List[FigmaFare], 
        web_elements: List[ExtractedElement]
    ) -> np.ndarray:
        from difflib import SequenceMatcher
        
        N, M = len(figma_elements), len(web_elements)
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)
        
        similarity_matrix = np.zeros((N, M), dtype=np.float32)
        
        for i, figma in enumerate(figma_elements):
            figma_text = getattr(figma.extracted, 'text', '') or ''
            
            for j, web in enumerate(web_elements):
                web_text = getattr(web, 'text', '') or ''
                
                if not figma_text or not web_text:
                    similarity_matrix[i, j] = 0.0
                else:
                    similarity = SequenceMatcher(None, figma_text, web_text).ratio()
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix


class FeatureSimilarityStrategy(SimilarityStrategy):
    """특징 벡터 유사도 계산 전략"""
    
    def calculate_similarity(
        self, 
        figma_elements: List[FigmaFare], 
        web_elements: List[ExtractedElement]
    ) -> np.ndarray:
        N, M = len(figma_elements), len(web_elements)
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)
        
        # 특징 벡터 추출
        figma_features = []
        web_features = []
        
        for figma in figma_elements:
            feature = getattr(figma.extracted, 'feature', None)
            if feature is not None:
                if isinstance(feature, torch.Tensor):
                    feature = feature.cpu().numpy()
                figma_features.append(feature.flatten())
            else:
                figma_features.append(np.zeros(512))
        
        for web in web_elements:
            feature = getattr(web, 'feature', None)
            if feature is not None:
                if isinstance(feature, torch.Tensor):
                    feature = feature.cpu().numpy()
                web_features.append(feature.flatten())
            else:
                web_features.append(np.zeros(512))
        
        figma_features = np.array(figma_features)
        web_features = np.array(web_features)
        
        # 코사인 유사도 계산
        similarity_matrix = self._cosine_similarity(figma_features, web_features)
        
        return similarity_matrix
    
    def _cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """코사인 유사도 계산

        정규화된 벡터 간의 코사인 유사도는 [-1, 1] 범위를 가지며:
        - 1.0: 완전히 동일한 방향 (매우 유사)
        - 0.0: 직교 (무관)
        - -1.0: 정반대 방향 (매우 다름)

        음수 값은 유사하지 않음을 의미하므로 0으로 클리핑합니다.
        """
        # 정규화
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)

        # 내적으로 코사인 유사도 계산
        similarity = np.dot(A_norm, B_norm.T)

        # 음수 값을 0으로 클리핑 (유사하지 않은 경우)
        # 이제 유사한 이미지는 0.7~1.0, 다른 이미지는 0.0~0.2 범위를 가짐
        similarity = np.clip(similarity, 0.0, 1.0)

        return similarity.astype(np.float32)


class SizeSimilarityStrategy(SimilarityStrategy):
    """크기 유사도 계산 전략"""
    
    def calculate_similarity(
        self, 
        figma_elements: List[FigmaFare], 
        web_elements: List[ExtractedElement]
    ) -> np.ndarray:
        N, M = len(figma_elements), len(web_elements)
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)
        
        similarity_matrix = np.zeros((N, M), dtype=np.float32)
        
        for i, figma in enumerate(figma_elements):
            figma_box = getattr(figma.extracted, 'box', figma.box)
            figma_w = figma_box[2] - figma_box[0]
            figma_h = figma_box[3] - figma_box[1]
            figma_area = figma_w * figma_h
            
            for j, web in enumerate(web_elements):
                web_box = web.box
                web_w = web_box[2] - web_box[0]
                web_h = web_box[3] - web_box[1]
                web_area = web_w * web_h
                
                if figma_area == 0 or web_area == 0:
                    similarity_matrix[i, j] = 0.0
                else:
                    # 면적 비율 기반 유사도
                    area_ratio = min(figma_area, web_area) / max(figma_area, web_area)
                    
                    # 종횡비 유사도
                    figma_aspect = figma_w / max(figma_h, 1)
                    web_aspect = web_w / max(web_h, 1)
                    aspect_ratio = min(figma_aspect, web_aspect) / max(figma_aspect, web_aspect)
                    
                    # 종합 크기 유사도
                    similarity = (area_ratio + aspect_ratio) / 2
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix


class CoordinateSimilarityStrategy(SimilarityStrategy):
    """좌표 유사도 계산 전략"""
    
    def calculate_similarity(
        self, 
        figma_elements: List[FigmaFare], 
        web_elements: List[ExtractedElement]
    ) -> np.ndarray:
        N, M = len(figma_elements), len(web_elements)
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)
        
        similarity_matrix = np.zeros((N, M), dtype=np.float32)
        
        # 화면 크기 추정 (최대 좌표 기반)
        all_boxes = []
        for figma in figma_elements:
            all_boxes.append(getattr(figma.extracted, 'box', figma.box))
        for web in web_elements:
            all_boxes.append(web.box)
        
        if not all_boxes:
            return similarity_matrix
        
        all_boxes = np.array(all_boxes)
        screen_w = np.max(all_boxes[:, 2])
        screen_h = np.max(all_boxes[:, 3])
        screen_diagonal = np.sqrt(screen_w**2 + screen_h**2)
        
        for i, figma in enumerate(figma_elements):
            figma_box = getattr(figma.extracted, 'box', figma.box)
            figma_center = np.array([
                (figma_box[0] + figma_box[2]) / 2,
                (figma_box[1] + figma_box[3]) / 2
            ])
            
            for j, web in enumerate(web_elements):
                web_box = web.box
                web_center = np.array([
                    (web_box[0] + web_box[2]) / 2,
                    (web_box[1] + web_box[3]) / 2
                ])
                
                # 중심점 간 거리
                distance = np.linalg.norm(figma_center - web_center)
                
                # 화면 대각선 대비 정규화된 거리
                normalized_distance = distance / max(screen_diagonal, 1)
                
                # 유사도 (거리가 가까울수록 높음)
                similarity = np.exp(-normalized_distance * 2)  # 지수 감소
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix


class MatrixNormalizer:
    """행렬 정규화 처리"""
    
    @staticmethod
    def quantile_normalize(
        matrix: np.ndarray, 
        q_low: float = 0.05, 
        q_high: float = 0.95
    ) -> np.ndarray:
        """분위수 기반 행별 정규화"""
        if matrix.size == 0:
            return matrix.astype(np.float32)
        
        low = np.quantile(matrix, q_low, axis=1, keepdims=True)
        high = np.quantile(matrix, q_high, axis=1, keepdims=True)
        range_ = np.clip(high - low, 1e-6, None)
        mat_clipped = np.clip(matrix, low, high)
        
        return ((mat_clipped - low) / range_).astype(np.float32)
    
    @staticmethod
    def softmax_normalize(matrix: np.ndarray, tau: float = 0.25) -> np.ndarray:
        """Softmax 기반 행별 정규화"""
        if matrix.size == 0:
            return matrix.astype(np.float32)
        
        # 수치 안정성을 위한 행별 최대값 제거
        row_max = np.max(matrix, axis=1, keepdims=True)
        exp = np.exp((matrix - row_max) / max(tau, 1e-6))
        sum_exp = np.sum(exp, axis=1, keepdims=True) + 1e-9
        
        return (exp / sum_exp).astype(np.float32)


@dataclass
class SimilarityResult:
    """유사도 계산 결과"""
    text_matrix: np.ndarray
    feature_matrix: np.ndarray
    size_matrix: np.ndarray
    coordinate_matrix: np.ndarray
    combined_relative: np.ndarray
    combined_absolute: np.ndarray


class SimilarityMatcher:
    """유사도 기반 매칭 클래스"""
    
    def __init__(
        self,
        config: Optional[SimilarityConfig] = None,
        text_strategy: Optional[SimilarityStrategy] = None,
        feature_strategy: Optional[SimilarityStrategy] = None,
        size_strategy: Optional[SimilarityStrategy] = None,
        coordinate_strategy: Optional[SimilarityStrategy] = None
    ):
        self.config = config or SimilarityConfig.from_env()
        self.weights = self.config.weights
        
        # 전략 객체들
        self.text_strategy = text_strategy or TextSimilarityStrategy()
        self.feature_strategy = feature_strategy or FeatureSimilarityStrategy()
        self.size_strategy = size_strategy or SizeSimilarityStrategy()
        self.coordinate_strategy = coordinate_strategy or CoordinateSimilarityStrategy()
        
        self.normalizer = MatrixNormalizer()
        self.logger = logging.getLogger(__name__)
    
    def calculate_similarities(
        self,
        figma_elements: List[FigmaFare],
        web_elements: List[ExtractedElement]
    ) -> SimilarityResult:
        """모든 유사도 행렬 계산"""
        
        # 개별 유사도 계산
        text_matrix = self.text_strategy.calculate_similarity(figma_elements, web_elements)
        feature_matrix = self.feature_strategy.calculate_similarity(figma_elements, web_elements)
        size_matrix = self.size_strategy.calculate_similarity(figma_elements, web_elements)
        coordinate_matrix = self.coordinate_strategy.calculate_similarity(figma_elements, web_elements)
        
        # 정규화
        if self.config.use_softmax_relative:
            text_rel = self.normalizer.softmax_normalize(text_matrix, self.config.softmax_tau)
            feature_rel = self.normalizer.softmax_normalize(feature_matrix, self.config.softmax_tau)
            size_rel = self.normalizer.softmax_normalize(size_matrix, self.config.softmax_tau)
            coord_rel = self.normalizer.softmax_normalize(coordinate_matrix, self.config.softmax_tau)
        else:
            text_rel = self.normalizer.quantile_normalize(text_matrix, self.weights.QUANTILE_LOW, self.weights.QUANTILE_HIGH)
            feature_rel = self.normalizer.quantile_normalize(feature_matrix, self.weights.QUANTILE_LOW, self.weights.QUANTILE_HIGH)
            size_rel = self.normalizer.quantile_normalize(size_matrix, self.weights.QUANTILE_LOW, self.weights.QUANTILE_HIGH)
            coord_rel = self.normalizer.quantile_normalize(coordinate_matrix, self.weights.QUANTILE_LOW, self.weights.QUANTILE_HIGH)
        
        # 절대값 정규화 (클리핑)
        text_abs = text_matrix.astype(np.float32)
        feature_abs = np.clip(feature_matrix, 0.0, 1.0).astype(np.float32)
        size_abs = np.clip(size_matrix, 0.0, 1.0).astype(np.float32)
        coord_abs = np.clip(coordinate_matrix, 0.0, 1.0).astype(np.float32)
        
        # 가중치 계산
        weight_matrices = self._calculate_weights(
            figma_elements, web_elements, text_abs
        )
        
        # 결합된 유사도 계산
        combined_rel = self._combine_similarities(
            text_rel, feature_rel, size_rel, coord_rel, weight_matrices
        )
        
        combined_abs = self._combine_similarities(
            text_abs, feature_abs, size_abs, coord_abs, weight_matrices
        )
        
        return SimilarityResult(
            text_matrix=text_matrix,
            feature_matrix=feature_matrix,
            size_matrix=size_matrix,
            coordinate_matrix=coordinate_matrix,
            combined_relative=combined_rel,
            combined_absolute=combined_abs
        )
    
    def _calculate_weights(
        self,
        figma_elements: List[FigmaFare],
        web_elements: List[ExtractedElement],
        text_similarity: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """텍스트 기반 동적 가중치 계산

        핵심 전략:
        1. 둘 다 텍스트가 있고 유사도가 낮으면 -> 텍스트 가중치 대폭 증가 (페널티 효과)
        2. 둘 다 텍스트가 있고 유사도가 높으면 -> 텍스트 가중치 증가
        3. 텍스트가 없으면 -> 텍스트 가중치 최소화, 다른 특징에 의존
        """
        N, M = len(figma_elements), len(web_elements)

        if N == 0 or M == 0:
            return {
                'text': np.array([]),
                'feature': np.array([]),
                'size': np.array([]),
                'coordinate': np.array([])
            }

        # 텍스트 존재 여부 확인
        figma_has_text = np.array([
            bool(getattr(f.extracted, 'text', '') or '')
            for f in figma_elements
        ], dtype=bool)

        web_has_text = np.array([
            bool(getattr(w, 'text', '') or '')
            for w in web_elements
        ], dtype=bool)

        both_have_text = np.outer(figma_has_text, web_has_text)

        # 기본 가중치 초기화
        w_text = np.full((N, M), self.weights.TEXT_WITHOUT, dtype=np.float32)
        w_feature = np.full((N, M), self.weights.FEATURE_BASE, dtype=np.float32)
        w_size = np.full((N, M), self.weights.SIZE_BASE, dtype=np.float32)
        w_coord = np.full((N, M), self.weights.COORDINATE_BASE, dtype=np.float32)

        # 둘 다 텍스트가 있는 경우: 텍스트 유사도에 따라 가중치 조정
        if both_have_text.any():
            # 텍스트 유사도가 낮으면 텍스트 가중치를 매우 높게 (페널티)
            # 텍스트 유사도가 높으면 적당히 높게
            # 0.0~0.3: 매우 높은 가중치 (0.7) -> 낮은 점수로 페널티
            # 0.3~0.7: 높은 가중치 (0.5)
            # 0.7~1.0: 적당한 가중치 (0.4)
            text_weight_both = np.where(
                text_similarity < 0.3,
                0.7,  # 텍스트가 매우 다르면 텍스트 가중치 70%
                np.where(
                    text_similarity < 0.7,
                    0.5,  # 텍스트가 다르면 가중치 50%
                    0.4   # 텍스트가 유사하면 가중치 40%
                )
            )

            w_text[both_have_text] = text_weight_both[both_have_text]

            # 텍스트 가중치가 증가한 만큼 다른 가중치 감소
            w_feature[both_have_text] = (1.0 - text_weight_both[both_have_text]) * 0.5
            w_size[both_have_text] = (1.0 - text_weight_both[both_have_text]) * 0.25
            w_coord[both_have_text] = (1.0 - text_weight_both[both_have_text]) * 0.25

        return {
            'text': w_text,
            'feature': w_feature,
            'size': w_size,
            'coordinate': w_coord
        }
    
    def _combine_similarities(
        self,
        text_sim: np.ndarray,
        feature_sim: np.ndarray,
        size_sim: np.ndarray,
        coord_sim: np.ndarray,
        weights: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """가중 평균으로 유사도 결합

        가중치는 이미 합이 1.0이 되도록 설정되어 있으므로
        (TEXT=0.3, FEATURE=0.4, SIZE=0.15, COORD=0.15)
        가중치 합으로 나누지 않고 직접 가중합을 사용합니다.
        이렇게 하면 개별 유사도가 높을 때 최종 점수도 높게 유지됩니다.
        """
        if text_sim.size == 0:
            return text_sim

        w_text = weights['text']
        w_feature = weights['feature']
        w_size = weights['size']
        w_coord = weights['coordinate']

        # 동적 가중치로 인해 합이 1.0이 아닐 수 있으므로 정규화
        w_sum = w_text + w_feature + w_size + w_coord + 1e-8

        # 정규화된 가중치로 가중합 계산
        combined = (
            text_sim * (w_text / w_sum) +
            feature_sim * (w_feature / w_sum) +
            size_sim * (w_size / w_sum) +
            coord_sim * (w_coord / w_sum)
        )

        return combined.astype(np.float32)
    
    def find_matches(
        self,
        figma_elements: List[FigmaFare],
        web_elements: List[ExtractedElement],
        min_similarity: Optional[float] = None
    ) -> Tuple[List[MatchResult], List[MatchResult], List[MatchResult]]:
        """매칭 수행"""

        min_similarity = min_similarity or self.weights.MIN_SIMILARITY

        # 유사도 계산
        sim_result = self.calculate_similarities(figma_elements, web_elements)

        # 디버그 시각화
        if self.config.debug_mode:
            self._save_debug_visualizations(sim_result)

        # 헝가리안 알고리즘으로 최적 매칭 (개별 유사도도 전달)
        matches = self._hungarian_matching(
            sim_result.combined_relative,
            sim_result.combined_absolute,
            sim_result,  # 개별 유사도도 전달
            figma_elements,
            web_elements,
            min_similarity
        )

        return matches
    
    def _hungarian_matching(
        self,
        relative_similarity: np.ndarray,
        absolute_similarity: np.ndarray,
        sim_result: SimilarityResult,
        figma_elements: List[FigmaFare],
        web_elements: List[ExtractedElement],
        min_similarity: float
    ) -> Tuple[List[MatchResult], List[MatchResult], List[MatchResult]]:
        """헝가리안 알고리즘으로 최적 매칭"""

        N, M = len(figma_elements), len(web_elements)

        if N == 0 or M == 0:
            unmatched_figma = [
                MatchResult(
                    figma=figma, web=None,
                    feature_similarity=0.0, text_similarity=0.0,
                    size_similarity=0.0, coordinate_similarity=0.0,
                    score=0.0, errorCategories=[G_ERROR_NOT_MATCHED]
                ) for figma in figma_elements
            ]
            unmatched_web = [
                MatchResult(
                    figma=None, web=web,
                    feature_similarity=0.0, text_similarity=0.0,
                    size_similarity=0.0, coordinate_similarity=0.0,
                    score=0.0, errorCategories=[G_ERROR_NOT_MATCHED]
                ) for web in web_elements
            ]
            return [], unmatched_figma, unmatched_web

        # 비용 행렬 (1 - 유사도)
        cost_matrix = 1.0 - relative_similarity

        # 헝가리안 알고리즘 실행
        figma_indices, web_indices = linear_sum_assignment(cost_matrix)

        matches = []
        matched_figma = set()
        matched_web = set()

        # 매칭 결과 처리
        for figma_idx, web_idx in zip(figma_indices, web_indices):
            absolute_score = absolute_similarity[figma_idx, web_idx]

            # 전체 점수 임계값 확인
            if absolute_score < min_similarity:
                continue

            # 개별 유사도 추출
            feat_sim = float(sim_result.feature_matrix[figma_idx, web_idx])
            text_sim = float(sim_result.text_matrix[figma_idx, web_idx])
            size_sim = float(sim_result.size_matrix[figma_idx, web_idx])
            coord_sim = float(sim_result.coordinate_matrix[figma_idx, web_idx])

            # === 개별 유사도 최소 임계값 (Hard Constraints) ===
            # 환경변수로 설정 가능, 기본값 지정
            MIN_SIZE_SIMILARITY = float(os.environ.get('MIN_SIZE_SIM', '0.30'))
            MIN_TEXT_SIMILARITY_BOTH = float(os.environ.get('MIN_TEXT_SIM_BOTH', '0.50'))

            # 텍스트 존재 여부 확인
            figma_text = getattr(figma_elements[figma_idx].extracted, 'text', '') or ''
            web_text = getattr(web_elements[web_idx], 'text', '') or ''
            both_have_text = bool(figma_text.strip()) and bool(web_text.strip())

            # 1. 크기 유사도 체크 (모든 경우)
            if size_sim < MIN_SIZE_SIMILARITY:
                self.logger.info(f"✗ Rejected by size: figma={figma_elements[figma_idx].name[:30]} web={getattr(web_elements[web_idx], 'text', 'no-text')[:30]} size_sim={size_sim:.2f}")
                continue

            # 2. 텍스트 유사도 체크 (둘 다 텍스트가 있는 경우)
            if both_have_text and text_sim < MIN_TEXT_SIMILARITY_BOTH:
                self.logger.info(f"✗ Rejected by text: figma='{figma_text[:30]}' web='{web_text[:30]}' text_sim={text_sim:.2f}")
                continue

            # 3. XOR 케이스: 한쪽만 텍스트가 있는 경우 (OCR 실패 보호)
            figma_has = bool(figma_text.strip())
            web_has = bool(web_text.strip())
            if figma_has ^ web_has:  # XOR: 한쪽만 True
                longer_text = figma_text if figma_has else web_text
                MIN_FEAT_SIM_XOR = float(os.environ.get('MIN_FEAT_SIM_XOR', '0.85'))

                # 텍스트가 3글자 이상이고 feature 유사도가 낮으면 거부
                if len(longer_text.strip()) >= 3 and feat_sim < MIN_FEAT_SIM_XOR:
                    self.logger.info(f"✗ Rejected by XOR text: one_side_text='{longer_text[:30]}' feat_sim={feat_sim:.2f} (< {MIN_FEAT_SIM_XOR})")
                    continue

            # 디버그: 매칭 승인 로그
            if both_have_text:
                self.logger.info(f"✓ Matched (both_text): figma='{figma_text[:30]}' web='{web_text[:30]}' text_sim={text_sim:.2f} score={absolute_score:.2f}")
            else:
                self.logger.debug(f"✓ Matched (no_text): figma={figma_elements[figma_idx].name[:30]} score={absolute_score:.2f}")

            # 모든 임계값 통과 -> 매칭 승인
            match = MatchResult(
                figma=figma_elements[figma_idx],
                web=web_elements[web_idx],
                feature_similarity=feat_sim,
                text_similarity=text_sim,
                size_similarity=size_sim,
                coordinate_similarity=coord_sim,
                score=float(absolute_score),
                errorCategories=[]
            )
            matches.append(match)
            matched_figma.add(figma_idx)
            matched_web.add(web_idx)
        
        # 매칭되지 않은 요소들
        unmatched_figma = [
            MatchResult(
                figma=figma_elements[i], web=None,
                feature_similarity=0.0, text_similarity=0.0,
                size_similarity=0.0, coordinate_similarity=0.0,
                score=0.0, errorCategories=[G_ERROR_NOT_MATCHED]
            ) for i in range(N) if i not in matched_figma
        ]
        
        unmatched_web = [
            MatchResult(
                figma=None, web=web_elements[j],
                feature_similarity=0.0, text_similarity=0.0,
                size_similarity=0.0, coordinate_similarity=0.0,
                score=0.0, errorCategories=[G_ERROR_NOT_MATCHED]
            ) for j in range(M) if j not in matched_web
        ]
        
        self.logger.info(f"Found {len(matches)} matches, "
                        f"{len(unmatched_figma)} unmatched Figma, "
                        f"{len(unmatched_web)} unmatched Web elements")
        
        return matches, unmatched_figma, unmatched_web
    
    def _save_debug_visualizations(self, sim_result: SimilarityResult) -> None:
        """디버그 시각화 저장"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            debug_dir = Path(__file__).parent / 'debug_sim'
            debug_dir.mkdir(exist_ok=True)
            
            def save_heatmap(matrix: np.ndarray, title: str, filename: str):
                plt.figure(figsize=(10, 7))
                sns.heatmap(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
                plt.title(title)
                plt.xlabel("Web Elements")
                plt.ylabel("Figma Elements")
                plt.tight_layout()
                plt.savefig(debug_dir / filename, dpi=150)
                plt.close()
            
            # 각 유사도 행렬 저장
            save_heatmap(sim_result.text_matrix, 'Text Similarity', 'text_similarity.png')
            save_heatmap(sim_result.feature_matrix, 'Feature Similarity', 'feature_similarity.png')
            save_heatmap(sim_result.size_matrix, 'Size Similarity', 'size_similarity.png')
            save_heatmap(sim_result.coordinate_matrix, 'Coordinate Similarity', 'coordinate_similarity.png')
            save_heatmap(sim_result.combined_absolute, 'Combined Absolute', 'combined_absolute.png')
            save_heatmap(sim_result.combined_relative, 'Combined Relative', 'combined_relative.png')
            
        except Exception as e:
            self.logger.warning(f"Failed to save debug visualizations: {e}")


# 팩토리 함수
def create_similarity_matcher(config: Optional[SimilarityConfig] = None) -> SimilarityMatcher:
    """SimilarityMatcher 생성"""
    return SimilarityMatcher(config=config)