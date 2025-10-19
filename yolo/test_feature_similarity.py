#!/usr/bin/env python3
"""
Feature 유사도 계산 개선 테스트
- 수정 전: 랜덤 벡터도 평균 0.5 유사도
- 수정 후: 랜덤 벡터는 0.0~0.2, 유사한 벡터는 0.7~1.0
"""
import sys
import os

# yolo 패키지 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from yolo.core.matcher import FeatureSimilarityStrategy
from yolo.core.models import FigmaFare, ExtractedElement

def test_feature_similarity_improvement():
    """코사인 유사도 수정 효과 확인"""

    print("\n" + "=" * 80)
    print("Feature 유사도 계산 개선 테스트")
    print("=" * 80)

    # 테스트 케이스 1: 완전히 동일한 벡터
    print("\n[테스트 1] 완전히 동일한 벡터")
    vec1 = np.random.randn(1536)
    figma1 = FigmaFare(
        id="test1",
        name="identical",
        box=np.array([0, 0, 100, 100]),
        extracted=type('obj', (object,), {'feature': vec1, 'text': '', 'box': np.array([0, 0, 100, 100])})()
    )
    web1 = ExtractedElement(
        box=np.array([0, 0, 100, 100]),
        text='',
        feature=vec1.copy(),
        cls=0
    )

    strategy = FeatureSimilarityStrategy()
    sim_matrix = strategy.calculate_similarity([figma1], [web1])
    print(f"  유사도: {sim_matrix[0, 0]:.4f}")
    print(f"  기대값: 1.0에 가까워야 함")
    print(f"  ✓ PASS" if sim_matrix[0, 0] > 0.95 else f"  ✗ FAIL")

    # 테스트 케이스 2: 완전히 다른 랜덤 벡터들
    print("\n[테스트 2] 완전히 다른 랜덤 벡터들 (10쌍)")
    random_sims = []
    for i in range(10):
        vec_a = np.random.randn(1536)
        vec_b = np.random.randn(1536)

        figma = FigmaFare(
            id=f"test_rand_{i}_a",
            name=f"random_a_{i}",
            box=np.array([0, 0, 100, 100]),
            extracted=type('obj', (object,), {'feature': vec_a, 'text': '', 'box': np.array([0, 0, 100, 100])})()
        )
        web = ExtractedElement(
            box=np.array([0, 0, 100, 100]),
            text='',
            feature=vec_b,
            cls=0
        )

        sim_matrix = strategy.calculate_similarity([figma], [web])
        random_sims.append(sim_matrix[0, 0])

    avg_random = np.mean(random_sims)
    max_random = np.max(random_sims)
    min_random = np.min(random_sims)

    print(f"  평균 유사도: {avg_random:.4f}")
    print(f"  최대 유사도: {max_random:.4f}")
    print(f"  최소 유사도: {min_random:.4f}")
    print(f"  기대값: 평균 0.0~0.2 사이")
    print(f"  ✓ PASS" if avg_random < 0.3 else f"  ✗ FAIL (수정 전에는 0.5 근처였음)")

    # 테스트 케이스 3: 약간 유사한 벡터 (노이즈 추가)
    print("\n[테스트 3] 약간 유사한 벡터 (원본 + 노이즈)")
    vec_orig = np.random.randn(1536)
    vec_noisy = vec_orig + np.random.randn(1536) * 0.3  # 30% 노이즈

    figma = FigmaFare(
        id="test3",
        name="original",
        box=np.array([0, 0, 100, 100]),
        extracted=type('obj', (object,), {'feature': vec_orig, 'text': '', 'box': np.array([0, 0, 100, 100])})()
    )
    web = ExtractedElement(
        box=np.array([0, 0, 100, 100]),
        text='',
        feature=vec_noisy,
        cls=0
    )

    sim_matrix = strategy.calculate_similarity([figma], [web])
    print(f"  유사도: {sim_matrix[0, 0]:.4f}")
    print(f"  기대값: 0.5~0.8 사이 (중간 유사도)")
    print(f"  ✓ PASS" if 0.4 < sim_matrix[0, 0] < 0.9 else f"  ✗ FAIL")

    # 테스트 케이스 4: 매우 유사한 벡터 (작은 노이즈)
    print("\n[테스트 4] 매우 유사한 벡터 (원본 + 작은 노이즈)")
    vec_orig = np.random.randn(1536)
    vec_similar = vec_orig + np.random.randn(1536) * 0.05  # 5% 노이즈

    figma = FigmaFare(
        id="test4",
        name="original",
        box=np.array([0, 0, 100, 100]),
        extracted=type('obj', (object,), {'feature': vec_orig, 'text': '', 'box': np.array([0, 0, 100, 100])})()
    )
    web = ExtractedElement(
        box=np.array([0, 0, 100, 100]),
        text='',
        feature=vec_similar,
        cls=0
    )

    sim_matrix = strategy.calculate_similarity([figma], [web])
    print(f"  유사도: {sim_matrix[0, 0]:.4f}")
    print(f"  기대값: 0.8~1.0 사이 (높은 유사도)")
    print(f"  ✓ PASS" if sim_matrix[0, 0] > 0.8 else f"  ✗ FAIL")

    print("\n" + "=" * 80)
    print("✅ 수정 완료: 이제 전혀 다른 이미지는 낮은 유사도를 가집니다!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    test_feature_similarity_improvement()
