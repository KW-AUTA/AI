#!/usr/bin/env python3
"""
빈 텍스트 케이스 테스트
- Figma 텍스트가 비어있고 Web에만 텍스트가 있는 경우
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from yolo.core.matcher import SimilarityMatcher
from yolo.core.models import FigmaFare, ExtractedElement, SimilarityConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def test_empty_text_matching():
    """빈 텍스트인 경우 매칭 동작 테스트"""

    print("\n" + "=" * 80)
    print("빈 텍스트 케이스 테스트")
    print("=" * 80)

    # 공통 feature (유사하게)
    common_feature_base = np.random.randn(1536)
    figma_feature = common_feature_base + np.random.randn(1536) * 0.1  # 10% 노이즈
    web_feature = common_feature_base + np.random.randn(1536) * 0.1

    # 케이스 1: Figma 텍스트 비어있음, Web 텍스트 있음
    print("\n[케이스 1] Figma 텍스트 비어있음, Web 텍스트 있음")
    figma1 = FigmaFare(
        id="test1",
        name="logo-ycombinator_svg",
        box=np.array([10, 10, 100, 50]),
        extracted=type('obj', (object,), {
            'text': '',  # 빈 텍스트 (OCR 실패 상황)
            'box': np.array([10, 10, 100, 50]),
            'feature': figma_feature
        })()
    )

    web1 = ExtractedElement(
        box=np.array([12, 12, 102, 52]),  # 크기와 위치 비슷
        text='Y Combinator · Product clarity',
        feature=web_feature,
        cls=0
    )

    print(f"  Figma 텍스트: '{figma1.extracted.text}' (빈 문자열)")
    print(f"  Web 텍스트: '{web1.text}'")

    config = SimilarityConfig.from_env()
    matcher = SimilarityMatcher(config=config)

    matches, unmatched_figma, unmatched_web = matcher.find_matches(
        [figma1],
        [web1],
        min_similarity=0.3
    )

    if matches:
        match = matches[0]
        print(f"\n  ✓ 매칭됨:")
        print(f"    text_sim={match.text_similarity:.3f}")
        print(f"    feat_sim={match.feature_similarity:.3f}")
        print(f"    size_sim={match.size_similarity:.3f}")
        print(f"    coord_sim={match.coordinate_similarity:.3f}")
        print(f"    score={match.score:.3f}")
        print(f"\n  ⚠️ 한쪽만 텍스트가 있으면 텍스트 체크를 건너뜀!")
    else:
        print(f"\n  ✗ 매칭 안 됨")

    # 케이스 2: 둘 다 텍스트 있지만 완전히 다름 (위치/크기만 비슷)
    print(f"\n{'=' * 80}")
    print("[케이스 2] 둘 다 텍스트 있지만 완전히 다름")

    figma2 = FigmaFare(
        id="test2",
        name="logo-ycombinator_svg",
        box=np.array([10, 10, 100, 50]),
        extracted=type('obj', (object,), {
            'text': 'WingRiders',  # 다른 텍스트
            'box': np.array([10, 10, 100, 50]),
            'feature': figma_feature
        })()
    )

    web2 = ExtractedElement(
        box=np.array([12, 12, 102, 52]),
        text='Y Combinator · Product clarity',
        feature=web_feature,
        cls=0
    )

    print(f"  Figma 텍스트: '{figma2.extracted.text}'")
    print(f"  Web 텍스트: '{web2.text}'")

    matches2, _, _ = matcher.find_matches(
        [figma2],
        [web2],
        min_similarity=0.3
    )

    if matches2:
        match = matches2[0]
        print(f"\n  ⚠️ 매칭됨 (문제!):")
        print(f"    text_sim={match.text_similarity:.3f}")
        print(f"    feat_sim={match.feature_similarity:.3f}")
        print(f"    size_sim={match.size_similarity:.3f}")
        print(f"    coord_sim={match.coordinate_similarity:.3f}")
        print(f"    score={match.score:.3f}")
    else:
        print(f"\n  ✓ 매칭 안 됨 (올바른 동작!)")

    print(f"\n{'=' * 80}\n")


if __name__ == '__main__':
    test_empty_text_matching()
