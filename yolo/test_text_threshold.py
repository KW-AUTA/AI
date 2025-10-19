#!/usr/bin/env python3
"""
텍스트 임계값 테스트
- 둘 다 텍스트가 있는데 text_sim=0.0인 경우 거부되는지 확인
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_text_threshold():
    """텍스트 유사도 임계값 테스트"""

    print("\n" + "=" * 80)
    print("텍스트 유사도 임계값 테스트")
    print("=" * 80)

    # 환경변수 설정
    os.environ['MIN_TEXT_SIM_BOTH'] = '0.50'

    # 테스트 데이터: 둘 다 텍스트가 있지만 완전히 다름
    figma = FigmaFare(
        id="test1",
        name="logo-ycombinator_svg",
        box=np.array([10, 10, 100, 50]),
        extracted=type('obj', (object,), {
            'text': 'Y Combinator',  # Figma 텍스트
            'box': np.array([10, 10, 100, 50]),
            'feature': np.random.randn(1536)
        })()
    )

    web = ExtractedElement(
        box=np.array([10, 10, 100, 50]),  # 크기와 위치는 비슷
        text='Product clarity',  # 완전히 다른 텍스트
        feature=np.random.randn(1536),  # 다른 feature
        cls=0
    )

    print(f"\nFigma 텍스트: '{figma.extracted.text}'")
    print(f"Web 텍스트: '{web.text}'")

    # Matcher 생성
    config = SimilarityConfig.from_env()
    matcher = SimilarityMatcher(config=config)

    # 매칭 실행
    print(f"\n임계값 설정:")
    print(f"  MIN_TEXT_SIM_BOTH = 0.50")
    print(f"\n매칭 시작...\n")

    matches, unmatched_figma, unmatched_web = matcher.find_matches(
        [figma],
        [web],
        min_similarity=0.3
    )

    print(f"\n{'=' * 80}")
    print("결과:")
    print(f"{'=' * 80}")
    print(f"매칭된 요소: {len(matches)}개")
    print(f"미매칭 Figma: {len(unmatched_figma)}개")
    print(f"미매칭 Web: {len(unmatched_web)}개")

    if matches:
        for match in matches:
            print(f"\n⚠️ 매칭됨 (예상과 다름!):")
            print(f"  Figma: '{match.figma.extracted.text}'")
            print(f"  Web: '{match.web.text}'")
            print(f"  text_sim={match.text_similarity:.3f}")
            print(f"  feat_sim={match.feature_similarity:.3f}")
            print(f"  size_sim={match.size_similarity:.3f}")
            print(f"  score={match.score:.3f}")
    else:
        print(f"\n✓ 매칭 안 됨 (올바른 동작!)")
        print(f"  이유: 텍스트 유사도가 MIN_TEXT_SIM_BOTH(0.50) 미만")

    print(f"\n{'=' * 80}\n")


if __name__ == '__main__':
    test_text_threshold()
