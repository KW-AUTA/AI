#!/usr/bin/env python3
"""
매칭 결과를 분석하는 스크립트
실제 매칭된 요소들의 텍스트와 점수를 확인
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolo.core.mapping import mapping_legacy
from yolo.figma.figma_parser import parse_figma_json
from yolo.web.web_navigator import WebNavigator, WebNavigatorConfig
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def analyze_matches():
    """매칭 결과를 상세히 분석"""

    # 테스트 설정 (component_test.py와 동일하게)
    base_url = 'https://heeeesssuu.github.io/web3studio'
    json_url = 'https://api.figma.com/v1/files/PbZUlZHMdFfAWfpKLYNkRd?depth=10'
    current_page = '78:2'

    print("\n" + "=" * 80)
    print("매칭 결과 상세 분석")
    print("=" * 80)

    # Figma 데이터 파싱
    print("\n[1] Figma 데이터 로드 중...")
    figma_raw, figma_tree, figma_interactions = parse_figma_json(json_url)
    print(f"✓ Figma 노드 수: {len(figma_tree)}")

    # 웹 네비게이터 설정
    print("\n[2] 웹 페이지 로드 중...")
    nav_config = WebNavigatorConfig(headless=True, base_url=base_url)
    web_navigator = WebNavigator(config=nav_config)

    try:
        web_navigator.navigate(current_page)
        web_img = web_navigator.capture_full_page()
        print(f"✓ 웹 페이지 로드 완료")

        # 매칭 실행
        print("\n[3] 매칭 실행 중...")
        matches = mapping_legacy(
            web_navigator=web_navigator,
            web_img=web_img,
            figma_raw=figma_raw,
            figma_tree=figma_tree,
            figma_interactions=figma_interactions,
            current_page=current_page,
            base_url=base_url
        )

        print(f"\n{'=' * 80}")
        print(f"매칭 결과: 총 {len(matches)}개")
        print(f"{'=' * 80}\n")

        # 각 매칭 결과 출력
        for i, match in enumerate(matches, 1):
            print(f"\n[Match #{i}]")
            print(f"  Component Name: {match.componentName}")
            print(f"  Destination Figma Page: {match.destinationFigmaPage}")
            print(f"  Destination URL: {match.destinationUrl}")
            print(f"  Actual URL: {match.actualUrl}")
            print(f"  Is Success: {match.isSuccess}")
            print(f"  Is Routing: {match.isRouting}")
            if match.failReason:
                print(f"  Fail Reason: {match.failReason}")

        print(f"\n{'=' * 80}")
        print("분석 완료")
        print(f"{'=' * 80}\n")

    finally:
        web_navigator.quit()

if __name__ == '__main__':
    analyze_matches()
