#!/usr/bin/env python3
"""
PaddleOCR + YOLO 통합 테스트
환경변수 USE_PADDLEOCR=true로 설정해야 함
"""
import sys
import os

# CRITICAL: 이 환경변수를 ElementExtractor import 전에 설정해야 함
os.environ['USE_PADDLEOCR'] = 'true'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import numpy as np
from yolo.core.element_matcher import ElementExtractor

print("=" * 80)
print("PaddleOCR + YOLO 통합 테스트 (import 순서 수정)")
print("=" * 80)

# 테스트 이미지
test_image_path = "/Users/song-inseop/dev/re_AUTA/AI/yolo/core/debug_sim/crops/20251019-083015/003_logo-ycombinator_svg.png"

if not os.path.exists(test_image_path):
    print(f"⚠️ 테스트 이미지를 찾을 수 없습니다: {test_image_path}")
    exit(1)

print(f"\n테스트 이미지: {test_image_path}")

# ElementExtractor 초기화
print("\nElementExtractor 초기화 중 (PaddleOCR 활성화)...")
try:
    extractor = ElementExtractor(use_paddleocr=True)
    print("✓ ElementExtractor 초기화 완료")
except Exception as e:
    print(f"✗ 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 이미지 로드
img = Image.open(test_image_path)
print(f"이미지 크기: {img.size}")

# OCR 테스트
box = np.array([0, 0, img.width, img.height])

print("\nOCR 실행 중...")
try:
    extracted_text = extractor.extract_text(img, box)

    print(f"\n{'=' * 80}")
    print("OCR 결과:")
    print(f"{'=' * 80}")
    print(f"추출된 텍스트: '{extracted_text}'")

    # 키워드 체크
    keywords = ['combinator', 'product', 'clarity', 'y']
    found = [kw for kw in keywords if kw.lower() in extracted_text.lower()]

    print(f"\n발견된 키워드: {found} ({len(found)}/{len(keywords)})")

    if len(found) >= 3:
        print("\n✓✓✓ OCR 성공!")
    elif len(found) >= 2:
        print("\n✓ 부분 성공")
    else:
        print("\n✗ OCR 실패")

except Exception as e:
    print(f"\n✗ OCR 실패: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'=' * 80}\n")
