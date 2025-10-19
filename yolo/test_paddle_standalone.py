#!/usr/bin/env python3
"""
PaddleOCR 단독 테스트 - YOLO 없이 실행
주의: ElementExtractor를 import하지 않음 (YOLO 충돌 방지)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # macOS OpenMP 충돌 방지

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

print("=" * 80)
print("PaddleOCR 단독 테스트")
print("=" * 80)

# 테스트 이미지 경로
test_image_path = "/Users/song-inseop/dev/re_AUTA/AI/yolo/core/debug_sim/crops/20251019-083015/003_logo-ycombinator_svg.png"

if not os.path.exists(test_image_path):
    print(f"⚠️ 테스트 이미지를 찾을 수 없습니다: {test_image_path}")
    exit(1)

print(f"\n테스트 이미지: {test_image_path}")

# PaddleOCR 초기화
print("\nPaddleOCR 초기화 중 (영어 모델)...")
ocr = PaddleOCR(lang='en')
print("✓ 초기화 완료!")

# 이미지 로드
img = Image.open(test_image_path)
img_array = np.array(img)
print(f"이미지 크기: {img.size}")

# OCR 실행
print("\nOCR 실행 중...")
result = ocr.predict(img_array)

# 결과 파싱
if result and isinstance(result, list) and len(result) > 0:
    ocr_result = result[0]
    texts = ocr_result.get('rec_texts', [])
    scores = ocr_result.get('rec_scores', [])

    print(f"\n{'=' * 80}")
    print("OCR 결과:")
    print(f"{'=' * 80}")

    if texts:
        print(f"\n✓ 총 {len(texts)}개 텍스트 발견:\n")
        for i, text in enumerate(texts):
            score = scores[i] if i < len(scores) else 0.0
            print(f"  {i+1}. '{text}' (신뢰도: {score:.2f})")

        # 전체 텍스트 결합
        all_text = " ".join(texts)
        print(f"\n결합된 텍스트: '{all_text}'")

        # 키워드 체크
        keywords = ['y', 'combinator', 'product', 'clarity']
        found = [kw for kw in keywords if kw.lower() in all_text.lower()]
        print(f"\n발견된 키워드: {found} ({len(found)}/{len(keywords)})")

        if len(found) >= 3:
            print("\n✓✓✓ OCR 성공!")
        elif len(found) >= 2:
            print("\n✓ 부분 성공")
        else:
            print("\n✗ 키워드 부족")
    else:
        print("\n✗ 텍스트를 찾지 못했습니다.")
else:
    print("\n✗ OCR 결과가 비어있습니다.")

print(f"\n{'=' * 80}\n")
