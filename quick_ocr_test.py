#!/usr/bin/env python3
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from yolo.element_matcher import ElementExtractor
from PIL import Image
import numpy as np

# 이미지 경로 (사용자가 제공한 스크린샷)
# 주의: 실제 경로로 변경 필요
image_paths = [
    "figma_elements_extracted.png",
    "figma_elements_extracted_with_interaction.png",
]

# ElementExtractor 초기화
print("EasyOCR 초기화 중...")
extractor = ElementExtractor(ocr_engine="easyocr")

# 테스트할 박스들 (예시)
test_boxes = [
    # 발전기금 영역 (왼쪽)
    np.array([5, 60, 155, 200]),  # x1, y1, x2, y2
    # 캠퍼스안내 영역 (오른쪽)
    np.array([165, 60, 315, 340]),
]

for img_path in image_paths:
    if not os.path.exists(img_path):
        print(f"이미지를 찾을 수 없습니다: {img_path}")
        continue

    print(f"\n{'='*60}")
    print(f"이미지: {img_path}")
    print(f"{'='*60}")

    img = Image.open(img_path)
    print(f"이미지 크기: {img.size}")

    for i, box in enumerate(test_boxes):
        print(f"\n[박스 {i+1}] 좌표: {box}")
        try:
            text = extractor.extract_text(img, box, box_idx=i)
            print(f"추출된 텍스트: '{text}'")
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()

print("\n완료!")
