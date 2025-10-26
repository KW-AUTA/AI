import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from yolo.element_matcher import ElementExtractor
from PIL import Image
import numpy as np

print("=" * 50)
print("EasyOCR 테스트 시작")
print("=" * 50)

# ElementExtractor 초기화 (EasyOCR 사용)
print("\n1. ElementExtractor 초기화 중...")
extractor = ElementExtractor(ocr_engine="easyocr")
print(f"   ✓ OCR 엔진: {extractor.ocr_engine}")

# 테스트 이미지 로드 (실제 이미지 경로를 지정하세요)
print("\n2. 테스트 이미지 로드...")
# 예시: test_image_path = "figma_elements_extracted.png"
test_image_path = input("테스트할 이미지 경로를 입력하세요: ").strip()

if not os.path.exists(test_image_path):
    print(f"   ✗ 이미지를 찾을 수 없습니다: {test_image_path}")
    exit(1)

img = Image.open(test_image_path)
print(f"   ✓ 이미지 크기: {img.size}")

# 테스트할 박스 영역 입력
print("\n3. OCR 테스트할 박스 영역 입력")
print("   형식: x1,y1,x2,y2 (예: 10,10,100,50)")
box_input = input("   박스 좌표: ").strip()

try:
    x1, y1, x2, y2 = map(int, box_input.split(','))
    box = np.array([x1, y1, x2, y2])
    print(f"   ✓ 박스: [{x1}, {y1}, {x2}, {y2}]")
except:
    print("   ✗ 잘못된 형식입니다. 전체 이미지로 테스트합니다.")
    box = np.array([0, 0, img.width, img.height])

# OCR 실행
print("\n4. OCR 실행 중... (첫 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
try:
    text = extractor.extract_text(img, box)
    print(f"\n{'=' * 50}")
    print(f"추출된 텍스트: '{text}'")
    print(f"{'=' * 50}")
except Exception as e:
    print(f"\n   ✗ OCR 실행 오류: {e}")
    import traceback
    traceback.print_exc()
