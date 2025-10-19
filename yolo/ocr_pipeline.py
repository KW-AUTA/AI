#!/usr/bin/env python3
"""
2단계 OCR 파이프라인: YOLO 탐지 → PaddleOCR 텍스트 추출
메모리에서 완전히 분리하여 충돌 방지
"""
import os
import sys
import pickle
import subprocess
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

# Step 1 스크립트 (YOLO만 사용)
DETECT_SCRIPT = """
import sys
import pickle
from PIL import Image
from ultralytics import YOLO

def detect_boxes(image_path, model_path):
    # YOLO 모델 로드
    yolo = YOLO(model_path, task='detect', verbose=False)

    # 이미지 로드
    img = Image.open(image_path)

    # 박스 탐지
    results = yolo.predict(source=img, conf=0.05, iou=0.1, max_det=500, verbose=False)[0]

    # 박스 좌표 추출
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    # 필터링
    mask = scores >= 0.05
    boxes_final = boxes[mask]
    scores_final = scores[mask]
    classes_final = classes[mask]

    return {
        'boxes': boxes_final,
        'scores': scores_final,
        'classes': classes_final,
        'image_size': (img.width, img.height)
    }

if __name__ == '__main__':
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]

    result = detect_boxes(image_path, model_path)

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"Detected {len(result['boxes'])} boxes")
"""

# Step 2 스크립트 (PaddleOCR만 사용)
OCR_SCRIPT = """
import sys
import pickle
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

def extract_text_from_boxes(image_path, boxes):
    # PaddleOCR 초기화
    ocr = PaddleOCR(lang='en')

    # 이미지 로드
    img = Image.open(image_path)
    img_array = np.array(img)

    results = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # ROI 추출 (패딩 추가)
        pad = 12
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(img.width, x2 + pad)
        y2p = min(img.height, y2 + pad)

        roi = img_array[y1p:y2p, x1p:x2p]

        if roi.size == 0:
            results.append({'box': box, 'text': '', 'score': 0.0})
            continue

        # OCR 실행
        try:
            ocr_result = ocr.predict(roi)

            if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
                texts = ocr_result[0].get('rec_texts', [])
                scores = ocr_result[0].get('rec_scores', [])

                # Confidence 0.5 이상만
                filtered_texts = [
                    texts[j] for j in range(len(texts))
                    if j < len(scores) and scores[j] >= 0.5
                ]

                text = ' '.join(filtered_texts)
                avg_score = np.mean(scores) if scores else 0.0
            else:
                text = ''
                avg_score = 0.0

            results.append({
                'box': box,
                'text': text,
                'score': float(avg_score)
            })
        except Exception as e:
            print(f"OCR failed for box {i}: {e}")
            results.append({'box': box, 'text': '', 'score': 0.0})

    return results

if __name__ == '__main__':
    image_path = sys.argv[1]
    boxes_path = sys.argv[2]
    output_path = sys.argv[3]

    # 박스 로드
    with open(boxes_path, 'rb') as f:
        detect_result = pickle.load(f)

    boxes = detect_result['boxes']

    # OCR 실행
    ocr_results = extract_text_from_boxes(image_path, boxes)

    # 결과 저장
    with open(output_path, 'wb') as f:
        pickle.dump({
            'detect_result': detect_result,
            'ocr_results': ocr_results
        }, f)

    print(f"OCR completed for {len(ocr_results)} boxes")
"""


class TwoStageOCRPipeline:
    """2단계 OCR 파이프라인"""

    def __init__(self, yolo_model_path: str = None):
        if yolo_model_path is None:
            current_dir = Path(__file__).parent
            yolo_model_path = str(current_dir / "models_weights" / "best_one.pt")

        self.yolo_model_path = yolo_model_path
        self.temp_dir = Path("/tmp/yolo_paddle_pipeline")
        self.temp_dir.mkdir(exist_ok=True)

        # 스크립트 파일 생성
        self.detect_script_path = self.temp_dir / "detect.py"
        self.ocr_script_path = self.temp_dir / "ocr.py"

        self.detect_script_path.write_text(DETECT_SCRIPT)
        self.ocr_script_path.write_text(OCR_SCRIPT)

    def process(self, image_path: str) -> dict:
        """이미지 처리 (YOLO → PaddleOCR)"""

        image_path = str(Path(image_path).absolute())

        # Step 1: YOLO 탐지
        print("Step 1: YOLO 박스 탐지...")
        boxes_path = self.temp_dir / "boxes.pkl"

        result = subprocess.run([
            sys.executable,
            str(self.detect_script_path),
            image_path,
            self.yolo_model_path,
            str(boxes_path)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"YOLO 탐지 실패: {result.stderr}")

        print(result.stdout.strip())

        # Step 2: PaddleOCR 텍스트 추출
        print("\nStep 2: PaddleOCR 텍스트 추출...")
        output_path = self.temp_dir / "output.pkl"

        result = subprocess.run([
            sys.executable,
            str(self.ocr_script_path),
            image_path,
            str(boxes_path),
            str(output_path)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"OCR 실패: {result.stderr}")

        print(result.stdout.strip())

        # 결과 로드
        with open(output_path, 'rb') as f:
            final_result = pickle.load(f)

        return final_result


def test_pipeline():
    """파이프라인 테스트"""
    print("=" * 80)
    print("2단계 OCR 파이프라인 테스트")
    print("=" * 80)

    test_image = "/Users/song-inseop/dev/re_AUTA/AI/yolo/core/debug_sim/crops/20251019-083015/003_logo-ycombinator_svg.png"

    if not Path(test_image).exists():
        print(f"⚠️ 테스트 이미지를 찾을 수 없습니다: {test_image}")
        return

    print(f"\n테스트 이미지: {test_image}\n")

    # 파이프라인 실행
    pipeline = TwoStageOCRPipeline()
    result = pipeline.process(test_image)

    # 결과 출력
    print("\n" + "=" * 80)
    print("결과:")
    print("=" * 80)

    detect_result = result['detect_result']
    ocr_results = result['ocr_results']

    print(f"\n탐지된 박스: {len(detect_result['boxes'])}개")
    print(f"OCR 결과: {len(ocr_results)}개\n")

    for i, ocr_res in enumerate(ocr_results):
        if ocr_res['text']:
            print(f"{i+1}. '{ocr_res['text']}' (score: {ocr_res['score']:.2f})")

    # 키워드 체크
    all_texts = ' '.join([r['text'] for r in ocr_results])
    keywords = ['y', 'combinator', 'product', 'clarity']
    found = [kw for kw in keywords if kw.lower() in all_texts.lower()]

    print(f"\n전체 텍스트: '{all_texts}'")
    print(f"발견된 키워드: {found} ({len(found)}/{len(keywords)})")

    if len(found) >= 3:
        print("\n✓✓✓ 성공!")
    else:
        print("\n△ 부분 성공")

    print("=" * 80)


if __name__ == '__main__':
    test_pipeline()
