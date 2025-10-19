"""
PaddleOCR Helper - 별도 프로세스로 PaddleOCR 실행
ElementExtractor에서 사용하기 위한 헬퍼 모듈
"""
import os
import sys
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image


# PaddleOCR 실행 스크립트 (별도 프로세스)
PADDLE_OCR_SCRIPT = """
import sys
import pickle
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

def run_ocr(image_path, boxes_pkl_path, output_pkl_path):
    # PaddleOCR 초기화
    ocr = PaddleOCR(lang='en')

    # 박스 로드
    with open(boxes_pkl_path, 'rb') as f:
        boxes = pickle.load(f)

    # 이미지 로드
    img = Image.open(image_path)
    img_array = np.array(img)

    results = []
    for box in boxes:
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

                # 모든 텍스트 결합 (필터링 없음 - standalone과 동일)
                text = ' '.join(texts) if texts else ''
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
            results.append({'box': box, 'text': '', 'score': 0.0})

    # 결과 저장
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    image_path = sys.argv[1]
    boxes_pkl_path = sys.argv[2]
    output_pkl_path = sys.argv[3]
    run_ocr(image_path, boxes_pkl_path, output_pkl_path)
"""


class PaddleOCRHelper:
    """PaddleOCR을 별도 프로세스로 실행하는 헬퍼"""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "paddle_ocr_helper"
        self.temp_dir.mkdir(exist_ok=True)

        # 스크립트 파일 생성
        self.script_path = self.temp_dir / "paddle_ocr_worker.py"
        self.script_path.write_text(PADDLE_OCR_SCRIPT)

    def extract_text_batch(self, image_path: str, boxes: List[np.ndarray]) -> List[str]:
        """박스 목록에서 텍스트 일괄 추출

        Args:
            image_path: 이미지 파일 경로
            boxes: 박스 좌표 리스트 [(x1,y1,x2,y2), ...]

        Returns:
            추출된 텍스트 리스트
        """
        if len(boxes) == 0:
            return []

        # 임시 파일 경로
        boxes_pkl = self.temp_dir / f"boxes_{os.getpid()}.pkl"
        output_pkl = self.temp_dir / f"output_{os.getpid()}.pkl"

        try:
            # 박스 저장
            with open(boxes_pkl, 'wb') as f:
                pickle.dump(boxes, f)

            # PaddleOCR 실행 (별도 프로세스)
            result = subprocess.run([
                sys.executable,
                str(self.script_path),
                image_path,
                str(boxes_pkl),
                str(output_pkl)
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"⚠️ PaddleOCR 프로세스 실패: {result.stderr}")
                return [""] * len(boxes)

            # 결과 로드
            with open(output_pkl, 'rb') as f:
                ocr_results = pickle.load(f)

            # 텍스트만 추출
            texts = [r['text'] for r in ocr_results]
            return texts

        except Exception as e:
            print(f"⚠️ PaddleOCR 실행 실패: {e}")
            return [""] * len(boxes)

        finally:
            # 임시 파일 정리
            if boxes_pkl.exists():
                boxes_pkl.unlink()
            if output_pkl.exists():
                output_pkl.unlink()

    def extract_text_single(self, image_path: str, box: np.ndarray) -> str:
        """단일 박스에서 텍스트 추출"""
        results = self.extract_text_batch(image_path, [box])
        return results[0] if results else ""
