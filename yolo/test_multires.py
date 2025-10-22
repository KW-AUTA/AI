"""
Multi-resolution YOLO detection 테스트 스크립트

사용법:
    python test_multires.py

비교:
    - 기존 single-resolution: 1024x1024 고정
    - 새로운 multi-resolution: 640x640, 1024x1024, 1280x1280 동시 사용
"""

from PIL import Image
from element_matcher import ElementExtractor
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_single_vs_multi_resolution(image_path: str):
    """Single vs Multi-resolution 성능 비교"""

    # 이미지 로드
    img = Image.open(image_path)
    logging.info(f"Loaded image: {image_path}, size: {img.size}")

    # ElementExtractor 초기화
    matcher = ElementExtractor(resize_size=(1024, 1024))

    # 1. Single-resolution detection (기존 방식)
    logging.info("\n=== Single-Resolution Detection ===")
    start_time = time.time()
    boxes_single, scores_single, classes_single = matcher.detect_boxes_yolo(
        img,
        conf_thresh=0.15,
        max_det=500,
        extract_features=False,
        use_multires=False  # 기존 방식
    )
    single_time = time.time() - start_time

    logging.info(f"Single-resolution results:")
    logging.info(f"  - Detected boxes: {len(boxes_single)}")
    logging.info(f"  - Time: {single_time:.2f}s")

    # 2. Multi-resolution detection (새로운 방식)
    logging.info("\n=== Multi-Resolution Detection ===")
    start_time = time.time()
    boxes_multi, scores_multi, classes_multi = matcher.detect_boxes_yolo(
        img,
        conf_thresh=0.15,
        max_det=500,
        extract_features=False,
        use_multires=True,  # 새로운 방식
        resolutions=[(640, 640), (1024, 1024), (1280, 1280)]  # 3가지 해상도
    )
    multi_time = time.time() - start_time

    logging.info(f"Multi-resolution results:")
    logging.info(f"  - Detected boxes: {len(boxes_multi)}")
    logging.info(f"  - Time: {multi_time:.2f}s")

    # 3. 비교 분석
    logging.info("\n=== Comparison ===")
    additional_boxes = len(boxes_multi) - len(boxes_single)
    time_overhead = multi_time / single_time

    logging.info(f"Additional boxes detected: {additional_boxes} ({additional_boxes/max(1, len(boxes_single))*100:.1f}% increase)")
    logging.info(f"Time overhead: {time_overhead:.2f}x")

    # 박스 크기 분포 분석
    if len(boxes_single) > 0 and len(boxes_multi) > 0:
        def get_box_sizes(boxes):
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = widths * heights
            return areas

        areas_single = get_box_sizes(boxes_single)
        areas_multi = get_box_sizes(boxes_multi)

        logging.info(f"\nBox size statistics:")
        logging.info(f"  Single-resolution:")
        logging.info(f"    - Mean area: {areas_single.mean():.1f}")
        logging.info(f"    - Min area: {areas_single.min():.1f}")
        logging.info(f"    - Max area: {areas_single.max():.1f}")

        logging.info(f"  Multi-resolution:")
        logging.info(f"    - Mean area: {areas_multi.mean():.1f}")
        logging.info(f"    - Min area: {areas_multi.min():.1f}")
        logging.info(f"    - Max area: {areas_multi.max():.1f}")

    return {
        'single': {'boxes': len(boxes_single), 'time': single_time},
        'multi': {'boxes': len(boxes_multi), 'time': multi_time}
    }


def test_custom_resolutions(image_path: str):
    """커스텀 해상도 조합 테스트"""
    img = Image.open(image_path)
    matcher = ElementExtractor(resize_size=(1024, 1024))

    resolution_sets = [
        # 1. 빠른 탐지 (2개 해상도)
        ("Fast", [(640, 640), (1024, 1024)]),

        # 2. 균형 탐지 (3개 해상도) - 기본값
        ("Balanced", [(640, 640), (1024, 1024), (1280, 1280)]),

        # 3. 정밀 탐지 (4개 해상도)
        ("Precise", [(512, 512), (768, 768), (1024, 1024), (1280, 1280)]),

        # 4. 작은 요소 집중 (큰 해상도만)
        ("Small-focused", [(1024, 1024), (1280, 1280), (1536, 1536)]),
    ]

    logging.info("\n=== Testing Different Resolution Sets ===")
    results = {}

    for name, resolutions in resolution_sets:
        logging.info(f"\n{name}: {resolutions}")
        start_time = time.time()

        boxes, scores, classes = matcher.detect_boxes_yolo(
            img,
            conf_thresh=0.15,
            max_det=500,
            extract_features=False,
            use_multires=True,
            resolutions=resolutions
        )

        elapsed = time.time() - start_time
        results[name] = {'boxes': len(boxes), 'time': elapsed}

        logging.info(f"  - Boxes: {len(boxes)}")
        logging.info(f"  - Time: {elapsed:.2f}s")

    return results


if __name__ == "__main__":
    # 테스트할 이미지 경로 (실제 경로로 변경 필요)
    test_image = "test_image.png"  # 실제 이미지 경로로 변경

    # 1. Single vs Multi 비교
    test_single_vs_multi_resolution(test_image)

    # 2. 다양한 해상도 조합 테스트
    test_custom_resolutions(test_image)
