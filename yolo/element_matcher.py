import numpy as np
from PIL import Image
from PIL import ImageFilter
import torch
import torch.nn.functional as F
import torchvision.ops
import torchvision.transforms as T
from torchvision.transforms.functional import pad
from ultralytics import YOLO
import tesserocr
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Any, Dict
import cv2
import os
import re
import unicodedata
import logging
from scipy.optimize import linear_sum_assignment
from .models import FigmaFare, ExtractedElement, MatchResult
from .errorChecker import ErrorChecker
# Letterbox function: resize and pad image to meet new_shape, maintaining aspect ratio.
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image to meet new_shape, maintaining aspect ratio.
    Returns padded image, resize ratio, and padding (left, top).
    """
    # Convert PIL numpy array (H, W, C)
    import cv2
    shape = im.shape[:2]  # current shape [height, width]
    # Compute resize ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Compute unpadded new size
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    # Compute padding
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    # Resize image
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # Compute border sizes
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # Add border
    im_padded = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im_padded, r, (left, top)


# Non-Maximum Suppression for axis-aligned bounding boxes
def non_max_suppression(
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5
) -> List[int]:
    """
    Perform Non-Maximum Suppression on axis-aligned bounding boxes using torchvision.ops.nms.
    boxes: numpy array of shape (N,4) in (x1,y1,x2,y3)
    scores: confidence scores array of shape (N,), defaults to all 1s
    returns: list of indices to keep
    """
    if boxes.size == 0:
        return []

    # Convert numpy arrays to torch tensors
    boxes_tensor = torch.from_numpy(boxes).float()
    if scores is None:
        scores_tensor = torch.ones(len(boxes), dtype=torch.float32)
    else:
        scores_tensor = torch.from_numpy(scores).float()

    # Perform NMS
    keep_indices_tensor = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)

    # Convert result back to a list of integers
    return keep_indices_tensor.cpu().numpy().tolist()
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules.conv import Conv
from torch.nn import Conv2d

# 모델 import
from .models import (
    InteractionType, TriggerType, NavigationType, PositionType,
    Vector, Overlay, NodeInteraction, UrlInteraction, Interaction,
    FigmaElement, WebElement, MatchResult, ExtractedElement
)

class ElementExtractor:
    """요소 매칭을 수행하는 클래스"""
    def __init__(self, yolo_model_path: str = None, resize_size: Tuple[int, int] = (736, 736), debug_ocr: bool = False):
        current_dir = os.path.dirname(__file__)
        if yolo_model_path is None:
            yolo_model_path = os.path.join(current_dir, "best_one.pt")
        self.yolo = YOLO(yolo_model_path, task='detect', verbose=False)
        self.resize_size = resize_size
        self.debug_ocr = debug_ocr  # OCR 전처리 디버그 모드

        # tesserocr 설정
        tessdata_dir = "/usr/local/share/tessdata"

        self.api = tesserocr.PyTessBaseAPI(path=tessdata_dir, lang='kor+eng')
        self.api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
        self.api.SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가-힣 ")
        # self.api.SetVariable("user_defined_dpi", "300")

        self.transform = T.Compose([
            T.Resize(resize_size),
            # T.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # OCR 디버그 디렉토리 설정
        if self.debug_ocr:
            self.ocr_debug_dir = os.path.join(os.path.dirname(__file__), 'debug_ocr')
            os.makedirs(self.ocr_debug_dir, exist_ok=True)
            logging.info(f"OCR debug mode enabled. Results will be saved to: {self.ocr_debug_dir}")

    def __del__(self):
        """tesserocr API 정리"""
        if hasattr(self, 'api'):
            self.api.End()

    def _pick_psm(self, w: int, h: int) -> int:
        """ROI 크기/종횡비에 따라 최적화된 PSM 선택"""
        if w == 0 or h == 0:
            return tesserocr.PSM.SINGLE_BLOCK

        aspect = w / max(1, h)
        area = w * h

        # 매우 작은 영역: 단일 문자나 숫자
        if area < 20 * 20:
            return tesserocr.PSM.SINGLE_CHAR
        # 작은 영역: 단어 단위
        elif area < 50 * 50:
            return tesserocr.PSM.SINGLE_WORD
        # 매우 가로로 긴 형태: 단일 라인
        elif aspect >= 4.0:
            return tesserocr.PSM.SINGLE_LINE
        # 세로로 긴 형태나 중간 크기의 가로 긴 형태: 단일 라인
        elif aspect <= 0.5 or aspect >= 2.0:
            return tesserocr.PSM.SINGLE_LINE
        # 기본: 단일 블록 처리
        else:
            return tesserocr.PSM.SINGLE_BLOCK

    def _preprocess_roi(
        self,
        pil_crop: Image.Image,
        scale: float = None,
        pad: int = 15
    ) -> Image.Image:
        """개선된 ROI 전처리로 OCR 인식률 향상

        어두운 배경 + 밝은 텍스트 자동 감지 및 반전 처리
        """
        arr = np.array(pil_crop)
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        # 동적 스케일링 (ROI 크기에 따라 조정)
        h, w = gray.shape
        if scale is None:
            if max(h, w) < 30:
                scale = 4.0
            elif max(h, w) < 60:
                scale = 3.0
            elif max(h, w) < 120:
                scale = 2.0
            else:
                scale = 1.5

        # 스케일링 (Lanczos 보간법)
        if scale > 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 노이즈 제거
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)

        # 배경 밝기 자동 감지
        # 이미지 외곽 영역의 평균 밝기로 배경 추정
        border_width = max(5, min(denoised.shape) // 10)
        top_border = denoised[:border_width, :]
        bottom_border = denoised[-border_width:, :]
        left_border = denoised[:, :border_width]
        right_border = denoised[:, -border_width:]

        background_brightness = np.mean([
            np.mean(top_border),
            np.mean(bottom_border),
            np.mean(left_border),
            np.mean(right_border)
        ])

        # 배경이 어두우면 (밝기 < 128) 반전 필요
        # 어두운 배경 + 밝은 텍스트 → 밝은 배경 + 어두운 텍스트로 변환
        needs_inversion = background_brightness < 128

        if needs_inversion:
            # THRESH_BINARY_INV: 어두운 배경을 흰색으로, 밝은 텍스트를 검은색으로
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            # THRESH_BINARY: 일반적인 경우 (밝은 배경 + 어두운 텍스트)
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

        # 모폴로지 연산으로 텍스트 구조 개선
        if needs_inversion:
            # 어두운 배경의 경우: 외곽선 텍스트를 채우는 강력한 처리
            # 1. 강한 Dilation으로 외곽선 채우기
            kernel_dilate = np.ones((4, 4), np.uint8)
            binary = cv2.dilate(binary, kernel_dilate, iterations=2)

            # 2. Closing으로 남은 구멍 메우기
            kernel_close = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

            # 3. Erosion으로 너무 두꺼워진 글자 다시 얇게 (약하게)
            kernel_erode = np.ones((2, 2), np.uint8)
            binary = cv2.erode(binary, kernel_erode, iterations=1)
        else:
            # 일반적인 경우: 기본 Closing만
            kernel_close = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # 여백 추가
        binary = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

        return Image.fromarray(binary)

    def extract_text(self, img: Image.Image, box: np.ndarray, box_idx: int = None) -> str:
        """
        OCR을 사용하여 이미지에서 텍스트 추출

        Args:
            img: 전체 이미지
            box: 텍스트 영역 박스 [x1, y1, x2, y2]
            box_idx: 박스 인덱스 (디버그 시각화용, optional)

        Returns:
            추출된 텍스트
        """
        x1, y1, x2, y2 = map(int, box)
        crop_img = img.crop((x1, y1, x2, y2))

        # 디버그 모드일 때 전처리 단계별 저장
        if self.debug_ocr and box_idx is not None:
            preprocessing_steps = {}

        # 1. 원본 이미지
        img_np = np.array(crop_img)
        if self.debug_ocr and box_idx is not None:
            preprocessing_steps['1_original'] = img_np.copy()

        # 2. Grayscale 변환
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        if self.debug_ocr and box_idx is not None:
            preprocessing_steps['2_grayscale'] = gray.copy()

        # 디버그 시각화 저장
        if self.debug_ocr and box_idx is not None:
            self._save_ocr_preprocessing_debug(preprocessing_steps, box_idx, box)

        # OCR 실행 (grayscale 이미지 사용)
        gray_pil = Image.fromarray(gray)
        self.api.SetImage(gray_pil)
        text = self.api.GetUTF8Text()

        # 후처리
        text = ' '.join(text.split())
        text = ''.join(c for c in text if c.isalnum() or c.isspace() or '\uAC00' <= c <= '\uD7A3')

        return text.strip()

    def _save_ocr_preprocessing_debug(self, steps: Dict[str, np.ndarray], box_idx: int, box: np.ndarray):
        """OCR 전처리 단계별 결과를 시각화하여 저장"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # 그리드 레이아웃: 1행 2열 (Original + Grayscale)
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 박스 정보를 제목에 표시
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        fig.suptitle(f'OCR Preprocessing - Box #{box_idx} (pos: [{x1},{y1}], size: {w}x{h}px)',
                     fontsize=14, fontweight='bold')

        step_names = {
            '1_original': 'Original',
            '2_grayscale': 'Grayscale'
        }

        # 각 단계별 이미지 표시
        for idx, (step_key, step_name) in enumerate(step_names.items()):
            if step_key not in steps:
                continue

            img_data = steps[step_key]
            ax = fig.add_subplot(gs[0, idx])

            # 컬러 이미지는 RGB로, 그레이스케일은 gray 컬러맵으로
            if len(img_data.shape) == 3:  # RGB
                ax.imshow(img_data)
            else:  # Grayscale
                ax.imshow(img_data, cmap='gray', vmin=0, vmax=255)

            ax.set_title(step_name, fontsize=12, fontweight='bold')
            ax.axis('off')

            # 이미지 통계 정보 추가
            if len(img_data.shape) == 2:  # Grayscale
                stats_text = f'Mean: {img_data.mean():.1f}\nStd: {img_data.std():.1f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # 저장
        output_path = os.path.join(self.ocr_debug_dir, f'box_{box_idx:04d}_preprocessing.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logging.debug(f"Saved OCR preprocessing visualization: {output_path}")

    def _normalize_for_match(self, s: str) -> str:
        """텍스트 매칭용 정규화: NFKC, 소문자, 공백 정리"""
        if not s:
            return ""
        s = unicodedata.normalize('NFKC', s)
        s = ' '.join(s.lower().split())
        return s

    def _tokenize_text(self, s: str) -> List[str]:
        """한글/영문/숫자 토큰화"""
        if not s:
            return []
        return re.findall(r"[A-Za-z0-9\uAC00-\uD7A3]+", s)

    def _extract_numbers(self, s: str) -> List[str]:
        """숫자 추출"""
        if not s:
            return []
        return re.findall(r"\d+", s)

    def text_similarity(self, text1: str, text2: str) -> float:
        """개선된 텍스트 유사도 계산"""
        # 둘 다 비어있는 경우
        if not text1 and not text2:
            return 0.2
        if not text1 or not text2:
            return 0.0

        n1 = self._normalize_for_match(text1)
        n2 = self._normalize_for_match(text2)
        if n1 == n2:
            return 1.0

        # 길이 유사도
        len1, len2 = len(n1), len(n2)
        len_sim = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

        # 토큰 기반 일치도
        toks1, toks2 = self._tokenize_text(n1), self._tokenize_text(n2)
        set1, set2 = set(toks1), set(toks2)
        jacc = (len(set1 & set2) / max(1, len(set1 | set2))) if (set1 or set2) else 0.0

        # 숫자 불일치 체크
        nums1, nums2 = set(self._extract_numbers(n1)), set(self._extract_numbers(n2))
        if nums1 and nums2 and nums1.isdisjoint(nums2):
            return 0.0

        # 극단적 길이/토큰 차이 필터링
        max_len = max(len1, len2)
        min_len = min(len1, len2)
        max_tok = max(len(set1), len(set2))
        min_tok = min(len(set1), len(set2))

        if (max_len >= 12 or max_tok >= 4) and (min_len <= 3 or min_tok <= 1):
            return 0.0

        # 커버리지 기반 필터링
        coverage_long = (len(set1 & set2) / max(1, max_tok)) if max_tok > 0 else 0.0
        if len_sim < 0.6 and coverage_long < 0.6:
            return 0.0

        if jacc < 0.2 and len_sim < 0.8:
            return 0.0

        # 문자 기반 유사도
        sorted_join1 = ' '.join(sorted(toks1))
        sorted_join2 = ' '.join(sorted(toks2))
        char_sim = max(
            SequenceMatcher(None, n1, n2).ratio(),
            SequenceMatcher(None, sorted_join1, sorted_join2).ratio()
        )

        # n-gram 겹침
        def bigrams(s: str) -> set:
            return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else set()

        b1, b2 = bigrams(n1), bigrams(n2)
        ng_sim = (2 * len(b1 & b2) / max(1, len(b1) + len(b2))) if (b1 or b2) else 0.0

        # 최종 점수
        base = 0.5 * char_sim + 0.3 * jacc + 0.2 * ng_sim
        base *= (0.4 + 0.6 * len_sim)

        return float(base) if base >= 0.5 else 0.0


    def detect_boxes_yolo_multires(self,
                        pil_img: Image.Image,
                        conf_thresh: float = 0.15,
                        max_det: int = 500,
                        extract_features: bool = False,
                        resolutions: List[Tuple[int, int]] = None,
                        nms_iou_threshold: float = 0.5):
        """
        다양한 해상도로 YOLO를 실행하고 결과를 합쳐서 반환

        Args:
            pil_img: 입력 이미지
            conf_thresh: confidence 임계값
            max_det: 최대 탐지 개수
            extract_features: 특징 추출 여부
            resolutions: 사용할 해상도 리스트 [(h, w), ...]. None이면 기본값 사용
            nms_iou_threshold: NMS IOU 임계값

        Returns:
            boxes, scores, classes [, feat_maps, (orig_w, orig_h)]
        """
        if resolutions is None:
            # 기본 해상도: 작은 요소, 중간 요소, 큰 요소를 위한 다양한 스케일
            resolutions = [
                (640, 640),   # 작은 해상도 - 큰 요소에 집중
                (736, 736), # 중간 해상도 - 균형잡힌 탐지
                (1024, 1024), # 큰 해상도 - 작은 요소에 집중
            ]

        orig_w, orig_h = pil_img.size
        logging.info(f"Multi-resolution detection with {len(resolutions)} resolutions: {resolutions}")

        # 각 해상도별로 탐지 수행
        all_boxes = []
        all_scores = []
        all_classes = []
        all_feat_maps = []

        for res_idx, resolution in enumerate(resolutions):
            logging.info(f"  Processing resolution {res_idx+1}/{len(resolutions)}: {resolution}")

            # 현재 해상도로 탐지 수행
            if extract_features:
                boxes, scores, classes, feat_maps, _ = self._detect_single_resolution(
                    pil_img, resolution, conf_thresh, max_det, extract_features=True
                )
                # 특징 맵에 해상도 정보 추가 (나중에 구분하기 위해)
                all_feat_maps.append((res_idx, resolution, feat_maps))
            else:
                boxes, scores, classes = self._detect_single_resolution(
                    pil_img, resolution, conf_thresh, max_det, extract_features=False
                )

            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)
                logging.info(f"    Found {len(boxes)} boxes at resolution {resolution}")

        if not all_boxes:
            logging.warning("No boxes detected at any resolution")
            if extract_features:
                return np.array([]), np.array([]), np.array([]), [], (orig_w, orig_h)
            else:
                return np.array([]), np.array([]), np.array([])

        # 모든 결과 합치기
        merged_boxes = np.vstack(all_boxes)
        merged_scores = np.concatenate(all_scores)
        merged_classes = np.concatenate(all_classes)

        logging.info(f"Total boxes before NMS: {len(merged_boxes)}")

        # NMS로 중복 제거
        keep_indices = non_max_suppression(merged_boxes, merged_scores, nms_iou_threshold)
        final_boxes = merged_boxes[keep_indices]
        final_scores = merged_scores[keep_indices]
        final_classes = merged_classes[keep_indices]

        logging.info(f"Total boxes after NMS: {len(final_boxes)}")

        # Top-k 선택
        if len(final_boxes) > max_det:
            top_k_indices = np.argsort(-final_scores)[:max_det]
            final_boxes = final_boxes[top_k_indices]
            final_scores = final_scores[top_k_indices]
            final_classes = final_classes[top_k_indices]
            logging.info(f"Reduced to top-{max_det} boxes")

        if extract_features:
            # 특징 맵은 가장 높은 해상도의 것을 사용 (가장 디테일한 특징)
            # 또는 중간 해상도 사용 (속도와 품질 균형)
            best_feat_maps = all_feat_maps[-1][2]  # 마지막(가장 큰) 해상도의 특징 사용
            return final_boxes, final_scores, final_classes, best_feat_maps, (orig_w, orig_h)
        else:
            return final_boxes, final_scores, final_classes

    def _detect_single_resolution(self,
                        pil_img: Image.Image,
                        resize_size: Tuple[int, int],
                        conf_thresh: float = 0.15,
                        max_det: int = 500,
                        extract_features: bool = False):
        """단일 해상도에서 YOLO 탐지 수행 (내부 메서드)"""
        orig_w, orig_h = pil_img.size

        # 1) 샤픈
        pil_img_sharp = pil_img.filter(
            ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1)
        )

        # 2) 단순 리사이즈 (종횡비 유지 제거)
        img_resized = pil_img_sharp.resize(resize_size, Image.Resampling.LANCZOS)
        img_np = np.array(img_resized)  # H×W×C, RGB

        # BGR, float32, [0–1]
        img_input = img_np[..., ::-1].astype(np.float32) / 255.0
        img_input = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0)
        
        # 3) Multi-scale feature map 추출을 위한 hook
        feat_maps = []  # 여러 레이어의 특징 맵 저장
        handles = []  # hook 핸들 저장

        if extract_features:
            backbone_modules = list(self.yolo.model.model.children())

            # 여러 스테이지에서 특징 추출
            # Layer 6: 초기 특징 (에지, 텍스처) - 아이콘/Vector 구분에 유용
            # Layer 8: 중간 특징 (패턴, 형태)
            # Layer 10: 고수준 특징 (의미론적) - 텍스트 요소에 유용
            target_layers = [6, 8, 10]

            for layer_idx in target_layers:
                hook_layer = backbone_modules[layer_idx]
                handle = hook_layer.register_forward_hook(
                    lambda m, inp, out, idx=layer_idx: feat_maps.append((idx, out.detach()))
                )
                handles.append(handle)

        # 4) Inference
        results = self.yolo.predict(
            source=img_input,
            conf=conf_thresh,
            iou=0.1,
            max_det=max_det,
            verbose=False
        )[0]

        # Hook 제거 및 multi-scale feature maps 정리
        if extract_features:
            for handle in handles:
                handle.remove()

            # 레이어 인덱스순으로 정렬
            feat_maps.sort(key=lambda x: x[0])
            # feat_map은 (layer_idx, tensor) 튜플의 리스트
            feat_map = feat_maps  # [(6, tensor), (8, tensor), (10, tensor)]

        # 5) 박스 좌표 복원 (단순 스케일링)
        boxes = results.boxes.xyxy.cpu().numpy().copy()
        scale_x = orig_w / resize_size[1]
        scale_y = orig_h / resize_size[0]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        # 6) confidence 필터 → score 순 정렬 → top-k
        mask = scores >= conf_thresh
        boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
        order = scores.argsort()[::-1][:max_det]
        boxes_final = boxes[order]
        scores_final = scores[order]
        classes_final = classes[order]

        if extract_features:
            return boxes_final, scores_final, classes_final, feat_map, (orig_w, orig_h)
        else:
            return boxes_final, scores_final, classes_final

    def detect_boxes_yolo(self,
                        pil_img: Image.Image,
                        conf_thresh: float = 0.15,
                        max_det: int = 500,
                        extract_features: bool = False,
                        use_multires: bool = False,
                        resolutions: List[Tuple[int, int]] = None):
        """
        YOLO를 사용한 박스 검출

        Args:
            pil_img: 입력 이미지
            conf_thresh: confidence 임계값
            max_det: 최대 탐지 개수
            extract_features: 특징 추출 여부
            use_multires: multi-resolution detection 사용 여부
            resolutions: multi-resolution 사용 시 해상도 리스트

        Returns:
            boxes, scores, classes [, feat_maps, (orig_w, orig_h)]
        """
        if use_multires:
            # Multi-resolution detection
            return self.detect_boxes_yolo_multires(
                pil_img, conf_thresh, max_det, extract_features, resolutions
            )
        else:
            # Single resolution detection (기존 방식)
            return self._detect_single_resolution(
                pil_img, self.resize_size, conf_thresh, max_det, extract_features
            )

    @staticmethod
    def calculate_iou(rect_box1, rect_box2):
        # 입력 타입에 따라 좌표 추출
        if isinstance(rect_box1, dict):
            # 딕셔너리 형태: {"x": x, "y": y, "width": w, "height": h}
            x1_min, y1_min = rect_box1["x"], rect_box1["y"]
            x1_max = rect_box1["x"] + rect_box1["width"]
            y1_max = rect_box1["y"] + rect_box1["height"]
        elif isinstance(rect_box1, (list, tuple, np.ndarray)):
            # 리스트/튜플/배열 형태: [x1, y1, x2, y2]
            x1_min, y1_min, x1_max, y1_max = rect_box1
        else:
            raise TypeError(f"Unsupported type for rect_box1: {type(rect_box1)}")
        
        if isinstance(rect_box2, dict):
            # 딕셔너리 형태: {"x": x, "y": y, "width": w, "height": h}
            x2_min, y2_min = rect_box2["x"], rect_box2["y"]
            x2_max = rect_box2["x"] + rect_box2["width"]
            y2_max = rect_box2["y"] + rect_box2["height"]
        elif isinstance(rect_box2, (list, tuple, np.ndarray)):
            # 리스트/튜플/배열 형태: [x1, y1, x2, y2]
            x2_min, y2_min, x2_max, y2_max = rect_box2
        else:
            raise TypeError(f"Unsupported type for rect_box2: {type(rect_box2)}")

        # 2) 교집합 영역 좌표
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # 3) 교집합 너비·높이 (겹치지 않으면 0)
        inter_w = max(0.0, inter_x_max - inter_x_min)
        inter_h = max(0.0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        # 4) 각 박스 면적
        area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
        area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

        # 5) 합집합 면적
        union_area = area1 + area2 - inter_area
        if union_area <= 0:
            return 0.0

        # 6) IoU 계산
        return inter_area / union_area

    

    def extract_features_from_map(
        self,
        feat_maps: list,  # [(layer_idx, tensor), ...] 형식
        boxes: np.ndarray,
        original_img_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Multi-scale feature maps에서 박스 영역의 특징 추출 및 결합"""
        if len(boxes) == 0:
            return torch.empty(0)

        # 레거시 호환성: 단일 텐서가 전달된 경우
        if isinstance(feat_maps, torch.Tensor):
            feat_maps = [(10, feat_maps)]  # 기존 단일 레이어로 처리

        orig_w, orig_h = original_img_size
        resized_w, resized_h = self.resize_size

        # 박스 좌표를 리사이즈된 이미지 좌표로 변환
        scale_x_orig_to_resized = resized_w / orig_w
        scale_y_orig_to_resized = resized_h / orig_h

        boxes_resized_coords = boxes.copy()
        boxes_resized_coords[:, [0, 2]] *= scale_x_orig_to_resized
        boxes_resized_coords[:, [1, 3]] *= scale_y_orig_to_resized

        # 각 레이어에서 특징 추출
        multi_scale_features = []

        for layer_idx, full_feat_map in feat_maps:
            # Feature map 좌표로 변환
            feat_map_h, feat_map_w = full_feat_map.shape[2:]
            scale_x_resized_to_feat = feat_map_w / resized_w
            scale_y_resized_to_feat = feat_map_h / resized_h

            boxes_feat_map_coords = boxes_resized_coords.copy()
            boxes_feat_map_coords[:, [0, 2]] *= scale_x_resized_to_feat
            boxes_feat_map_coords[:, [1, 3]] *= scale_y_resized_to_feat

            # ROI Align을 위한 배치 인덱스 추가
            batch_indices = torch.zeros((boxes_feat_map_coords.shape[0], 1), dtype=torch.float32)
            roi_boxes = torch.cat([batch_indices, torch.from_numpy(boxes_feat_map_coords).float()], dim=1)

            # ROI Align 수행
            pooled_features = torchvision.ops.roi_align(
                full_feat_map, roi_boxes, output_size=(1, 1), spatial_scale=1.0
            )

            # Flatten
            pooled_features = pooled_features.view(pooled_features.size(0), -1)
            multi_scale_features.append(pooled_features)

        # Multi-scale 특징 결합 (Concatenation)
        combined_features = torch.cat(multi_scale_features, dim=1)

        # 정규화
        normalized_features = torch.nn.functional.normalize(combined_features, p=2, dim=1)

        return normalized_features.cpu()

    def compute_feature_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        return 0.0 if feat1.numel() == 1 or feat2.numel() == 1 else float(torch.dot(feat1, feat2))

    def calculate_similarity(self, img1: Image.Image, img2: Image.Image, figma_fare: List[FigmaFare], web_extracted: List[ExtractedElement]) -> Dict[str, np.ndarray]:
        # Extract features for Figma elements from img1
        boxes1 = np.array([f.extracted.box for f in figma_fare])

        # Extract features for Web elements from img2
        boxes2 = np.array([e.box for e in web_extracted])

        text_sim = self.calculate_text_similarity_matrix(figma_fare, web_extracted)
        feature_sim = self.calculate_feature_similarity_matrix(figma_fare, web_extracted)
        size_sim = self.calculate_size_similarity_matrix(figma_fare, web_extracted)
        coordinate_sim = self.calculate_coordinate_similarity_matrix(img1, img2, boxes1, boxes2)
        return {'text': text_sim, 'feature': feature_sim, 'size': size_sim, 'coordinate': coordinate_sim}

    def calculate_text_similarity_matrix(self, figma_fare: List[FigmaFare], web_extracted: List[ExtractedElement]) -> np.ndarray:
        """텍스트 유사도 행렬 계산"""
        textsA = [f.extracted.text for f in figma_fare]
        textsB = [e.text for e in web_extracted]
        text_sim = np.zeros((len(textsA), len(textsB)))

        for i in range(len(textsA)):
            for j in range(len(textsB)):
                text_sim[i, j] = self.text_similarity(textsA[i], textsB[j])

        return text_sim.astype(np.float32)

    def calculate_feature_similarity_matrix(
        self,
        figma_fare: List[FigmaFare],
        web_extracted: List[ExtractedElement]
    ) -> np.ndarray:
        """특징 유사도 행렬 계산"""
        N, M = len(figma_fare), len(web_extracted)
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)

        with torch.no_grad():
            # 안전한 텐서 변환
            def _to_float_tensor(x) -> torch.Tensor:
                if isinstance(x, torch.Tensor):
                    return x.detach().to(dtype=torch.float32, copy=False)
                arr = np.asarray(x)
                if not arr.flags.writeable:
                    arr = arr.copy()
                return torch.from_numpy(arr.astype(np.float32, copy=False))

            featuresA = torch.stack([
                _to_float_tensor(getattr(f.extracted, "feature"))
                for f in figma_fare
            ], dim=0)
            featuresB = torch.stack([
                _to_float_tensor(getattr(e, "feature"))
                for e in web_extracted
            ], dim=0)

            # 정규화
            featuresA = torch.nn.functional.normalize(featuresA, p=2, dim=1)
            featuresB = torch.nn.functional.normalize(featuresB, p=2, dim=1)

            # 유사도 행렬 (cosine similarity)
            sim_matrix = featuresA @ featuresB.T
            return sim_matrix.cpu().numpy().astype(np.float32)

    def calculate_size_similarity_matrix(self, figma_fare: List[FigmaFare], web_extracted: List[ExtractedElement]) -> np.ndarray:
        boxes1 = np.array([FigmaFare.extracted.box for FigmaFare in figma_fare])
        boxes2 = np.array([extracted2.box for extracted2 in web_extracted])
        size_sim = np.zeros((len(boxes1), len(boxes2)))
        for i in range(len(boxes1)):
            for j in range(len(boxes2)):
                size_sim[i, j] = ElementExtractor.calculate_iou(boxes1[i], boxes2[j])
        return size_sim

    def calculate_coordinate_similarity_matrix(self, img1: Image.Image, img2: Image.Image, boxes1: np.ndarray, boxes2: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        sim_mat = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        for i in range(len(boxes1)):
            for j in range(len(boxes2)):
                sim_mat[i, j] = self.compute_coordinate_similarity(boxes1[i], boxes2[j], img1.size, img2.size, sigma=sigma)
        return sim_mat

    def compute_coordinate_similarity(self, box1: np.ndarray, box2: np.ndarray, size1: Tuple[int,int], size2: Tuple[int,int], sigma: float = 0.2) -> float:
        cx1, cy1 = (box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0
        cx2, cy2 = (box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0
        W1, H1 = size1
        W2, H2 = size2
        nx1, ny1 = (cx1 / W1, cy1 / H1) if W1 > 0 and H1 > 0 else (0.0, 0.0)
        nx2, ny2 = (cx2 / W2, cy2 / H2) if W2 > 0 and H2 > 0 else (0.0, 0.0)
        dist_norm = np.sqrt((nx1 - nx2)**2 + (ny1 - ny2)**2)
        return float(np.exp(- (dist_norm**2) / (2 * sigma**2)))

    def resize_and_adjust_boxes(self, img: Image.Image, boxes: np.ndarray, target_size: Tuple[int, int]) -> Tuple[Image.Image, np.ndarray]:
        orig_w, orig_h = img.size
        target_w, target_h = target_size
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        if len(boxes) > 0:
            adjusted_boxes = boxes.copy()
            adjusted_boxes[:, [0, 2]] *= target_w / orig_w
            adjusted_boxes[:, [1, 3]] *= target_h / orig_h
        else:
            adjusted_boxes = boxes
        return resized_img, adjusted_boxes 
    

    def get_matches(
        self,
        sim_dict: Dict[str, np.ndarray],
        figma_elements_data: List[FigmaFare],
        web_elements_data: List[ExtractedElement],
        min_similarity: float = 0.6
    ) -> Tuple[List[MatchResult], List[MatchResult], List[MatchResult]]:
        # 1. Weighted similarity calculation
        sim_matrix = (
            # sim_dict['text'] * 0.35 +
            sim_dict['feature'] * 0.7 +
            sim_dict['size'] * 0.15 +
            sim_dict['coordinate'] * 0.15
        )

        # Save debug similarity plots
        N, M = sim_matrix.shape
        # Normalize for visualization (absolute values)
        text_abs = sim_dict['text'].copy()
        feat_abs = sim_dict['feature'].copy()
        size_abs = sim_dict['size'].copy()
        coord_abs = sim_dict['coordinate'].copy()
        combined_abs = sim_matrix.copy()

        # Normalize for visualization (relative values - normalized to [0,1])
        def normalize_matrix(mat):
            if mat.size == 0:
                return mat
            min_val, max_val = mat.min(), mat.max()
            if max_val - min_val < 1e-8:
                return np.zeros_like(mat)
            return (mat - min_val) / (max_val - min_val)

        text_rel = normalize_matrix(text_abs)
        feat_rel = normalize_matrix(feat_abs)
        size_rel = normalize_matrix(size_abs)
        coord_rel = normalize_matrix(coord_abs)
        combined_rel = normalize_matrix(combined_abs)

        self._save_debug_similarity_plots(
            text_abs, feat_abs, size_abs, coord_abs, combined_abs,
            text_rel, feat_rel, size_rel, coord_rel, combined_rel,
            N, M
        )

        # # 2. Normalize to [0,1]
        # min_val, max_val = sim_matrix.min(), sim_matrix.max()
        # sim_matrix = (sim_matrix - min_val) / (max_val - min_val + 1e-8)

        # 3. Greedy matching for high-confidence
        matches: List[MatchResult] = []
        unmatched_figma = set(range(sim_matrix.shape[0]))
        unmatched_web   = set(range(sim_matrix.shape[1]))
        temp_matrix = sim_matrix.copy()
        while True:
            i, j = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
            score = temp_matrix[i, j]
            if score < min_similarity:
                break
            print(f"figma text: {figma_elements_data[i].extracted.text}")
            print(f"web text: {web_elements_data[j].text}")
            mr = MatchResult(
                figma=figma_elements_data[i],
                web=web_elements_data[j],
                feature_similarity=float(sim_dict['feature'][i, j]),
                text_similarity=float(sim_dict['text'][i, j]),
                size_similarity=float(sim_dict['size'][i, j]),
                coordinate_similarity=float(sim_dict['coordinate'][i, j]),
                score=float(score),
                errorCategories=ErrorChecker().check_error(
                    figma_elements_data[i].extracted.box,
                    web_elements_data[j].box,
                    figma_elements_data[i].extracted.text,
                    web_elements_data[j].text
                )
            )
            matches.append(mr)
            unmatched_figma.discard(i)
            unmatched_web.discard(j)
            temp_matrix[i, :] = -np.inf
            temp_matrix[:, j] = -np.inf

        # TODO
        # 매칭된 요소들과 IOU가 큰 요소가 큰 값 제거 (NMS)
        # 그래도 매칭이 안되는 요소들을 unmatched로 설정

        print(f"unmatched_figma: {unmatched_figma}")
        print(f"unmatched_web: {unmatched_web}")
        unmatched_figma_idxs = unmatched_figma
        unmatched_web_idxs = unmatched_web

        # 4. (선택) IOU 기반 제외 처리 — 필요 없으면 건너뛰셔도 됩니다.
        for mr in matches:
            # figma 중 IOU 높으면 제거
            to_remove = {
                i for i in unmatched_figma_idxs
                if ElementExtractor.calculate_iou(
                    mr.figma.extracted.box,
                    figma_elements_data[i].extracted.box
                ) > 0.5
            }
            unmatched_figma_idxs -= to_remove

            # web 중 IOU 높으면 제거
            to_remove = {
                j for j in unmatched_web_idxs
                if ElementExtractor.calculate_iou(
                    mr.web.box,
                    web_elements_data[j].box
                ) > 0.5
            }
            unmatched_web_idxs -= to_remove

        # 5. unmatched MatchResult 생성
        unmatched_figma = [
            MatchResult(
                figma=figma_elements_data[i],
                web=None,
                feature_similarity=0.0,
                text_similarity=0.0,
                size_similarity=0.0,
                coordinate_similarity=0.0,
                score=0.0,
                errorCategories="Not Matched"
            )
            for i in sorted(unmatched_figma_idxs)
        ]
        unmatched_web = [
            MatchResult(
                figma=None,
                web=web_elements_data[j],
                feature_similarity=0.0,
                text_similarity=0.0,
                size_similarity=0.0,
                coordinate_similarity=0.0,
                score=0.0,
                errorCategories="Not Matched"
            )
            for j in sorted(unmatched_web_idxs)
        ]

        return matches, unmatched_figma, unmatched_web

    def _save_debug_similarity_plots(
        self,
        text_abs, feat_abs, size_abs, coord_abs, combined_abs,
        text_rel, feat_rel, size_rel, coord_rel, combined_rel,
        N, M
    ):
        """디버그용 유사도 히트맵 저장"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            debug_dir = os.path.join(os.path.dirname(__file__), 'debug_sim')
            os.makedirs(debug_dir, exist_ok=True)

            def save_heatmap(mat: np.ndarray, title: str, fname: str, vmin: float = 0.0, vmax: float = 1.0):
                plt.figure(figsize=(10, 7))
                sns.heatmap(mat, vmin=vmin, vmax=vmax, cmap="viridis")
                plt.title(title)
                plt.xlabel("Web Elements")
                plt.ylabel("Figma Elements")
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, fname), dpi=150)
                plt.close()

            # Absolute
            save_heatmap(text_abs, 'Text (abs)', 'text_abs.png')
            save_heatmap(feat_abs, 'Feature (abs)', 'feature_abs.png')
            save_heatmap(size_abs, 'Size (abs)', 'size_abs.png')
            save_heatmap(coord_abs, 'Coord (abs)', 'coord_abs.png')
            save_heatmap(combined_abs, 'Combined (abs)', 'combined_abs.png')

            # Relative
            save_heatmap(text_rel, 'Text (rel)', 'text_rel.png')
            save_heatmap(feat_rel, 'Feature (rel)', 'feature_rel.png')
            save_heatmap(size_rel, 'Size (rel)', 'size_rel.png')
            save_heatmap(coord_rel, 'Coord (rel)', 'coord_rel.png')
            save_heatmap(combined_rel, 'Combined (rel)', 'combined_rel.png')

            # Top-K summary
            try:
                K = min(5, M)
                lines = []
                for i in range(N):
                    order_abs = np.argsort(-combined_abs[i])[:K]
                    order_rel = np.argsort(-combined_rel[i])[:K]
                    lines.append(f"Figma {i} | ABS top{K}: " + ", ".join([f"j={int(j)}:{combined_abs[i,j]:.2f}" for j in order_abs]))
                    lines.append(f"Figma {i} | REL top{K}: " + ", ".join([f"j={int(j)}:{combined_rel[i,j]:.2f}" for j in order_rel]))
                open(os.path.join(debug_dir, 'topk.txt'), 'w').write("\n".join(lines))
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: Failed to save debug plots: {e}")
