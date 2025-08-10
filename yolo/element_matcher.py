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
    MatchResult, ExtractedElement
)

class ElementExtractor:
    """요소 매칭을 수행하는 클래스"""
    def __init__(self, yolo_model_path: str = None, resize_size: Tuple[int, int] = (736, 736)):
        current_dir = os.path.dirname(__file__)
        if yolo_model_path is None:
            yolo_model_path = os.path.join(current_dir, "best_forest.pt")
        self.yolo = YOLO(yolo_model_path, task='detect', verbose=False)
        self.resize_size = resize_size
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

    def __del__(self):
        """tesserocr API 정리"""
        if hasattr(self, 'api'):
            self.api.End()

    def extract_text(self, img: Image.Image, box: np.ndarray) -> str:
        x1, y1, x2, y2 = map(int, box)
        crop_img = img.crop((x1, y1, x2, y2))
        img_np = np.array(crop_img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # 노이즈 제거 (선택적)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        # 대비 향상 (선택적)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # tesserocr 사용
        binary_pil = Image.fromarray(binary)
        self.api.SetImage(binary_pil)
        text = self.api.GetUTF8Text()
        text = ' '.join(text.split())
        text = ''.join(c for c in text if c.isalnum() or c.isspace() or '\uAC00' <= c <= '\uD7A3')
        return text.strip()

    def text_similarity(self, text1: str, text2: str) -> float:
        if not text1 and not text2: return 1.0
        if not text1 or not text2: return -1.0
        if text1 == text2: return 1.0
        len1, len2 = len(text1), len(text2)
        len_sim = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        if text1 in text2 or text2 in text1: return 0.7 + (0.2 * len_sim)
        content_sim = SequenceMatcher(None, text1, text2).ratio()
        similarity = content_sim * len_sim
        return similarity if similarity >= 0.5 else -1.0


    def detect_boxes_yolo(self,
                        pil_img: Image.Image,
                        conf_thresh: float = 0.15,
                        max_det: int = 500,
                        extract_features: bool = False):
        orig_w, orig_h = pil_img.size
        resize_size = self.resize_size  # e.g. (640,640) 혹은 (1024,1024)

        # 1) 샤픈
        pil_img = pil_img.filter(
            ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1)
        )

        # 2) 단순 리사이즈 (종횡비 유지 제거)
        img_resized = pil_img.resize(resize_size, Image.Resampling.LANCZOS)
        img_np = np.array(img_resized)  # H×W×C, RGB
        
        # BGR, float32, [0–1]
        img_input = img_np[..., ::-1].astype(np.float32) / 255.0
        img_input = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0)
        
        # 3) Hook backbone to extract feature map during inference
        feat_map = None
        feature_map_storage = []
        if extract_features:
            # Hook the actual backbone layer (SPPF at index 9) to capture output
            backbone_modules = list(self.yolo.model.model.children())
            hook_layer = backbone_modules[9]  # SPPF
            handle = hook_layer.register_forward_hook(
                lambda m, inp, out: feature_map_storage.append(out.detach())
            )
        # 4) Inference (remove hook after prediction)
        results = self.yolo.predict(
            source=img_input,
            conf=conf_thresh,
            iou=0.3,
            max_det=max_det,
            verbose=False
        )[0]        # 4) Remove hook and retrieve feature map
        if extract_features:
            handle.remove()
            feat_map = feature_map_storage[0]

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

    

    def extract_features_from_map(self, full_feat_map: torch.Tensor, boxes: np.ndarray, original_img_size: Tuple[int, int]) -> torch.Tensor:
        if len(boxes) == 0:
            return torch.empty(0)

        orig_w, orig_h = original_img_size
        resized_w, resized_h = self.resize_size

        # Scale boxes from original image coordinates to resized image coordinates
        scale_x_orig_to_resized = resized_w / orig_w
        scale_y_orig_to_resized = resized_h / orig_h

        boxes_resized_coords = boxes.copy()
        boxes_resized_coords[:, [0, 2]] *= scale_x_orig_to_resized
        boxes_resized_coords[:, [1, 3]] *= scale_y_orig_to_resized

        # Convert resized image coordinates to feature map coordinates
        feat_map_h, feat_map_w = full_feat_map.shape[2:]
        scale_x_resized_to_feat = feat_map_w / resized_w
        scale_y_resized_to_feat = feat_map_h / resized_h

        boxes_feat_map_coords = boxes_resized_coords.copy()
        boxes_feat_map_coords[:, [0, 2]] *= scale_x_resized_to_feat
        boxes_feat_map_coords[:, [1, 3]] *= scale_y_resized_to_feat

        # Add batch index for roi_align
        batch_indices = torch.zeros((boxes_feat_map_coords.shape[0], 1), dtype=torch.float32)
        roi_boxes = torch.cat([batch_indices, torch.from_numpy(boxes_feat_map_coords).float()], dim=1)

        # Perform ROI Align
        pooled_features = torchvision.ops.roi_align(full_feat_map, roi_boxes, output_size=(1, 1), spatial_scale=1.0)
        
        # Flatten and normalize
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        normalized_features = torch.nn.functional.normalize(pooled_features, p=2, dim=1)
        
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
        textsA = [FigmaFare.extracted.text for FigmaFare in figma_fare]
        textsB = [extracted2.text for extracted2 in web_extracted]
        text_sim = np.zeros((len(textsA), len(textsB)))
        for i in range(len(textsA)):
            for j in range(len(textsB)):
                text_sim[i, j] = self.text_similarity(textsA[i], textsB[j])
        text_sim = np.where(text_sim > 0.7, text_sim, -1)
        return text_sim

    def calculate_feature_similarity_matrix(
        self,
        figma_fare: List[FigmaFare],
        web_extracted: List[ExtractedElement],
    ) -> np.ndarray:
        # 1) numpy.ndarray인 feature를 Tensor로 변환
        tensor_list_A = [
            torch.tensor(f.extracted.feature, dtype=torch.float32)
            for f in figma_fare
        ]
        tensor_list_B = [
            torch.tensor(e.feature, dtype=torch.float32)
            for e in web_extracted
        ]

        # 2) stack 후 행렬 곱
        featuresA = torch.stack(tensor_list_A, dim=0)  # (N, D)
        featuresB = torch.stack(tensor_list_B, dim=0)  # (M, D)

        sim_matrix = featuresA @ featuresB.T          # (N, M) 유사도 행렬
        # 3) 먼저 numpy array로 변환한 후 필터링
        sim_matrix = sim_matrix.cpu().numpy()
        sim_matrix = np.where(sim_matrix > 0.7, sim_matrix, -1)
        return sim_matrix

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
    

    def temp_matches(self, sim_dict, figma_elements_data, web_elements_data, min_similarity: float = 0.8):
        matrix = (sim_dict['text'] * 0.35 + sim_dict['feature'] * 0.35 + sim_dict['size'] * 0.2 + sim_dict['coordinate'] * 0.1)

        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-8)
        if matrix.size == 0 or np.max(matrix) < min_similarity:
            return []

        box_fair = []
        sim_matrix = matrix.copy()
        max_sim = np.max(sim_matrix)

        while max_sim > min_similarity:
            max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            i, j = max_idx
            box_fair.append((i, j))
            sim_matrix[i, :] = 0
            sim_matrix[:, j] = 0
            max_sim = np.max(sim_matrix)
        result = []
        for i, j in box_fair:
            figma_element = figma_elements_data[i]
            web_element = web_elements_data[j]
        
            # MatchResult 객체 생성
            match_result = MatchResult(
                figma=figma_element,
                web=web_element,
                feature_similarity=sim_dict['feature'][i, j],
                text_similarity=sim_dict['text'][i, j],
                size_similarity=sim_dict['size'][i, j],
                coordinate_similarity=sim_dict['coordinate'][i, j],
                score=float(matrix[i, j]),
                errorCategories=[]
            )
            result.append(match_result)
            
        return result

    def get_matches(
        self,
        sim_dict: Dict[str, np.ndarray],
        figma_elements_data: List[FigmaFare],
        web_elements_data: List[ExtractedElement],
        min_similarity: float = 0.8
    ) -> Tuple[List[MatchResult], List[MatchResult], List[MatchResult]]:
        # 1. Weighted similarity calculation
        sim_matrix = (
            sim_dict['text'] * 0.35 +
            sim_dict['feature'] * 0.35 +
            sim_dict['size'] * 0.15 +
            sim_dict['coordinate'] * 0.15
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
