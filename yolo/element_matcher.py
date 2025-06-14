import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from ultralytics import YOLO
import pytesseract
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import os
from scipy.optimize import linear_sum_assignment
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules.conv import Conv
from torch.nn import Conv2d

@dataclass
class MatchResult:
    """매칭 결과를 저장하는 데이터 클래스"""
    figma_id: str
    figma_name: str
    figma_box: Tuple[float, float, float, float]
    web_box: Tuple[float, float, float, float]
    score: float
    text_score: float
    feature_score: float
    coordinate_score: float
    size_score: float
    visual_score: float
    interaction: str
    figma_dest: str
    web_dest: str
    errorCategories: List[str]

class ElementMatcher:
    """요소 매칭을 수행하는 클래스"""
    def __init__(self, yolo_model_path: str = None):
        current_dir = os.path.dirname(__file__)
        yolo_model_path = os.path.join(current_dir, "best.pt")
        if yolo_model_path is None:
            yolo_model_path = 'best.pt'
        # YOLO 모델 초기화
        self.yolo = YOLO(yolo_model_path, task='detect')
        
        # Tesseract OCR 설정 - 경로 수정
        pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # M1 Mac 경로
        self.config = '--oem 3 --psm 6 -l kor+eng'  # 한글+영어 지원
        
        # 특징 추출 모델 초기화
        self.feature_extractor = self.yolo.model.model[:12]
        self.feature_extractor.eval()
        
        # 이미지 전처리
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_text(self, img: Image.Image, box: np.ndarray) -> str:
        """박스 영역에서 텍스트 추출 및 전처리"""
        x1, y1, x2, y2 = map(int, box)
        crop_img = img.crop((x1, y1, x2, y2))
        
        # 이미지 전처리로 OCR 정확도 향상
        img_np = np.array(crop_img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Tesseract OCR 수행
        text = pytesseract.image_to_string(binary, config=self.config)
        
        # 텍스트 전처리
        text = ' '.join(text.split())  # 연속된 공백 제거
        text = ''.join(c for c in text if c.isalnum() or c.isspace() or '\uAC00' <= c <= '\uD7A3')
        
        return text.strip()

    def text_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트의 유사도 계산 (텍스트 길이 고려)"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return -1.0

        # 정확히 일치하는 경우
        if text1 == text2:
            return 1.0
            
        # 텍스트 길이 유사도 계산
        len1, len2 = len(text1), len(text2)
        len_sim = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        
        # 부분 문자열 포함 여부 확인
        if text1 in text2 or text2 in text1:
            # 길이가 비슷할수록 더 높은 점수
            return 0.7 + (0.2 * len_sim)
            
        # SequenceMatcher로 유사도 계산
        content_sim = SequenceMatcher(None, text1, text2).ratio()
        
        # 최종 유사도 = 내용 유사도 * 길이 유사도
        similarity = content_sim * len_sim
        
        # 유사도가 낮은 경우 음수로 변환
        if similarity < 0.5:  # 임계값 설정
            return -1.0
        
        return similarity

    def detect_boxes_yolo(self, pil_img: Image.Image, conf_thresh: float = 0.3, max_det: int = 500) -> np.ndarray:
        """YOLO 모델로 박스 검출"""
        results = self.yolo(pil_img)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy()
        keep = scores >= conf_thresh
        return boxes[keep][:max_det], cls[keep][:max_det]

    def compute_iou(self, boxA, boxB):
        """두 박스의 IoU 계산"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        return interArea / unionArea if unionArea > 0 else 0

    def extract_features(self, img: Image.Image, box: np.ndarray) -> torch.Tensor:
        """Extract per-box embedding via YOLOv8's built-in embed argument."""
        x1, y1, x2, y2 = map(int, box)
        crop = img.crop((x1, y1, x2, y2)).convert('RGB')
        crop_t = self.transform(crop).unsqueeze(0).to('cpu')

        feat_map = self.feature_extractor(crop_t)
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            feat_map.unsqueeze(0), (1, 1)
        ).view(-1)

        feat = torch.nn.functional.normalize(pooled, p=2, dim=0)
        return feat.cpu()

    def compute_feature_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """두 특징 벡터의 코사인 유사도 계산"""
        if feat1.numel() == 1 or feat2.numel() == 1:  # 0 벡터인 경우
            return 0.0
        return float(torch.dot(feat1, feat2))

    def calculate_similarity(self, img1: Image.Image, img2: Image.Image, data1: List[dict], data2: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """두 이미지의 요소들을 매칭"""
        boxes1 = []
        boxes2 = []
        for item in data1:
            boxes1.extend(item['boxes'])
        for item in data2:
            boxes2.extend(item['boxes'])

        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # 리스트를 numpy 배열로 변환
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        # 두 이미지의 크기 조정
        target_size = (min(img1.width, img2.width), min(img1.height, img2.height))
        resized_img1, adjusted_boxes1 = self.resize_and_adjust_boxes(img1, boxes1, target_size)
        resized_img2, adjusted_boxes2 = self.resize_and_adjust_boxes(img2, boxes2, target_size)

        # 텍스트 유사도 계산
        text_sim = self.calculate_text_similarity_matrix(data1, data2)
        feature_sim = self.calculate_feature_similarity_matrix(data1, data2)    
        size_sim = self.calculate_size_similarity_matrix(data1, data2)
        coordinate_sim = self.calculate_coordinate_similarity_matrix(resized_img1, resized_img2, adjusted_boxes1, adjusted_boxes2)

        return text_sim, feature_sim, size_sim, coordinate_sim

    
    def calculate_text_similarity_matrix(self, data1: List[dict], data2: List[dict]) -> np.ndarray:
        """두 이미지의 요소들을 매칭"""
        textsA = []
        textsB = []
        for item in data1:
            textsA.extend(item['text'])
        for item in data2:
            textsB.extend(item['text'])
        text_sim = np.zeros((len(textsA), len(textsB)))

        for i in range(len(textsA)):
            for j in range(len(textsB)):
                text_sim[i, j] = self.text_similarity(textsA[i], textsB[j])
        return text_sim

    def calculate_feature_similarity_matrix(self, data1: List[dict], data2: List[dict]) -> np.ndarray:
        """두 이미지의 요소들을 매칭"""
        featuresA = []
        featuresB = []
        for item in data1:
            featuresA.extend(item['feature'])
        for item in data2:
            featuresB.extend(item['feature'])
            
        # 특징 벡터들을 numpy 배열로 변환
        featuresA = torch.stack(featuresA).numpy()
        featuresB = torch.stack(featuresB).numpy()
        features_matrix = featuresA @ featuresB.T
        return features_matrix

    def calculate_size_similarity_matrix(self, data1: List[dict], data2: List[dict]) -> np.ndarray:
        """두 이미지의 요소들을 매칭"""
        # 모든 박스를 하나의 리스트로 모음
        boxes1 = []
        boxes2 = []
        for item in data1:
            boxes1.extend(item['boxes'])
        for item in data2:
            boxes2.extend(item['boxes'])
        
        # numpy 배열로 변환
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        
        # IoU 행렬 계산
        size_sim = np.zeros((len(boxes1), len(boxes2)))
        for i in range(len(boxes1)):
            for j in range(len(boxes2)):
                size_sim[i, j] = self.compute_iou(boxes1[i], boxes2[j])
        return size_sim

    def calculate_coordinate_similarity_matrix(self, img1: Image.Image, img2: Image.Image, boxes1: np.ndarray, boxes2: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """두 이미지의 요소들을 매칭"""
        n = len(boxes1)
        m = len(boxes2)
        size1 = img1.size
        size2 = img2.size

        sim_mat = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                sim_mat[i, j] = self.compute_coordinate_similarity(
                    boxes1[i],
                    boxes2[j],
                    size1,
                    size2,
                    sigma=sigma
                )
        return sim_mat

    def compute_coordinate_similarity(self, box1: np.ndarray, box2: np.ndarray, size1: Tuple[int,int], size2: Tuple[int,int], sigma: float = 0.2) -> float:
        """두 박스의 중심점을 각 이미지 크기로 정규화한 뒤, Gaussian RBF 커널로 유사도(0~1)를 계산."""
        cx1 = (box1[0] + box1[2]) / 2.0
        cy1 = (box1[1] + box1[3]) / 2.0
        cx2 = (box2[0] + box2[2]) / 2.0
        cy2 = (box2[1] + box2[3]) / 2.0

        W1, H1 = size1
        W2, H2 = size2
        if W1 > 0 and H1 > 0:
            nx1 = cx1 / W1
            ny1 = cy1 / H1
        else:
            nx1, ny1 = 0.0, 0.0

        if W2 > 0 and H2 > 0:
            nx2 = cx2 / W2
            ny2 = cy2 / H2
        else:
            nx2, ny2 = 0.0, 0.0

        dx = nx1 - nx2
        dy = ny1 - ny2
        dist_norm = np.sqrt(dx * dx + dy * dy)

        sim = np.exp(- (dist_norm * dist_norm) / (2 * sigma * sigma))
        return float(sim)

    def resize_and_adjust_boxes(self, img: Image.Image, boxes: np.ndarray, target_size: Tuple[int, int]) -> Tuple[Image.Image, np.ndarray]:
        """이미지 크기를 조정하고 박스 좌표를 조정"""
        orig_w, orig_h = img.size
        target_w, target_h = target_size
        
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        if len(boxes) > 0:
            adjusted_boxes = boxes.copy()
            adjusted_boxes[:, 0] = boxes[:, 0] * target_w / orig_w
            adjusted_boxes[:, 1] = boxes[:, 1] * target_h / orig_h
            adjusted_boxes[:, 2] = boxes[:, 2] * target_w / orig_w
            adjusted_boxes[:, 3] = boxes[:, 3] * target_h / orig_h
        else:
            adjusted_boxes = boxes
            
        return resized_img, adjusted_boxes 
    
    def get_matches(self, sim_dict, figma_boxes, web_boxes, min_similarity: float = 0.8):
        """매칭 결과를 반환합니다."""
        #matrix: numpy array of shape (n_f, n_w)
        # If the matrix is empty or the highest similarity is below threshold, return empty list
        matrix = (sim_dict['text'] * 0.35 + sim_dict['feature'] * 0.35 + sim_dict['size'] * 0.15 + sim_dict['coordinate'] * 0.15)
        # normalize
        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-8)
        if matrix.size == 0 or np.max(matrix) < min_similarity:
            return []

        box_fair = []
        feature_sim_matrix = np.zeros(sim_dict['feature'].shape)
        text_sim_matrix = np.zeros(sim_dict['text'].shape)
        size_sim_matrix = np.zeros(sim_dict['size'].shape)
        coordinate_sim_matrix = np.zeros(sim_dict['coordinate'].shape)
        sim_matrix = matrix.copy()
        max_sim = np.max(sim_matrix)

        while max_sim > min_similarity:
            # 2d argmax
            max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            i, j = max_idx
            box_fair.append((i, j))

            feature_sim_matrix[i, j] = sim_dict['feature'][i, j]
            text_sim_matrix[i, j] = sim_dict['text'][i, j]
            size_sim_matrix[i, j] = sim_dict['size'][i, j]
            coordinate_sim_matrix[i, j] = sim_dict['coordinate'][i, j]
            visual_score = sim_matrix
            # Zero out the matched row and column
            sim_matrix[i, :] = 0
            sim_matrix[:, j] = 0

            max_sim = np.max(sim_matrix)

        # Build MatchResult list using only pairs above threshold
        matches = [MatchResult(
            figma_id='',
            figma_name='',
            figma_box=tuple(figma_boxes[i]),
            web_box=tuple(web_boxes[j]),
            score=float(matrix[i, j]),  
            text_score=text_sim_matrix[i, j],
            feature_score=feature_sim_matrix[i, j],
            coordinate_score=coordinate_sim_matrix[i, j],
            size_score=size_sim_matrix[i, j],
            visual_score=visual_score,
            interaction='',
            figma_dest='',
            web_dest='',
            errorCategories=[]
        ) for i, j in box_fair]

        return matches, sim_matrix