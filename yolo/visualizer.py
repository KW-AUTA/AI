import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
from typing import List
from .element_matcher import MatchResult

class Visualizer:
    """시각화를 담당하는 클래스"""
    @staticmethod
    def visualize_boxes(img: Image.Image, boxes: np.ndarray, title: str = "Detected Boxes"):
        """박스 시각화"""
        plt.figure(figsize=(20,10))
        plt.imshow(img)
        for box in boxes:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                                edgecolor='red', facecolor='none', linewidth=1))
            plt.text(x1, y1-5, f"({int(x1)},{int(y1)})", color='white', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))
        plt.title(title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def visualize_matches(figma_img: Image.Image, web_img: Image.Image, 
                         matches: List[MatchResult], label: str):
        """매칭 결과 시각화 (두 이미지를 이어붙이고, 매칭된 박스 중심끼리 선 연결)"""
        if not matches:
            print("\n[실패] 매칭된 요소가 없습니다.")
            return

        # 이미지 크기 맞추기 (세로 기준)
        h = max(figma_img.height, web_img.height)
        wF, wW = figma_img.width, web_img.width
        figma_np = np.array(figma_img)
        web_np = np.array(web_img)
        if figma_img.height < h:
            pad = np.zeros((h - figma_img.height, wF, 3), dtype=figma_np.dtype)
            figma_np = np.vstack([figma_np, pad])
        if web_img.height < h:
            pad = np.zeros((h - web_img.height, wW, 3), dtype=web_np.dtype)
            web_np = np.vstack([web_np, pad])
        # 이어붙이기
        concat_img = np.hstack([figma_np, web_np])
        overlay = concat_img.copy()

        # 색상
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(matches), 3)).tolist()

        for idx, (result, color) in enumerate(zip(matches, colors), 1):
            # Figma 박스
            x1, y1, x2, y2 = map(int, result.figma_box)
            # Web 박스
            wx1, wy1, wx2, wy2 = map(int, result.web_box)
            # 박스 그리기
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(overlay, (wx1 + wF, wy1), (wx2 + wF, wy2), color, 2)
            # 중심점
            cxF, cyF = (x1 + x2) // 2, (y1 + y2) // 2
            cxW, cyW = (wx1 + wx2) // 2 + wF, (wy1 + wy2) // 2
            # 선 그리기
            cv2.line(overlay, (cxF, cyF), (cxW, cyW), color, 2)
            # 텍스트
            cv2.putText(overlay, f"#{idx}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(overlay, f"#{idx}", (wx1 + wF, wy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # BGR->RGB 변환
        overlay_rgb = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20,10))
        plt.imshow(overlay_rgb)
        plt.axis('off')
        plt.title(f'Matching Visualization {label}')
        plt.show()
        print(f"\n[완료] 총 {len(matches)}개의 요소가 매칭되었습니다.") 

    @staticmethod
    def visualize_matches_with_error(figma_img: Image.Image, web_img: Image.Image, 
                            matches: List[MatchResult], label: str):
        """매칭 결과 시각화 (두 이미지를 이어붙이고, 매칭된 박스 중심끼리 선 연결)"""
        if not matches:
            print("\n[실패] 매칭된 요소가 없습니다.")
            return

        # 이미지 크기 맞추기 (세로 기준)
        h = max(figma_img.height, web_img.height)
        wF, wW = figma_img.width, web_img.width
        figma_np = np.array(figma_img)
        web_np = np.array(web_img)
        if figma_img.height < h:
            pad = np.zeros((h - figma_img.height, wF, 3), dtype=figma_np.dtype)
            figma_np = np.vstack([figma_np, pad])
        if web_img.height < h:
            pad = np.zeros((h - web_img.height, wW, 3), dtype=web_np.dtype)
            web_np = np.vstack([web_np, pad])
        # 이어붙이기
        concat_img = np.hstack([figma_np, web_np])
        overlay = concat_img.copy()

        # 색상
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(matches), 3)).tolist()

        for idx, (result, color) in enumerate(zip(matches, colors), 1):
            # Figma 박스
            x1, y1, x2, y2 = map(int, result.figma_box)
            # Web 박스
            wx1, wy1, wx2, wy2 = map(int, result.web_box)
            # 박스 그리기
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(overlay, (wx1 + wF, wy1), (wx2 + wF, wy2), color, 2)
            # 중심점
            cxF, cyF = (x1 + x2) // 2, (y1 + y2) // 2
            cxW, cyW = (wx1 + wx2) // 2 + wF, (wy1 + wy2) // 2
            # 선 그리기
            cv2.line(overlay, (cxF, cyF), (cxW, cyW), color, 2)

        # BGR->RGB 변환
        overlay_rgb = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20,10))
        plt.imshow(overlay_rgb)
        plt.axis('off')
        plt.title(f'Matching Visualization error {matches[0].errorCategories}')
        plt.show()
        print(f"\n[완료] 총 {len(matches)}개의 요소가 매칭되었습니다.") 