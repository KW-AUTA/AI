import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
from .element_matcher import MatchResult
import os
import time

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
        
        concat_img = np.hstack([figma_np, web_np])
        overlay = concat_img.copy()

        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(matches), 3)).tolist()

        for idx, (result, color) in enumerate(zip(matches, colors), 1):
            x1, y1, x2, y2 = map(int, result.figma.extracted.box)
            wx1, wy1, wx2, wy2 = map(int, result.web.box)
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(overlay, (wx1 + wF, wy1), (wx2 + wF, wy2), color, 2)
            
            cxF, cyF = (x1 + x2) // 2, (y1 + y2) // 2
            cxW, cyW = (wx1 + wx2) // 2 + wF, (wy1 + wy2) // 2
            
            cv2.line(overlay, (cxF, cyF), (cxW, cyW), color, 2)
            cv2.putText(overlay, f"#{idx}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(overlay, f"#{idx}", (wx1 + wF, wy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
        if not matches:
            print("\n[실패] 매칭된 요소가 없습니다.")
            return

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
        
        concat_img = np.hstack([figma_np, web_np])
        overlay = concat_img.copy()

        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(matches), 3)).tolist()

        for idx, (result, color) in enumerate(zip(matches, colors), 1):
            x1, y1, x2, y2 = map(int, result.figma.extracted.box)
            wx1, wy1, wx2, wy2 = map(int, result.web.box)
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(overlay, (wx1 + wF, wy1), (wx2 + wF, wy2), color, 2)
            
            cxF, cyF = (x1 + x2) // 2, (y1 + y2) // 2
            cxW, cyW = (wx1 + wx2) // 2 + wF, (wy1 + wy2) // 2
            
            cv2.line(overlay, (cxF, cyF), (cxW, cyW), color, 2)

        overlay_rgb = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20,10))
        plt.imshow(overlay_rgb)
        plt.axis('off')
        plt.title(f'Matching Visualization error {matches[0].errorCategories}')
        plt.show()
        print(f"\n[완료] 총 {len(matches)}개의 요소가 매칭되었습니다.")

    @staticmethod
    def _save_match_crops(figma_img: Image.Image, web_img: Image.Image, matches: List[MatchResult], out_root: Optional[str] = None) -> str:
        """Save side-by-side crops for each match with score annotations."""
        def _clamp_box(box, w, h):
            x1, y1, x2, y2 = map(float, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1: x2 = min(w, x1 + 1)
            if y2 <= y1: y2 = min(h, y1 + 1)
            return int(x1), int(y1), int(x2), int(y2)

        ts = time.strftime('%Y%m%d-%H%M%S')
        if out_root is None:
            out_root = os.path.join(os.path.dirname(__file__), 'debug_sim', 'crops', ts)
        os.makedirs(out_root, exist_ok=True)

        fw, fh = figma_img.size
        ww, wh = web_img.size

        def _safe_name(s: str) -> str:
            return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in (s or ''))[:40]

        for idx, m in enumerate(matches):
            if m.figma is None or m.web is None:
                continue
            fx1, fy1, fx2, fy2 = _clamp_box(m.figma.extracted.box, fw, fh)
            wx1, wy1, wx2, wy2 = _clamp_box(m.web.box, ww, wh)
            crop_f = figma_img.crop((fx1, fy1, fx2, fy2))
            crop_w = web_img.crop((wx1, wy1, wx2, wy2))

            # Compose side-by-side with header for text
            pad = 6
            header_h = 54
            h = max(crop_f.height, crop_w.height) + header_h + pad * 2
            w = crop_f.width + crop_w.width + pad * 3
            canvas = Image.new('RGB', (w, h), (30, 30, 30))
            draw = ImageDraw.Draw(canvas)

            # Header text: scores and texts
            header = (
                f"score={m.score:.2f} | feat={m.feature_similarity:.2f} "
                f"text={m.text_similarity:.2f} size={m.size_similarity:.2f} coord={m.coordinate_similarity:.2f}"
            )
            sub = (
                f"fig: '{(m.figma.extracted.text or '').strip()}'  |  web: '{(m.web.text or '').strip()}'"
            )
            draw.text((pad, pad), header, fill=(255, 255, 255))
            draw.text((pad, pad + 24), sub, fill=(200, 200, 200))

            # Paste crops
            y0 = header_h + pad
            x = pad
            canvas.paste(crop_f, (x, y0))
            x += crop_f.width + pad
            canvas.paste(crop_w, (x, y0))

            fname = f"{idx:03d}_{_safe_name(m.figma.name)}.png" if hasattr(m.figma, 'name') else f"{idx:03d}.png"
            canvas.save(os.path.join(out_root, fname))

        return out_root
