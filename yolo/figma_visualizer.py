"""
FigmaVisualizer - 간단하고 실용적인 Figma 시각화 클래스
bbox 시각화에 특화된 최소한의 기능 제공
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional
import os

class FigmaVisualizer:
    """Figma bbox 시각화 전용 클래스"""
    
    def __init__(self):
        # 기본 스타일 설정 (커스터마이징 없음)
        self.bbox_color = 'red'
        self.bbox_thickness = 2.0
        self.bbox_alpha = 0.8
        self.figure_size = (12, 8)
    
    def show_with_boxes(self, frame, save: bool = False, output_path: str = "figma_with_boxes.png"):
        """프레임 + bbox 시각화 (화면 표시)"""
        print("Creating visualization...")
        fig, ax = self._create_visualization(frame)
        print(f"Found {self._count_total_boxes(frame.element_tree)} boxes to draw")
        
        if save:
            self._save_figure(fig, output_path)
            print(f"Saved to {output_path}")
        
        plt.tight_layout()
        print("Showing plot...")
        plt.show()
        plt.close()
        print("Plot closed")
    
    def save_with_boxes(self, frame, output_path: str):
        """프레임 + bbox 시각화 (저장만, 화면 표시 안함)"""
        fig, ax = self._create_visualization(frame)
        self._save_figure(fig, output_path)
        plt.close()
        print(f"Visualization saved to: {output_path}")
    
    def _create_visualization(self, frame):
        """시각화 생성 (내부 메서드)"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 배경 이미지 표시
        ax.imshow(frame.img)
        
        # bbox 그리기
        self._draw_all_boxes(ax, frame.element_tree)
        
        # 스타일 설정
        ax.set_title(f"Figma Frame with Bounding Boxes")
        ax.axis('off')
        
        return fig, ax
    
    def _draw_all_boxes(self, ax, tree):
        """재귀적으로 모든 bbox 그리기"""
        # 현재 트리의 박스들 그리기
        for box_pair in tree.boxes:
            # render_box 사용 (기본값)
            box = box_pair.render_box
            
            rect = patches.Rectangle(
                (box.x, box.y), 
                box.width, 
                box.height,
                linewidth=self.bbox_thickness,
                edgecolor=self.bbox_color,
                facecolor='none',
                alpha=self.bbox_alpha
            )
            ax.add_patch(rect)
        
        # 자식 트리들 재귀 처리
        for child_tree in tree.children:
            self._draw_all_boxes(ax, child_tree)
    
    def _save_figure(self, fig, output_path: str):
        """그림 파일로 저장"""
        # 디렉토리가 없으면 생성
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 고품질로 저장
        fig.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    
    def _count_total_boxes(self, tree):
        """전체 박스 개수 계산"""
        count = len(tree.boxes)
        for child in tree.children:
            count += self._count_total_boxes(child)
        return count

# 사용 예시 및 테스트
if __name__ == "__main__":
    # 사용법 예시
    print("FigmaVisualizer 사용법:")
    print("1. visualizer = FigmaVisualizer()")
    print("2. visualizer.show_with_boxes(frame)  # 화면에 표시")
    print("3. visualizer.show_with_boxes(frame, save=True)  # 표시 + 저장")
    print("4. visualizer.save_with_boxes(frame, 'output.png')  # 저장만")