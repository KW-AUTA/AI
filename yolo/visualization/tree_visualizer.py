import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import base64
import io
import requests
import sys
from typing import List, Dict, Optional
from ..utils.tree_loader import TreeNode
from ..core.models import FigmaElement

class TreeNodeVisualizer:
    """
    TreeNode 객체를 직접 사용하여 Figma 요소들을 시각화하는 클래스
    """
    
    def __init__(self, tree_node: TreeNode, interactions: List[Dict] = None):
        """
        TreeNode 객체로 초기화
        """
        self.tree_node = tree_node
        self.interactions = interactions or []
    
    @staticmethod
    def _get_image(img_ref):
        """
        base64 문자열이나 URL에서 이미지를 PIL Image로 변환
        """
        if not img_ref or not isinstance(img_ref, str):
            return None

        if img_ref.startswith("http"):  # URL인 경우
            try:
                response = requests.get(img_ref, stream=True)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            except requests.exceptions.RequestException as e:
                print(f"Warning: Could not download image from URL {img_ref}. Error: {e}", file=sys.stderr)
                return None

        if img_ref.startswith("data:image"):
            img_str = img_ref.split(",", 1)[1]
        else:
            img_str = img_ref

        img_str = img_str.replace('\n', '').replace('\r', '').replace(' ', '')
        missing_padding = len(img_str) % 4
        if missing_padding:
            img_str += '=' * (4 - missing_padding)

        try:
            img_bytes = base64.b64decode(img_str)
            return Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            print(f"Warning: Could not decode or open image from base64 string. Error: {e}", file=sys.stderr)
            return None
    
    def _extract_boxes_from_node(self, node: TreeNode) -> List[Dict]:
        """
        TreeNode에서 모든 자식 노드의 바운딩 박스 정보를 추출
        """
        boxes = []
        
        # 현재 노드의 박스 정보 추가
        if hasattr(node.data, 'absolute_position'):
            pos = node.data.absolute_position
            boxes.append({
                "x": pos.get("x", 0),
                "y": pos.get("y", 0),
                "width": pos.get("width", 0),
                "height": pos.get("height", 0),
                "name": node.data.name
            })
        
        # 자식 노드들의 박스 정보 재귀적으로 추가
        for child in node.children:
            boxes.extend(self._extract_boxes_from_node(child))
        
        return boxes
    
    def visualize_tree_structure(self):
        """
        트리 구조를 시각화 (박스 없이)
        """
        def print_node(node: TreeNode, level: int = 0):
            indent = "  " * level
            print(f"{indent}├─ {node.data.name} (ID: {node.data.id})")
            for child in node.children:
                print_node(child, level + 1)
        
        print("Figma Tree Structure:")
        print_node(self.tree_node)
    
    def visualize_with_boxes(self, background_image: Image.Image = None):
        """
        트리 노드들의 바운딩 박스를 시각화
        """
        if background_image is None:
            print("배경 이미지가 필요합니다.")
            return
        
        all_boxes = self._extract_boxes_from_node(self.tree_node)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(background_image)
        ax.set_title(f"TreeNode Visualization: {self.tree_node.data.name}")
        ax.axis('off')
        
        # 각 박스를 그리기
        for i, bbox in enumerate(all_boxes):
            rect = patches.Rectangle(
                (bbox["x"], bbox["y"]), 
                bbox["width"], 
                bbox["height"],
                linewidth=2, 
                edgecolor='red', 
                facecolor='none',
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # 박스 중앙에 이름 표시
            center_x = bbox["x"] + bbox["width"] / 2
            center_y = bbox["y"] + bbox["height"] / 2
            ax.text(center_x, center_y, bbox["name"], 
                   ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_node_hierarchy(self):
        """
        노드 계층 구조를 트리 형태로 시각화
        """
        def get_node_info(node: TreeNode) -> str:
            return f"{node.data.name}\n({node.data.id})"
        
        def create_tree_plot(node: TreeNode, ax, x=0, y=0, level=0):
            # 현재 노드 그리기
            ax.text(x, y, get_node_info(node), 
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                   fontsize=10)
            
            if not node.children:
                return
            
            # 자식 노드들 그리기
            child_spacing = 2.0
            total_width = len(node.children) * child_spacing
            start_x = x - total_width / 2 + child_spacing / 2
            
            for i, child in enumerate(node.children):
                child_x = start_x + i * child_spacing
                child_y = y - 1.5
                
                # 연결선 그리기
                ax.plot([x, child_x], [y-0.3, child_y+0.3], 'k-', alpha=0.5)
                
                # 자식 노드 재귀적으로 그리기
                create_tree_plot(child, ax, child_x, child_y, level + 1)
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        create_tree_plot(self.tree_node, ax)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-10, 2)
        ax.set_title("TreeNode Hierarchy Visualization")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def find_node_by_id(self, node_id: str, node: TreeNode = None) -> Optional[TreeNode]:
        """
        ID로 노드를 찾기
        """
        if node is None:
            node = self.tree_node
        
        if node.data.id == node_id:
            return node
        
        for child in node.children:
            result = self.find_node_by_id(node_id, child)
            if result:
                return result
        
        return None
    
    def visualize_interaction(self, interaction: Dict, background_image: Image.Image = None):
        """
        특정 상호작용을 시각화
        """
        if background_image is None:
            print("배경 이미지가 필요합니다.")
            return
        
        interaction_type = interaction.get("interactionType", {})
        source_id = interaction_type.get("sourceId")
        
        if not source_id:
            print("상호작용에 sourceId가 없습니다.")
            return
        
        source_node = self.find_node_by_id(source_id)
        if not source_node:
            print(f"ID {source_id}를 가진 소스 노드를 찾을 수 없습니다.")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(background_image)
        ax.set_title(f"Interaction: {source_node.data.name}")
        ax.axis('off')
        
        # 소스 노드 박스 그리기
        pos = source_node.data.absolute_position
        rect = patches.Rectangle(
            (pos["x"], pos["y"]), 
            pos["width"], 
            pos["height"],
            linewidth=3, 
            edgecolor='red', 
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # 노드 이름 표시
        center_x = pos["x"] + pos["width"] / 2
        center_y = pos["y"] + pos["height"] / 2
        ax.text(center_x, center_y, source_node.data.name, 
               ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
               fontsize=12, color='white')
        
        plt.tight_layout()
        plt.show() 