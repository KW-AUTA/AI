"""
Tree utilities for Figma elements.

이 모듈은 Figma 요소 트리를 다루기 위한 클래스 기반 유틸리티를 제공합니다.
기존 함수형 구현을 TreeManager 클래스로 대체하여 일관된 API를 제공합니다.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..core.models import FigmaElement


class TreeNode:
    """
    트리의 기본 노드 구현체
    """

    def __init__(
        self,
        data: FigmaElement,
        children: Optional[List['TreeNode']] = None,
        parent: Optional['TreeNode'] = None
    ):
        self.data = data
        self.children: List['TreeNode'] = children if children is not None else []
        self.parent: Optional['TreeNode'] = parent
        for child in self.children:
            child.parent = self

    def add_child(self, child: 'TreeNode'):
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: 'TreeNode'):
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_depth(self) -> int:
        depth = 0
        node = self
        while node.parent:
            node = node.parent
            depth += 1
        return depth

    def get_ancestors(self) -> List['TreeNode']:
        ancestors = []
        node = self.parent
        while node:
            ancestors.append(node)
            node = node.parent
        return ancestors

    def get_descendants(self) -> List['TreeNode']:
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

    def traverse(self, visit: Callable[['TreeNode'], None]):
        visit(self)
        for child in self.children:
            child.traverse(visit)

    def find(self, predicate: Callable[['TreeNode'], bool]) -> Optional['TreeNode']:
        if predicate(self):
            return self
        for child in self.children:
            result = child.find(predicate)
            if result:
                return result
        return None

    def __repr__(self):
        return f"TreeNode(data={self.data}, children={len(self.children)})"


class Tree:
    """
    트리 전체를 관리하는 구현체
    """

    def __init__(self, root: TreeNode):
        self.root = root

    def traverse(self, visit: Callable[[TreeNode], None]):
        self.root.traverse(visit)

    def find(self, predicate: Callable[[TreeNode], bool]) -> Optional[TreeNode]:
        return self.root.find(predicate)

    def __repr__(self):
        return f"Tree(root={self.root})"


@dataclass(frozen=True)
class TreeManagerConfig:
    """
    TreeManager 동작 설정

    Attributes:
        include_invisible: visible 플래그가 False인 노드 포함 여부
        max_depth: 특정 깊이까지만 트리를 구성 (None이면 무제한)
    """

    include_invisible: bool = True
    max_depth: Optional[int] = None


class TreeManager:
    """
    Figma 트리를 로드하고 조작하기 위한 고수준 관리자
    """

    def __init__(self, root: TreeNode, config: Optional[TreeManagerConfig] = None):
        self._tree = Tree(root)
        self.config = config or TreeManagerConfig()

    @property
    def root(self) -> TreeNode:
        """Tree의 루트 노드 반환"""
        return self._tree.root

    def traverse(self, visit: Callable[[TreeNode], None]) -> None:
        """트리를 순회하며 visit 콜백 실행"""
        self._tree.traverse(visit)

    def iter_nodes(self) -> Iterable[TreeNode]:
        """트리의 모든 노드를 순회(iterable)"""
        stack: List[TreeNode] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def find_by_id(self, node_id: str) -> Optional[TreeNode]:
        """노드 ID로 TreeNode 검색"""
        return self._tree.find(lambda node: getattr(node.data, "id", None) == node_id)

    def get_min_render_x(self) -> float:
        """
        트리에서 렌더 좌표의 최소 x 값을 계산

        Returns:
            최소 x 좌표. 좌표가 없으면 0.0 반환
        """
        min_x = float("inf")

        def visit(node: TreeNode) -> None:
            nonlocal min_x
            position = getattr(node.data, "absolute_render_position", None) or {}
            if not position:
                position = getattr(node.data, "absolute_position", None) or {}
            x_value = position.get("x")
            if x_value is not None:
                min_x = min(min_x, float(x_value))

        self.traverse(visit)
        return 0.0 if math.isinf(min_x) else min_x

    @classmethod
    def from_figma_tree(
        cls,
        figma_tree: Dict[str, Any],
        config: Optional[TreeManagerConfig] = None
    ) -> 'TreeManager':
        """
        raw Figma tree(dict)로부터 TreeManager 생성
        """
        config = config or TreeManagerConfig()
        root_node = cls._build_tree(figma_tree, config=config, depth=0, parent=None)
        return cls(root_node, config=config)

    @classmethod
    def from_tree_node(
        cls,
        root: TreeNode,
        config: Optional[TreeManagerConfig] = None
    ) -> 'TreeManager':
        """기존 TreeNode를 기반으로 TreeManager 생성"""
        return cls(root, config=config)

    @classmethod
    def _build_tree(
        cls,
        node: Dict[str, Any],
        config: TreeManagerConfig,
        depth: int,
        parent: Optional[TreeNode]
    ) -> TreeNode:
        """
        재귀적으로 dict 구조를 TreeNode로 변환
        """
        data = cls._extract_figma_element(node)
        current_node = TreeNode(data=data, parent=parent)

        if config.max_depth is not None and depth >= config.max_depth:
            return current_node

        children = node.get("children", []) or []
        for child in children:
            if not config.include_invisible and not cls._is_visible(child):
                continue
            child_node = cls._build_tree(
                child,
                config=config,
                depth=depth + 1,
                parent=current_node
            )
            current_node.add_child(child_node)

        return current_node

    @staticmethod
    def _extract_figma_element(node: Dict[str, Any]) -> FigmaElement:
        """
        dict 데이터를 FigmaElement dataclass로 변환
        """
        node_data = node.get("data", {})

        return FigmaElement(
            id=node_data.get("id", ""),
            name=node_data.get("name", ""),
            absolute_position=node_data.get("absolutePosition", {}),
            relative_position=node_data.get("relativePosition", {}),
            absolute_render_position=node_data.get("absoluteRenderPosition", {}),
            relative_render_position=node_data.get("relativeRenderPosition", {}),
        )

    @staticmethod
    def _is_visible(node: Dict[str, Any]) -> bool:
        """
        노드가 화면에 표시되는지 여부 판단
        """
        node_data = node.get("data", {})
        return node_data.get("visible", True)


__all__ = [
    "TreeNode",
    "Tree",
    "TreeManager",
    "TreeManagerConfig",
]
