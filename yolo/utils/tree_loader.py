from typing import Any, List, Optional, Callable
from ..core.models import FigmaElement
class TreeNode:
    """
    트리의 기본 노드 구현체
    """
    def __init__(self, data: FigmaElement, children: Optional[List['TreeNode']] = None, parent: Optional['TreeNode'] = None):
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

