import importlib.util
import unittest
from pathlib import Path

_stub_spec = importlib.util.spec_from_file_location(
    "test_dependency_stubs",
    Path(__file__).resolve().parent / "stub_dependencies.py",
)
_stub_module = importlib.util.module_from_spec(_stub_spec)
assert _stub_spec and _stub_spec.loader
_stub_spec.loader.exec_module(_stub_module)
_stub_module.install_test_stubs()

import sys
import types

_yolo_base = Path(__file__).resolve().parents[1]
if "yolo" not in sys.modules:
    yolo_pkg = types.ModuleType("yolo")
    yolo_pkg.__path__ = [str(_yolo_base)]
    sys.modules["yolo"] = yolo_pkg

from yolo.utils.tree_loader import TreeManager, TreeManagerConfig  # noqa: E402


def build_raw_tree() -> dict:
    """Create a simplified Figma tree dictionary for TreeManager tests."""

    def node(node_id: str, x: float, y: float, children=None) -> dict:
        children = children or []
        base_box = {"x": x, "y": y, "width": 20, "height": 10}
        return {
            "data": {
                "id": node_id,
                "name": node_id,
                "absolutePosition": base_box.copy(),
                "relativePosition": base_box.copy(),
                "absoluteRenderPosition": base_box.copy(),
                "relativeRenderPosition": base_box.copy(),
                "visible": True,
            },
            "children": children,
        }

    return node(
        "root",
        x=10,
        y=5,
        children=[
            node("child-a", x=6, y=12),
            node("child-b", x=2, y=18, children=[node("grandchild-b1", x=-4, y=22)]),
        ],
    )


class TestTreeManager(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_tree = build_raw_tree()
        self.manager = TreeManager.from_figma_tree(self.raw_tree, TreeManagerConfig())

    def test_iter_nodes_traverses_entire_tree(self):
        nodes = list(self.manager.iter_nodes())
        node_ids = [node.data.id for node in nodes]

        self.assertEqual(len(nodes), 4)
        self.assertListEqual(
            sorted(node_ids),
            ["child-a", "child-b", "grandchild-b1", "root"],
        )

    def test_find_by_id_returns_expected_node(self):
        target = self.manager.find_by_id("grandchild-b1")
        self.assertIsNotNone(target)
        self.assertEqual(target.data.id, "grandchild-b1")

    def test_min_render_x_uses_absolute_render_position(self):
        min_x = self.manager.get_min_render_x()
        self.assertEqual(min_x, -4.0)


if __name__ == "__main__":
    unittest.main()
