import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

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

from yolo.figma.figma import (  # noqa: E402 (import after stubs)
    FigmaProcessor,
    FigmaProcessorConfig,
    decode_base64_image,
)


def create_sample_tree(image_data: str) -> dict:
    """Generate a minimal Figma tree structure for testing."""
    def node(
        node_id: str,
        x: float,
        y: float,
        width: float = 10.0,
        height: float = 10.0,
        visible: bool = True,
        children=None,
    ) -> dict:
        children = children or []
        base_box = {"x": x, "y": y, "width": width, "height": height}
        return {
            "data": {
                "id": node_id,
                "name": node_id,
                "visible": visible,
                "absolutePosition": base_box.copy(),
                "absoluteRenderPosition": base_box.copy(),
                "relativePosition": base_box.copy(),
                "relativeRenderPosition": base_box.copy(),
            },
            "children": children,
        }

    root = node(
        "frame-root",
        x=0,
        y=0,
        width=100,
        height=100,
        children=[
            node("child-alpha", x=15, y=20),
            node("child-beta", x=5, y=10),
        ],
    )
    root["data"]["image"] = image_data
    root["data"]["type"] = "FRAME"

    return {"tree": [root]}


def get_sample_png_base64() -> str:
    """Return a 2x2 PNG encoded as a data URL."""
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        img.save(tmp.name, format="PNG")
        tmp.seek(0)
        encoded = tmp.read()
    import base64

    data_url = "data:image/png;base64," + base64.b64encode(encoded).decode("utf-8")
    return data_url


class TestFigmaProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_image = get_sample_png_base64()
        self.sample_data = create_sample_tree(self.sample_image)

    def test_load_document_from_dict_preloads_image(self):
        processor = FigmaProcessor()
        document = processor.load_document(self.sample_data)

        self.assertEqual(len(document.get_all_frames()), 1)
        frame = document.get_frame(0)
        self.assertIsNotNone(frame.img, "Image should be decoded when preload_images=True")
        self.assertEqual(frame.img.size, (2, 2))

    def test_load_document_from_path_without_preload(self):
        processor = FigmaProcessor(FigmaProcessorConfig(preload_images=False))

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(self.sample_data, tmp)
            tmp_path = tmp.name

        try:
            document = processor.load_document(tmp_path)
        finally:
            import os

            os.unlink(tmp_path)

        frame = document.get_frame(0)
        self.assertIsNone(
            frame.img,
            "Image decoding should be skipped when preload_images=False",
        )

    def test_decode_image_handles_invalid_input(self):
        self.assertIsNone(decode_base64_image(None))
        self.assertIsNone(decode_base64_image(""))
        self.assertIsNone(
            decode_base64_image("data:image/png;base64,###INVALID###"),
            "Invalid base64 input should result in None",
        )


if __name__ == "__main__":
    unittest.main()
