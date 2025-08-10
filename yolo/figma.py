from __future__ import annotations
from PIL import Image
import matplotlib.pyplot as plt
import json
import argparse
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import io
from dataclasses import dataclass
import numpy as np

def decode_base64_image(image_data):
	"""
	Decodes a base64 string or downloads an image from a URL into a PIL Image.
	"""
	if not image_data or not isinstance(image_data, str):
		return None


	if image_data.startswith("data:image"):
		img_str = image_data.split(",", 1)[1]
	else:
		img_str = image_data

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

@dataclass
class FigmaBox:
	x: float
	y: float
	width: float
	height: float

@dataclass
class FigmaBoxPair:
	absolute_box: FigmaBox
	render_box: FigmaBox

@dataclass	
class FigmaElementTree:
	depth: int
	id: str
	boxes: list[FigmaBoxPair]
	children: list[FigmaElementTree]

class FigmaFrame:
	def __init__(self, data):
		self.raw_node_data = data
		self.img = decode_base64_image(data["data"]["image"])
		self.element_tree = FigmaElementTree(
			depth=0,
			id="",
			boxes=[],
			children=[]
		)
		self.min_x = float('inf')
		self.min_y = float('inf')
		self.element_tree = self.build_element_tree()
		self.adjust_start_point()
		
	def adjust_start_point(self):
		self._adjust_start_point_recursive(self.element_tree)
	
	def _adjust_start_point_recursive(self, tree):
		if tree.depth != 0:
			for box_pair in tree.boxes:
				box_pair.absolute_box.x -= self.min_x
				box_pair.absolute_box.y -= self.min_y
				box_pair.render_box.x -= self.min_x
				box_pair.render_box.y -= self.min_y
		for child in tree.children:
			self._adjust_start_point_recursive(child)

	def _build_element_tree_recursive(self, node, depth):
		node_data = node["data"]

		# 현재 노드의 박스 정보 생성
		absolute_box = FigmaBox(
			x=node_data["absolutePosition"]["x"],
			y=node_data["absolutePosition"]["y"],
			width=node_data["absolutePosition"]["width"],
			height=node_data["absolutePosition"]["height"]
		)
		render_box = FigmaBox(
			x=node_data["absoluteRenderPosition"]["x"],
			y=node_data["absoluteRenderPosition"]["y"],
			width=node_data["absoluteRenderPosition"]["width"],
			height=node_data["absoluteRenderPosition"]["height"]
		)

		# 현재 노드용 새로운 트리 생성
		current_tree = FigmaElementTree(
			depth=depth,
			id=node_data["id"],
			boxes=[FigmaBoxPair(absolute_box, render_box)],
			children=[]
		)

		# 현재 노드의 좌표를 전체 최소값에 반영
		self.min_x = min(self.min_x, render_box.x)
		self.min_y = min(self.min_y, render_box.y)

		# 자식 노드들 처리
		if "children" in node:
			current_tree.children = [self._build_element_tree_recursive(child, depth + 1) for child in node["children"]]

		return current_tree

	def build_element_tree(self):
		return self._build_element_tree_recursive(self.raw_node_data, 0)
	
	def visualize_raw(self):
		fig, ax = plt.subplots()
		ax.imshow(self.img)
		plt.show()
		
	
class FigmaDataLoader:
	"""JSON 파일에서 Figma 데이터를 로드하는 클래스"""
	@staticmethod
	def load_from_file(json_file_path: str) -> dict:
		with open(json_file_path, "r") as f:
			return json.load(f)
	
	@staticmethod
	def load_from_string(json_string: str) -> dict:
		return json.loads(json_string)

class FigmaDocument:
	"""Figma 문서 전체를 관리하는 클래스"""
	def __init__(self, data: dict):
		self.raw_data = data
		self.frames = [FigmaFrame(tree) for tree in data["tree"]]
	
	def get_frame(self, index: int = 0) -> FigmaFrame:
		return self.frames[index]
	
	def get_all_frames(self) -> list[FigmaFrame]:
		return self.frames
	
	def visualize_all_frames(self):
		for i, frame in enumerate(self.frames):
			print(f"Frame {i}:")
			frame.show_image()


def main():
	parser = argparse.ArgumentParser(description="Visualize Figma data from a JSON file.")
	parser.add_argument("--json", type=str, required=True, help="Path to the JSON file from the Figma plugin.")
	parser.add_argument("--visualize", action="store_true", help="Show visualization with bounding boxes")
	parser.add_argument("--save", action="store_true", help="Save visualization to file")
	parser.add_argument("--output", type=str, default="figma_output.png", help="Output file path")
	args = parser.parse_args()
   
	# 데이터 로딩
	data = FigmaDataLoader.load_from_file(args.json)
	document = FigmaDocument(data)
	first_frame = document.get_frame(0)
	print(first_frame.min_x)
	if args.visualize:
		import sys
		import os
		sys.path.append(os.path.dirname(__file__))
		from figma_visualizer import FigmaVisualizer
		
		visualizer = FigmaVisualizer()
		if args.save:
			visualizer.show_with_boxes(first_frame, save=True, output_path=args.output)
		else:
			visualizer.show_with_boxes(first_frame)
	elif args.save:
		# 저장만 하고 화면에는 표시 안함
		import sys
		import os
		sys.path.append(os.path.dirname(__file__))
		from figma_visualizer import FigmaVisualizer
		visualizer = FigmaVisualizer()
		visualizer.save_with_boxes(first_frame, args.output)


