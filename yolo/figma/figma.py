from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Union

import base64
import io
import json
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def decode_base64_image(image_data: Optional[str]) -> Optional[Image.Image]:
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


def get_frame_by_name_from_raw(figma_raw: list, name: str) -> Optional[dict]:
	"""Figma raw 데이터에서 이름으로 프레임 찾기"""
	for data in figma_raw:
		if data.get('data', {}).get('name') == name:
			return data
	return None


def get_img_by_id(id: str, figma_raw: list) -> Optional[Image.Image]:
	"""ID로 Figma 이미지 가져오기"""
	from ..utils.utils import get_min_x

	for figma_element in figma_raw:
		if figma_element.get('data', {}).get('id') == id:
			min_x = get_min_x(figma_element, 0)
			img = decode_base64_image(figma_element['data']['image'])
			if img:
				width = figma_element['data']['absolutePosition']['width']
				height = figma_element['data']['absolutePosition']['height']
				img = img.crop((-min_x, 0, width - min_x, height))
			return img
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


@dataclass(frozen=True)
class FigmaProcessorConfig:
	"""Figma 데이터 처리 설정"""

	auto_adjust_origin: bool = True
	preload_images: bool = True


class FigmaProcessor:
	"""
	Figma JSON 데이터를 로드하고 Frame/Document 객체로 변환하는 관리자
	"""

	def __init__(self, config: Optional[FigmaProcessorConfig] = None):
		self.config = config or FigmaProcessorConfig()

	def decode_image(self, image_data: Optional[str]) -> Optional[Image.Image]:
		"""이미지 데이터 디코딩"""
		return decode_base64_image(image_data)

	def load_json(self, source: Union[str, Path]) -> dict:
		"""JSON 파일 경로 또는 URL에서 데이터 로드"""
		# URL인 경우
		if isinstance(source, str) and source.startswith(('http://', 'https://')):
			return FigmaDataLoader.load_from_url(source)
		# 파일 경로인 경우
		return FigmaDataLoader.load_from_file(str(source))

	def load_document(self, source: Union[str, Path, dict]) -> FigmaDocument:
		"""
		Figma JSON 데이터로부터 FigmaDocument 생성

		Args:
			source: JSON 경로 또는 이미 로드된 dict
		"""
		if isinstance(source, (str, Path)):
			data = self.load_json(source)
		elif isinstance(source, dict):
			data = source
		else:
			raise TypeError(f"Unsupported Figma source type: {type(source)}")

		return FigmaDocument(
			data,
			frame_factory=lambda tree: self._create_frame(tree)
		)

	def _create_frame(self, tree: dict) -> FigmaFrame:
		"""Frame 인스턴스 생성"""
		image_loader = self.decode_image if self.config.preload_images else lambda _: None
		return FigmaFrame(
			data=tree,
			image_loader=image_loader,
			auto_adjust=self.config.auto_adjust_origin
		)

	@classmethod
	def create(cls, config: Optional[FigmaProcessorConfig] = None) -> 'FigmaProcessor':
		"""팩토리 함수"""
		return cls(config=config)

class FigmaFrame:
	"""Figma 프레임 데이터를 관리하는 클래스 - 이미지, 트리, 메타데이터 포함"""

	def __init__(
		self,
		data: dict,
		image_loader: Optional[Callable[[Optional[str]], Optional[Image.Image]]] = None,
		auto_adjust: bool = True,
	):
		self.raw_node_data = data
		self._image_loader = image_loader or decode_base64_image

		# 메타데이터 추출
		node_data = data.get("data", {})
		self.id = node_data.get("id", "")
		self.name = node_data.get("name", "")
		self.width = node_data.get("absolutePosition", {}).get("width", 0)
		self.height = node_data.get("absolutePosition", {}).get("height", 0)

		# 이미지 로드 및 RGB 변환
		image_data = node_data.get("image")
		self.img = self._load_and_convert_image(image_data)

		# 트리 구조 초기화
		self.element_tree = FigmaElementTree(
			depth=0,
			id="",
			boxes=[],
			children=[]
		)
		self.min_x = float('inf')
		self.min_y = float('inf')
		self.element_tree = self.build_element_tree()

		if auto_adjust:
			self.adjust_start_point()

	def _load_and_convert_image(self, image_data: Optional[str]) -> Optional[Image.Image]:
		"""이미지 로드 및 RGB 변환"""
		img = self._image_loader(image_data)
		if img is None:
			return None

		# RGBA -> RGB 변환 (YOLO는 3채널만 지원)
		if img.mode == 'RGBA':
			background = Image.new('RGB', img.size, (255, 255, 255))
			background.paste(img, mask=img.split()[3])  # alpha channel as mask
			return background
		elif img.mode != 'RGB':
			return img.convert('RGB')

		return img

	@property
	def size(self) -> tuple[int, int]:
		"""프레임 크기 (width, height)"""
		return (self.width, self.height)

	@property
	def has_image(self) -> bool:
		"""이미지가 로드되었는지 확인"""
		return self.img is not None
		
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
		node_data = node.get("data", {})

		# 현재 노드의 박스 정보 생성
		absolute_box = FigmaBox(
			x=node_data.get("absolutePosition", {}).get("x", 0.0),
			y=node_data.get("absolutePosition", {}).get("y", 0.0),
			width=node_data.get("absolutePosition", {}).get("width", 0.0),
			height=node_data.get("absolutePosition", {}).get("height", 0.0)
		)
		render_box = FigmaBox(
			x=node_data.get("absoluteRenderPosition", {}).get("x", 0.0),
			y=node_data.get("absoluteRenderPosition", {}).get("y", 0.0),
			width=node_data.get("absoluteRenderPosition", {}).get("width", 0.0),
			height=node_data.get("absoluteRenderPosition", {}).get("height", 0.0)
		)

		# 현재 노드용 새로운 트리 생성
		current_tree = FigmaElementTree(
			depth=depth,
			id=node_data.get("id", ""),
			boxes=[FigmaBoxPair(absolute_box, render_box)],
			children=[]
		)

		# 현재 노드의 좌표를 전체 최소값에 반영
		self.min_x = min(self.min_x, render_box.x)
		self.min_y = min(self.min_y, render_box.y)

		# 자식 노드들 처리
		if node.get("children"):
			current_tree.children = [
				self._build_element_tree_recursive(child, depth + 1)
				for child in node.get("children", [])
			]

		return current_tree

	def build_element_tree(self):
		return self._build_element_tree_recursive(self.raw_node_data, 0)
	
	def visualize_raw(self):
		fig, ax = plt.subplots()
		ax.imshow(self.img)
		plt.show()
		
	
class FigmaDataLoader:
	"""JSON 파일 또는 URL에서 Figma 데이터를 로드하는 클래스"""
	@staticmethod
	def load_from_file(json_file_path: str) -> dict:
		with open(json_file_path, "r") as f:
			return json.load(f)

	@staticmethod
	def load_from_string(json_string: str) -> dict:
		return json.loads(json_string)

	@staticmethod
	def load_from_url(url: str) -> dict:
		"""URL에서 JSON 데이터를 가져옴"""
		import requests
		response = requests.get(url)
		response.raise_for_status()
		return response.json()

class FigmaDocument:
	"""Figma 문서 전체를 관리하는 클래스 - 프레임, 인터랙션, 메타데이터 포함"""

	def __init__(
		self,
		data: dict,
		frame_factory: Optional[Callable[[dict], FigmaFrame]] = None
	):
		self.raw_data = data
		self._frame_factory = frame_factory or FigmaFrame

		# 프레임 로드
		self.frames = [
			self._frame_factory(tree)
			for tree in data.get("tree", [])
		]

		# 인터랙션 정보
		self.interactions = data.get("interactions", [])

		# 프레임 이름 매핑 (빠른 검색용)
		self._frame_by_name = {frame.name: frame for frame in self.frames}
		self._frame_by_id = {frame.id: frame for frame in self.frames}

	def get_frame(self, index: int = 0) -> FigmaFrame:
		"""인덱스로 프레임 가져오기"""
		return self.frames[index]

	def get_frame_by_name(self, name: str) -> Optional[FigmaFrame]:
		"""이름으로 프레임 찾기"""
		return self._frame_by_name.get(name)

	def get_frame_by_id(self, frame_id: str) -> Optional[FigmaFrame]:
		"""ID로 프레임 찾기"""
		return self._frame_by_id.get(frame_id)

	def get_all_frames(self) -> list[FigmaFrame]:
		"""모든 프레임 반환"""
		return self.frames

	def get_interactions_for_frame(self, frame_name: str) -> list[dict]:
		"""특정 프레임의 인터랙션만 필터링"""
		frame = self.get_frame_by_name(frame_name)
		if not frame:
			return []

		# 해당 프레임에 속한 인터랙션만 반환
		return [
			interaction for interaction in self.interactions
			# TODO: 프레임별 필터링 로직 추가 필요
		]

	@property
	def frame_names(self) -> list[str]:
		"""모든 프레임 이름 목록"""
		return [frame.name for frame in self.frames]

	@property
	def has_interactions(self) -> bool:
		"""인터랙션이 있는지 확인"""
		return len(self.interactions) > 0

	def visualize_all_frames(self):
		"""모든 프레임 시각화"""
		for i, frame in enumerate(self.frames):
			print(f"Frame {i} ({frame.name}):")
			frame.visualize_raw()


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
