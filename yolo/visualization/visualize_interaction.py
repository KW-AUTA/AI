import json
import argparse
import sys
from PIL import Image
import matplotlib.pyplot as plt
from .image_utils import decode_image_from_reference

class FigmaVisualizer:
	"""
	A class to visualize Figma data from a JSON file.
	It can display original frames, frames with bounding boxes, and interactions.
	"""

	def __init__(self, tree, interactions):
		"""
		Initializes the visualizer by loading data from the given JSON file.
		"""
		self.tree = tree
		self.interactions = interactions

	@staticmethod
	def _get_image(img_ref):
		"""
		Decodes a base64 string or downloads an image from a URL into a PIL Image.

		Note: This method is kept for backward compatibility.
		      For new code, consider using image_utils.decode_image_from_reference() directly.
		"""
		return decode_image_from_reference(img_ref)

	def _find_node_by_id(self, node_id, nodes=None):
		"""
		Recursively finds a node in the tree by its ID.
		"""
		if nodes is None:
			nodes = self.tree

		for node in nodes:
			if node.get("data", {}).get("id") == node_id:
				return node
			if "children" in node:
				found_child = self._find_node_by_id(node_id, node["children"])
				if found_child:
					return found_child
		return None

	def _find_root_for_node(self, node_id):
		"""
		Finds the root node (top-level frame/instance) for a given node ID by
		searching from the top-level nodes.
		"""
		for root_node in self.tree:
			# Check if the node_id is the root_node itself
			if root_node.get("data", {}).get("id") == node_id:
				return root_node
			# Check if the node_id is a descendant of the root_node
			if self._find_node_by_id(node_id, root_node.get("children", [])):
				return root_node
		return None

	@staticmethod
	def _extract_boxes_from_node(node):
		"""
		Recursively extracts absolute bounding box information from a node and its children.
		"""
		boxes = []
		if "absolutePosition" in node.get("data", {}):
			pos = node["data"]["absolutePosition"]
			boxes.append({
				"x": pos.get("x", 0),
				"y": pos.get("y", 0),
				"width": pos.get("width", 0),
				"height": pos.get("height", 0)
			})
		
		if "children" in node:
			for child in node["children"]:
				boxes.extend(FigmaVisualizer._extract_boxes_from_node(child))
		return boxes

	def visualize_original(self):
		"""
		Visualizes the top-level frames/instances from the Figma tree.
		"""
		if not self.tree:
			print("No 'tree' data found for visualization.")
			return

		num_frames = len(self.tree)
		if num_frames == 0:
			print("No top-level nodes to visualize.")
			return

		fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))
		if num_frames == 1:
			axs = [axs]

		for i, node in enumerate(self.tree):
			node_data = node.get("data", {})
			img = self._get_image(node_data.get("image"))
			
			ax = axs[i]
			if img:
				ax.imshow(img)
				ax.set_title(f"Node: {node_data.get('name', 'N/A')} (Level {node.get('level', 'N/A')})")
			else:
				ax.set_title(f"Node: {node_data.get('name', 'N/A')} (No Image)")
			ax.axis('off')

		plt.tight_layout()
		plt.show()

	def visualize_with_box(self, node_index=0):
		"""
		Visualizes a single frame with bounding boxes for all its descendant nodes.
		"""
		if not self.tree or node_index >= len(self.tree):
			print("No valid node found at the specified index.")
			return

		root_node = self.tree[node_index]
		root_data = root_node.get("data", {})
		img = self._get_image(root_data.get("image"))

		if not img:
			print(f"No image found for the root node: {root_data.get('name', 'N/A')}")
			return

		all_boxes = self._extract_boxes_from_node(root_node)

		fig, ax = plt.subplots(1, 1, figsize=(10, 10))
		ax.imshow(img)
		ax.set_title(f"Visualization of {root_data.get('name', 'N/A')}")
		ax.axis('off')

		root_render_pos = root_data.get("absoluteRenderPosition", {})
		root_render_x = root_render_pos.get("x", 0)
		root_render_y = root_render_pos.get("y", 0)

		for bbox in all_boxes:
			adjusted_x = bbox["x"] - root_render_x
			adjusted_y = bbox["y"] - root_render_y
			ax.add_patch(plt.Rectangle(
				(adjusted_x, adjusted_y), bbox["width"], bbox["height"], 
				fill=False, edgecolor='red', linewidth=2
			))
		
		plt.show()

	def visualize_interaction(self, interaction_index):
		"""
		Visualizes a specific interaction, showing source and destination nodes.
		"""
		if not self.interactions:
			print("No 'interactions' data found.")
			return

		if not (0 <= interaction_index < len(self.interactions)):
			print(f"Interaction index {interaction_index} is out of bounds. Total: {len(self.interactions)}")
			return

		interaction = self.interactions[interaction_index]
		interaction_type_info = interaction.get("interactionType", {})
		source_id = interaction_type_info.get("sourceId")

		if not source_id:
			print(f"No sourceId found for interaction {interaction_index}.")
			return
			
		source_node = self._find_node_by_id(source_id)
		if not source_node:
			print(f"Source node with ID {source_id} not found.")
			return
		source_top_data = self._find_root_for_node(source_id)

		if interaction.get("type") == "NODE":
			self._visualize_node_interaction(interaction, source_top_data, interaction_index)
		elif interaction.get("type") == "URL":
			self._visualize_url_interaction(interaction, source_top_data, interaction_index)
		else:
			print(f"Unsupported interaction type: {interaction.get('type')}")

	def _visualize_node_interaction(self, interaction, source_node, index):
		"""Helper to visualize NODE type interactions."""
		interaction_type_info = interaction.get("interactionType", {})
		destination_id = interaction_type_info.get("destinationId")
		
		if not destination_id:
			print(f"No destinationId found for interaction {index}.")
			return

		destination_node = self._find_node_by_id(destination_id)
		if not destination_node:
			print(f"Destination node with ID {destination_id} not found.")
			return

		source_data = source_node.get("data", {})
		source_top_data = self._find_root_for_node(source_data.get("id"))
		dest_data = destination_node.get("data", {})
		
		# The base image is the frame containing the interaction's source node
		source_frame = self._find_root_for_node(source_data.get("id"))
		if not source_frame:
			print(f"Could not find the root frame for the source node: {source_data.get('name', 'N/A')}")
			return
		
		source_frame_img = self._get_image(source_frame.get("data", {}).get("image"))

		if not source_frame_img:
			print(f"No image for source frame of node: {source_data.get('name', 'N/A')}")
			return
		
		fig, ax = plt.subplots(1, 1, figsize=(10, 10))
		ax.imshow(source_frame_img)
		ax.set_title(f"Interaction {index}: {source_data.get('name')} -> {dest_data.get('name')}")
		# ax.axis('off')

		# Draw bounding box for the trigger node on the source frame
		self._draw_bounding_box(ax, source_data, edgecolor='red', linewidth=3)

		navigation = interaction_type_info.get("navigation")
		if navigation == "OVERLAY":
			self._handle_overlay_interaction(ax, source_frame_img, destination_node, interaction_type_info)
		else: 
			# For NAVIGATE, SWAP, etc., a full visualization would show the destination frame.
			# Here, we just mark the trigger on the source frame.
			# The destination box is not drawn as it belongs to a different frame.
			pass
		
		plt.show()

	def _visualize_url_interaction(self, interaction, source_node, index):
		"""Helper to visualize URL type interactions."""
		source_data = source_node.get("data", {})
		source_frame = self._find_root_for_node(source_data.get("id"))
		if not source_frame:
			 source_frame = self.tree[0] # Fallback

		source_frame_img = self._get_image(source_frame.get("data", {}).get("image"))

		if not source_frame_img:
			print(f"No image for source frame of node: {source_data.get('name', 'N/A')}")
			return

		fig, ax = plt.subplots(1, 1, figsize=(10, 10))
		ax.imshow(source_frame_img)
		url = interaction.get("interactionType", {}).get('url', 'N/A')
		ax.set_title(f"Interaction {index}: {source_data.get('name')} -> URL")
		ax.text(0.5, 0.5, f"Opens URL: {url}", transform=ax.transAxes, fontsize=12, 
				color='blue', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
		
		self._draw_bounding_box(ax, source_data, edgecolor='red', linewidth=3)
		plt.show()

	def _handle_overlay_interaction(self, ax, source_img, dest_node, interaction_info):
		"""Handles the logic for displaying an OVERLAY interaction."""
		dest_data = dest_node.get("data", {})
		dest_img = self._get_image(dest_data.get("image"))

		if not dest_img:
			print(f"No image for destination overlay node: {dest_data.get('name', 'N/A')}")
			self._draw_bounding_box(ax, dest_node, edgecolor='blue', linewidth=3)
			return

		overlay_info = interaction_info.get("overlay", {})
		pos_type = overlay_info.get("positionType", "MANUAL")
		
		overlay_x, overlay_y, margin_left, margin_top = self._calculate_overlay_position(
			pos_type, overlay_info.get("position"), source_img.size, dest_node
		)
		# Composite the overlay onto the source image
		if source_img.mode != 'RGBA':
			source_img = source_img.convert('RGBA')
		
		combined_img = source_img.copy()
		combined_img.paste(dest_img, (int(overlay_x), int(overlay_y)), dest_img.convert('RGBA'))
		bbox_x, bbox_y = overlay_x - margin_left, overlay_y - margin_top
		ax.imshow(combined_img)
		# Draw bounding box for the destination node (the overlay)
		dest_pos = dest_data.get("absolutePosition")
		if dest_pos:
			ax.add_patch(plt.Rectangle(
				(bbox_x, bbox_y), dest_pos["width"], dest_pos["height"],
				fill=False, edgecolor='blue', linewidth=3
			))

	@staticmethod
	def _calculate_overlay_position(pos_type, manual_pos, source_size, dest_node):
		"""Calculates the top-left (x, y) position for an overlay."""
		source_w, source_h = source_size
		dest_w, dest_h = dest_node.get("data", {}).get("absolutePosition", {}).get("width"), dest_node.get("data", {}).get("absolutePosition", {}).get("height")
		dest_x, dest_y = dest_node.get("data", {}).get("absolutePosition", {}).get("x"), dest_node.get("data", {}).get("absolutePosition", {}).get("y")
		render_x, render_y = dest_node.get("data", {}).get("absoluteRenderPosition", {}).get("x"), dest_node.get("data", {}).get("absoluteRenderPosition", {}).get("y")

		margin_left = render_x - dest_x
		margin_top = render_y - dest_y

		if pos_type == "CENTER":
			return (source_w - dest_w) / 2 + margin_left, (source_h - dest_h) / 2 + margin_top, margin_left, margin_top
		if pos_type == "TOP_LEFT":
			return margin_left, margin_top, margin_left, margin_top
		if pos_type == "TOP_CENTER":
			return (source_w - dest_w) / 2 + margin_left, margin_top, margin_left, margin_top
		if pos_type == "TOP_RIGHT":
			return source_w - dest_w + margin_left, margin_top, margin_left, margin_top
		if pos_type == "BOTTOM_LEFT":
			return margin_left, source_h - dest_h + margin_top, margin_left, margin_top
		if pos_type == "BOTTOM_CENTER":
			return (source_w - dest_w) / 2 + margin_left, source_h - dest_h + margin_top, margin_left, margin_top
		if pos_type == "BOTTOM_RIGHT":
			return source_w - dest_w + margin_left, source_h - dest_h + margin_top, margin_left, margin_top
		if pos_type == "MANUAL" and manual_pos:
			return manual_pos.get("x", 0) + margin_left, manual_pos.get("y", 0) + margin_top, margin_left, margin_top
		
		return 0, 0, 0, 0 # Default fallback

	@staticmethod
	def _draw_bounding_box(ax, node_data, **kwargs):
		"""Draws a bounding box for a node relative to its containing frame."""
		pos = node_data.get("absolutePosition")
		render_pos = node_data.get("absoluteRenderPosition")
		if not pos or not render_pos:
			print(f"Warning: Missing position data for node {node_data.get('name')}", file=sys.stderr)
			return

		box_x = pos["x"] - render_pos["x"]
		box_y = pos["y"] - render_pos["y"]

		ax.add_patch(plt.Rectangle(
			(box_x, box_y), pos["width"], pos["height"],
			fill=False, **kwargs
		))

def main():
	"""
	Command-line interface for the Figma Visualizer.
	"""
	parser = argparse.ArgumentParser(description="Visualize Figma data from a JSON file.")
	parser.add_argument("--json", type=str, required=True, help="Path to the JSON file from the Figma plugin.")
	parser.add_argument("--visualize_type", type=str, default="box", choices=["original", "box", "interaction"], help="Type of visualization.")
	parser.add_argument("--interaction_index", type=int, help="Index of the interaction to visualize (required for 'interaction' type).")
	args = parser.parse_args()

	visualizer = FigmaVisualizer(args.json)

	if args.visualize_type == "original":
		visualizer.visualize_original()
	elif args.visualize_type == "box":
		visualizer.visualize_with_box()
	elif args.visualize_type == "interaction":
		if args.interaction_index is None:
			print("Error: --interaction_index is required for 'interaction' visualization.", file=sys.stderr)
			sys.exit(1)
		visualizer.visualize_interaction(args.interaction_index)

if __name__ == "__main__":
	main()
