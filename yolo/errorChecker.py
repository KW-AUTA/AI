from typing import List
from .models import MatchResult
from .error_list import *
class ErrorChecker:
	
	def _coordinate_error(self, figma_box: List[int], web_box: List[int]) -> bool:
		diff_x = abs(figma_box[0] - web_box[0])
		diff_y = abs(figma_box[1] - web_box[1])

		if diff_x < 10 and diff_y < 10:
			return True
		return False
	
	def _size_error(self, figma_box: List[int], web_box: List[int]) -> bool:
		diff_w = abs(figma_box[2] - web_box[2])
		diff_h = abs(figma_box[3] - web_box[3])

		if diff_w < 10 and diff_h < 10:
			return True
		return False
	
	def _text_error(self, figma_text: str, web_text: str) -> bool:

		if figma_text == web_text:
			return True
		return False
	
	def check_error(self, figma_box: List[int], web_box: List[int], figma_text: str, web_text: str) -> List[str]:
		errors = []
		if not self._coordinate_error(figma_box, web_box):
			errors.append(G_ERROR_COORDINATE_X)
		if not self._size_error(figma_box, web_box):
			errors.append(G_ERROR_SIZE)
		if not self._text_error(figma_text, web_text):
			errors.append(G_ERROR_TEXT)
		return errors
	
	def check_error_by_match(self, match: MatchResult) -> List[str]:
		errors = []
		errors.extend(self.check_error(match.figma.extracted.box, match.web.box, match.figma.extracted.text, match.web.text))
		if len(errors) == 0:
			return [NORMAL]
		match.errorCategories = errors
		return match