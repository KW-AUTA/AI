"""
웹 네비게이션 모듈

Selenium을 사용하여 웹 페이지 탐색, 스크린샷 캡처, 요소 상호작용을 제공합니다.
"""

import time
import io
import base64
from PIL import Image
from typing import Optional, Tuple
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@dataclass(frozen=True)
class WebNavigatorConfig:
	"""웹 네비게이터 설정 클래스"""
	headless: bool = True
	base_url: str = "https://www.kw.ac.kr/ko/index.jsp"
	page_load_timeout: int = 30
	element_wait_timeout: int = 10
	scroll_wait_time: float = 1.0
	page_settle_time: float = 1.0
	window_size_buffer: int = 100


class WebNavigator:
	"""
	웹 네비게이션을 담당하는 클래스

	Selenium을 사용하여 웹 페이지 탐색, 스크린샷 캡처, 요소 상호작용을 제공합니다.
	컨텍스트 매니저를 지원하여 자동으로 리소스를 정리합니다.
	"""

	def __init__(self, config: Optional[WebNavigatorConfig] = None):
		"""
		Args:
			config: 웹 네비게이터 설정. None이면 기본 설정 사용
		"""
		self.config = config or WebNavigatorConfig()
		self.opts = Options()

		if self.config.headless:
			self.opts.add_argument('--headless')

		service = ChromeService(ChromeDriverManager().install())
		self.driver = webdriver.Chrome(service=service, options=self.opts)

	# ============================================================================
	# Context Manager Support
	# ============================================================================

	def __enter__(self):
		"""컨텍스트 매니저 진입"""
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""컨텍스트 매니저 종료 - 자동으로 브라우저 정리"""
		self.quit()
		return False

	# ============================================================================
	# Navigation Methods
	# ============================================================================

	def navigate(self, url: str) -> None:
		"""
		웹 페이지로 이동

		Args:
			url: 이동할 URL
		"""
		self.driver.get(url)

		# 페이지 로딩 완료까지 대기
		try:
			# DOM 로딩 완료 대기
			WebDriverWait(self.driver, self.config.page_load_timeout).until(
				lambda driver: driver.execute_script("return document.readyState") == "complete"
			)

			# 추가로 body 요소가 로드될 때까지 대기
			WebDriverWait(self.driver, self.config.element_wait_timeout).until(
				EC.presence_of_element_located((By.TAG_NAME, "body"))
			)

			# JavaScript 실행 완료를 위한 짧은 추가 대기
			time.sleep(self.config.page_settle_time)

		except Exception as e:
			print(f"Warning: Page loading timeout or error: {e}")
			# 타임아웃이 발생해도 계속 진행

	# ============================================================================
	# Window & Viewport Methods
	# ============================================================================

	def resize_window(self, width: int, height: int) -> None:
		"""
		브라우저 창 크기 조정

		Args:
			width: 목표 viewport 너비
			height: 목표 viewport 높이
		"""
		# 1. 먼저 브라우저 창을 충분히 크게 설정
		self.driver.set_window_size(
			width + self.config.window_size_buffer,
			height + self.config.window_size_buffer
		)

		# 2. 페이지가 완전히 로드될 때까지 대기
		self.driver.execute_script("return document.readyState") == "complete"

		# 3. 실제 viewport 크기 가져오기
		viewport_width = self.driver.execute_script("return window.innerWidth")
		viewport_height = self.driver.execute_script("return window.innerHeight")

		# 4. 원하는 크기와 viewport 크기의 차이 계산
		width_diff = width - viewport_width
		height_diff = height - viewport_height

		# 5. 브라우저 창 크기 조정 (상태바 등을 고려)
		self.driver.set_window_size(width + width_diff, height + height_diff)

		# 6. 스크롤바가 생길 수 있으므로 viewport 크기 다시 확인
		viewport_width = self.driver.execute_script("return window.innerWidth")
		viewport_height = self.driver.execute_script("return window.innerHeight")

		# 7. 필요한 경우 미세 조정
		if viewport_width != width or viewport_height != height:
			width_diff = width - viewport_width
			height_diff = height - viewport_height
			current_size = self.driver.get_window_size()
			self.driver.set_window_size(
				current_size['width'] + width_diff,
				current_size['height'] + height_diff
			)

	# ============================================================================
	# Scroll Methods
	# ============================================================================

	def scroll_down(self, target_height: int) -> None:
		"""
		지정한 높이로 스크롤 다운

		Args:
			target_height: 스크롤할 목표 높이
		"""
		self.driver.execute_script(f"window.scrollTo(0, {target_height});")
		time.sleep(self.config.scroll_wait_time)

	def scroll_to_top(self) -> None:
		"""맨 위까지 스크롤"""
		self.driver.execute_script("window.scrollTo(0, 0);")
		time.sleep(self.config.scroll_wait_time)

	def scroll_to_bottom(self) -> int:
		"""
		맨 아래까지 스크롤

		Returns:
			최종 스크롤 높이
		"""
		last_height = self.driver.execute_script("return document.body.scrollHeight")

		while True:
			# END 키로 맨 아래로 스크롤
			actions = ActionChains(self.driver)
			actions.send_keys(Keys.END).perform()
			time.sleep(self.config.scroll_wait_time)

			# 새로운 스크롤 높이 계산
			new_height = self.driver.execute_script("return document.body.scrollHeight")

			# 더 이상 스크롤이 안 되면 종료
			if new_height == last_height:
				break

			last_height = new_height

		return last_height

	def scroll_to_y(self, y: int) -> None:
		"""
		지정한 y 위치로 스크롤

		Args:
			y: 스크롤할 y 좌표
		"""
		self.driver.execute_script(f"window.scrollTo(0, {y});")
		time.sleep(self.config.scroll_wait_time)

	# ============================================================================
	# Screenshot Methods
	# ============================================================================

	def capture_full_page(self) -> Image.Image:
		"""
		전체 페이지 스크린샷 캡처 (CDP 명령 사용)

		Returns:
			전체 페이지 스크린샷 이미지
		"""
		# CDP 명령으로 전체 페이지 스크린샷 캡처
		result = self.driver.execute_cdp_cmd(
			"Page.captureScreenshot",
			{
				"fromSurface": True,
				"captureBeyondViewport": True
			}
		)

		# Base64 이미지를 PIL Image로 변환
		return Image.open(io.BytesIO(base64.b64decode(result["data"]))).convert('RGB')

	def capture_full_page_with_scroll(
		self,
		root_img: Image.Image,
		target_height: int
	) -> Image.Image:
		"""
		스크롤을 포함한 전체 페이지 캡처

		Args:
			root_img: 기준 이미지 (너비를 맞추기 위해 사용)
			target_height: 목표 높이

		Returns:
			캡처된 전체 페이지 이미지
		"""
		# 브라우저 창 크기 조정
		self.resize_window(root_img.width, target_height)

		# 맨 아래까지 스크롤 후 맨 위로 복귀
		self.scroll_to_bottom()
		self.scroll_to_top()

		time.sleep(self.config.scroll_wait_time)

		# 전체 페이지 캡처
		web_img = self.capture_full_page()

		# 이미지 크기 조정
		scale = web_img.width / root_img.width
		web_img = web_img.resize((root_img.width, int(web_img.height / scale)))

		return web_img

	# ============================================================================
	# Element Interaction Methods
	# ============================================================================

	def get_element_xpath(self, element) -> str:
		"""
		주어진 WebElement의 절대 XPath를 생성하여 반환

		Args:
			element: XPath를 얻을 WebElement

		Returns:
			절대 XPath (예: /html/body/div[2]/ul/li[3]/a)
		"""
		script = """
		function absoluteXPath(el) {
			if (el.tagName.toLowerCase() == 'html')
				return '/html';
			if (el === document.body)
				return '/html/body';

			var ix = 0;
			var siblings = el.parentNode.childNodes;
			for (var i = 0; i < siblings.length; i++) {
				var sib = siblings[i];
				if (sib === el) {
					return absoluteXPath(el.parentNode) + '/' + el.tagName.toLowerCase() + '[' + (ix+1) + ']';
				}
				if (sib.nodeType === 1 && sib.tagName === el.tagName) {
					ix++;
				}
			}
		}
		return absoluteXPath(arguments[0]);
		"""
		xpath = self.driver.execute_script(script, element)
		return xpath

	def get_element_at_coordinate_and_xpath(
		self,
		page_x: int,
		page_y: int
	) -> Tuple[Optional[object], Optional[str]]:
		"""
		문서 전체 기준 (page_x, page_y) 좌표 위의 요소를 찾아서 반환

		Args:
			page_x: 페이지 x 좌표
			page_y: 페이지 y 좌표

		Returns:
			(WebElement, XPath) 튜플. 요소를 찾지 못하면 (None, None)
		"""
		# 현재 스크롤 위치를 가져와서 client 좌표로 변환
		scroll_x = self.driver.execute_script("return window.scrollX;")
		scroll_y = self.driver.execute_script("return window.scrollY;")
		client_x = page_x - scroll_x
		client_y = page_y - scroll_y

		# 뷰포트(client) 좌표 위의 요소를 DOM에서 검색
		get_el_script = """
		var cx = arguments[0], cy = arguments[1];
		return document.elementFromPoint(cx, cy);
		"""
		element = self.driver.execute_script(get_el_script, client_x, client_y)

		if element is None:
			return None, None

		# element의 XPath 계산
		xpath = self.get_element_xpath(element)
		return element, xpath

	def get_url_in_new_tab(self, xpath: str) -> str:
		"""
		XPath로 지정한 요소를 새 탭에서 열고 URL을 반환

		Args:
			xpath: 새 탭으로 열 요소의 XPath

		Returns:
			새 탭으로 연 페이지의 URL. 실패 시 빈 문자열
		"""
		try:
			# 요소 찾기 및 클릭 가능할 때까지 대기
			elem = WebDriverWait(self.driver, self.config.element_wait_timeout).until(
				EC.element_to_be_clickable((By.XPATH, xpath))
			)

			# 요소를 화면 중앙으로 스크롤
			self.driver.execute_script(
				"arguments[0].scrollIntoView({block: 'center'});", elem
			)

			# 스크롤 완료 대기
			time.sleep(0.5)

			# 새 탭으로 열기 시도 (여러 방법 시도)
			try:
				# 방법 1: Command+Click (macOS)
				actions = ActionChains(self.driver)
				actions.key_down(Keys.COMMAND).click(elem).key_up(Keys.COMMAND).perform()
			except Exception as click_error:
				print(f"  Command+Click failed, trying alternative: {click_error}")

				# 방법 2: href 속성으로 새 탭에서 직접 열기
				href = elem.get_attribute('href')
				if href:
					self.driver.execute_script(f"window.open('{href}', '_blank');")
				else:
					# 방법 3: JavaScript로 클릭 이벤트 발생
					self.driver.execute_script("arguments[0].click();", elem)

			# 새 탭이 생성될 때까지 대기
			WebDriverWait(self.driver, self.config.element_wait_timeout).until(
				lambda driver: len(driver.window_handles) > 1
			)

			# 탭 핸들 목록에서 마지막 핸들(새 탭)로 전환
			handles = self.driver.window_handles
			if len(handles) < 2:
				return ""

			new_tab = handles[-1]
			self.driver.switch_to.window(new_tab)

			# 새 탭의 페이지 로딩 완료 대기
			WebDriverWait(self.driver, 15).until(
				lambda driver: driver.execute_script("return document.readyState") == "complete"
			)

			# 새 탭 URL 반환
			return self.driver.current_url

		except Exception as e:
			print(f"Warning: Failed to get URL in new tab: {type(e).__name__}: {e}")
			print(f"  XPath: {xpath}")
			print(f"  Current URL: {self.driver.current_url}")
			print(f"  Window handles: {len(self.driver.window_handles)}")

			# 오류 발생 시 원래 탭으로 복귀 시도
			self._return_to_original_tab()
			return ""

	def _return_to_original_tab(self) -> None:
		"""원래 탭으로 복귀 (내부 헬퍼 메서드)"""
		try:
			handles = self.driver.window_handles
			if len(handles) > 0:
				self.driver.switch_to.window(handles[0])
				print(f"  Returned to original tab")
			else:
				print(f"  No window handles available")
		except Exception as return_error:
			print(f"  Failed to return to original tab: {return_error}")

	# ============================================================================
	# Resource Management
	# ============================================================================

	def quit(self) -> None:
		"""브라우저 종료"""
		if self.driver:
			try:
				self.driver.quit()
			except Exception as e:
				print(f"Warning: Error while quitting driver: {e}")


# ============================================================================
# Factory Functions
# ============================================================================

def create_web_navigator(
	config: Optional[WebNavigatorConfig] = None
) -> WebNavigator:
	"""
	WebNavigator 인스턴스를 생성하는 팩토리 함수

	Args:
		config: 웹 네비게이터 설정

	Returns:
		WebNavigator 인스턴스
	"""
	return WebNavigator(config)
