import time
import io
import base64
from PIL import Image
from typing import Tuple
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By

class WebNavigator:
	"""웹 네비게이션을 담당하는 클래스"""
	def __init__(self, headless: bool = False, base_url: str = "https://www.kw.ac.kr/ko/index.jsp"):
		self.opts = Options()
		self.BASE_URL = base_url
		if headless:
			self.opts.add_argument('--headless')
		service = ChromeService(ChromeDriverManager().install())
		self.driver = webdriver.Chrome(service=service, options=self.opts)

	def navigate(self, url: str, window_size: Tuple[int, int] = (1080, 1440)):
		"""웹 페이지로 이동"""
		self.driver.get(url)
		time.sleep(5)  # 페이지 로딩 대기

	
	def resize_window(self, width, height):
		"""브라우저 창 크기 조정"""
		# 1. 먼저 브라우저 창을 충분히 크게 설정
		self.driver.set_window_size(width + 100, height + 100)
		
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
			self.driver.set_window_size(
				self.driver.get_window_size()['width'] + width_diff,
				self.driver.get_window_size()['height'] + height_diff
			)

	def scroll_down(self, target_height: int):
		"""스크롤 다운"""
		self.driver.execute_script(f"window.scrollTo(0, {target_height});")
		time.sleep(1)

	def scroll_to_top(self):
		"""맨 위까지 스크롤"""
		self.driver.execute_script("window.scrollTo(0, 0);")
		time.sleep(1)

	def scroll_to_bottom(self):
		"""맨 아래까지 스크롤"""
		# 현재 스크롤 위치와 이전 스크롤 위치를 비교하여 더 이상 스크롤이 안 될 때까지 반복
		last_height = self.driver.execute_script("return document.body.scrollHeight")
		while True:
			# END 키로 맨 아래로 스크롤
			actions = ActionChains(self.driver)
			actions.send_keys(Keys.END).perform()
			time.sleep(1)  # 페이지 로딩 대기
			
			# 새로운 스크롤 높이 계산
			new_height = self.driver.execute_script("return document.body.scrollHeight")
			
			# 더 이상 스크롤이 안 되면 종료
			if new_height == last_height:
				break
				
			last_height = new_height
		return last_height

	def capture_full_page(self) -> Image.Image:
		"""전체 페이지 스크린샷 캡처 (CDP 명령 사용)"""
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

	def capture_full_page_with_scroll(self, root_img: Image.Image, target_height: int) -> Image.Image:
		# # 웹 네비게이션 설정
		
		self.navigate(self.BASE_URL)

		# 브라우저 창 크기 조정
		self.resize_window(root_img.width, target_height)
		
		# scroll down to bottom
		self.scroll_to_bottom()
		self.scroll_to_top()

		time.sleep(1)
		# 전체 페이지 캡처
		web_img = self.capture_full_page()
		scale = web_img.width / root_img.width
		web_img = web_img.resize((root_img.width, int(web_img.height / scale)))
		# navigator.quit()
		return web_img
	
	def scroll_to_y(self, y: int):
		"""y 위치로 스크롤"""
		self.driver.execute_script(f"window.scrollTo(0, {y});")
		time.sleep(1)

	def get_element_xpath(self, driver, element):
		"""
		주어진 WebElement의 절대 XPath를 생성하여 반환합니다.
		예: /html/body/div[2]/ul/li[3]/a
		"""
		# JavaScript를 통해, element를 인자로 받아 절대 XPath를 계산하는 스크립트를 실행
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
					// 태그 이름과 인덱스(1-based)를 붙인다
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

	def get_element_at_coordinate_and_xpath(self, page_x, page_y):
		"""
		문서 전체 기준 (page_x, page_y) 좌표 위의 요소를 찾아서,
		그 WebElement와 절대 XPath를 반환합니다.
		"""
		# 1) 현재 스크롤 위치를 가져와서 client 좌표로 변환
		#    (Selenium은 driver.execute_script로 JS를 실행하면 리턴값을 받아올 수 있음)
		scroll_x = self.driver.execute_script("return window.scrollX;")
		scroll_y = self.driver.execute_script("return window.scrollY;")
		client_x = page_x - scroll_x
		client_y = page_y - scroll_y

		# 2) 뷰포트(client) 좌표 위의 요소를 DOM에서 검색
		#    JS 콘텍스트에서 elementFromPoint(clientX, clientY)를 호출
		get_el_script = """
		var cx = arguments[0], cy = arguments[1];
		return document.elementFromPoint(cx, cy);
		"""
		element = self.driver.execute_script(get_el_script, client_x, client_y)

		if element is None:
				return None, None

		# 3) element가 WebElement 형태로 리턴되었으므로, XPath를 계산
		xpath = self.get_element_xpath(self.driver, element)
		return element, xpath

	def get_url_in_new_tab(self, xpath: str, wait_sec: float = 1.0) -> str:
		"""
		1) driver: 이미 ChromeDriver가 실행되어 있고, 원하는 페이지가 로드된 상태여야 합니다.
		2) xpath: 새 탭으로 열고자 하는 링크(<a>)나 버튼 등의 요소를 찾기 위한 XPath 문자열.
		3) wait_sec: 새 탭이 열리고 로딩된 뒤 URL을 가져오기 위해 잠시 대기할 시간(초).
		
		반환값: 새 탭으로 연 페이지의 URL 문자열. 실패 시 빈 문자열("") 반환.
		"""
		# 1) 먼저 해당 요소를 찾는다
		elem = self.driver.find_element(By.XPATH, xpath)

		# 2) 요소에 href 속성이 있으면 그걸 따로 꺼내서 window.open으로 열기
		href = elem.get_attribute("href")
		if href:
				# 새 탭 열기
				self.driver.execute_script("window.open(arguments[0], '_blank');", href)
		else:
				# href가 없으면, Ctrl+Click 으로 새 탭 열기 시도
				elem.send_keys(Keys.CONTROL + Keys.RETURN)

		# 3) 새 탭이 떠서 로드될 시간을 잠시 준다
		time.sleep(wait_sec)

		# 4) 현재 열린 모든 창(탭) 핸들 목록을 가져와서, 새 탭 핸들로 전환
		handles = self.driver.window_handles
		if len(handles) < 2:
				# 새 탭이 열리지 않았다면 빈 문자열 반환
				return ""
		new_tab = handles[-1]
		self.driver.switch_to.window(new_tab)
		time.sleep(wait_sec)

		# 5) 새 탭의 URL을 가져오기
		new_tab_url = self.driver.current_url

		# 6) 새 탭을 닫고, 원래 창(첫 번째 핸들)으로 다시 전환
		self.driver.close()
		self.driver.switch_to.window(handles[0])

		return new_tab_url

	def quit(self):
		"""브라우저 종료"""
		self.driver.quit() 