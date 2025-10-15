# YOLO-based Figma-Web Element Matching System

## 📋 개요
이 프로젝트는 YOLO 모델을 활용하여 Figma 디자인과 웹 페이지 간의 UI 요소를 자동으로 매칭하는 시스템입니다. 
디자인-개발 간 일치성 검증 및 자동화된 UI 테스팅을 목적으로 합니다.

## 📁 프로젝트 구조

```
yolo/
├── __init__.py                 # 패키지 초기화 및 메인 API
├── core/                       # 핵심 비즈니스 로직
│   ├── element_matcher.py      # YOLO 기반 웹 요소 추출 및 매칭
│   ├── mapping.py              # Figma-웹 요소 매핑 알고리즘
│   └── models.py               # 데이터 모델 정의
├── web/                        # 웹 관련 기능
│   └── web_navigator.py        # Selenium 기반 웹 네비게이션
├── figma/                      # Figma 관련 기능
│   ├── figma.py                # Figma 데이터 로더
│   └── figma_visualizer.py     # Figma 시각화
├── visualization/              # 시각화 모듈
│   ├── visualizer.py           # 일반 시각화 도구
│   ├── visualize_interaction.py # 인터랙션 시각화
│   └── tree_visualizer.py      # DOM 트리 시각화
├── utils/                      # 유틸리티 함수
│   ├── utils.py                # 공통 유틸리티
│   ├── tree_loader.py          # 트리 구조 로더
│   ├── errorChecker.py         # 에러 체크 유틸리티
│   └── error_list.py           # 에러 정의
├── models_weights/             # AI 모델 가중치
│   ├── best.pt                 # YOLO 모델 (19MB)
│   └── screensim-resnet-uda+web7k.torchscript # 화면 유사성 모델 (45MB)
├── data/                       # 데이터 파일
│   ├── framesData.json         # Figma 프레임 데이터
│   └── ttests.json             # 테스트 데이터
├── docs/                       # 문서화
│   ├── README.md               # 이 파일
│   └── memo.md                 # 개발 메모
├── tests/                      # 테스트 코드 (예정)
├── webui/                      # Web UI 서브모듈
└── weight/                     # 추가 모델 가중치
```

## 🔧 주요 기능

### 1. 웹 요소 추출 (`core/element_matcher.py`)
- YOLO 모델을 사용한 웹 페이지 UI 요소 검출
- OCR을 통한 텍스트 추출
- 특징 벡터 생성 및 분류

### 2. Figma 데이터 처리 (`figma/`)
- Figma JSON 데이터 파싱
- 요소 위치 및 속성 정보 추출
- 시각화 및 분석 도구

### 3. 매칭 알고리즘 (`core/mapping.py`)
- 시각적 유사성 기반 매칭
- 텍스트 유사성 분석
- 헝가리안 알고리즘을 통한 최적 매칭

### 4. 웹 자동화 (`web/web_navigator.py`)
- Selenium 기반 브라우저 제어
- 스크린샷 캡처
- 동적 페이지 로딩 대기

## 🚀 사용법

### 기본 사용 예시
```python
from yolo import ElementExtractor, WebNavigator
from yolo.figma import FigmaDataLoader

# 웹 네비게이터 초기화
navigator = WebNavigator(headless=False)
navigator.navigate("https://example.com")

# 요소 추출기 초기화
extractor = ElementExtractor()
screenshot = navigator.capture_screenshot()
web_elements = extractor.extract_elements(screenshot)

# Figma 데이터 로드
figma_loader = FigmaDataLoader("design.json")
figma_elements = figma_loader.get_elements()

# 매칭 수행
from yolo.core.mapping import perform_matching
matches = perform_matching(figma_elements, web_elements)
```

## 📦 의존성
- Python 3.8+
- PyTorch
- Ultralytics YOLO
- Selenium
- PIL/Pillow
- OpenCV
- tesserocr (OCR)
- scipy

## 🔄 설치 및 설정
```bash
# 의존성 설치
pip install torch ultralytics selenium pillow opencv-python scipy

# ChromeDriver 설치 (자동)
pip install webdriver-manager

# OCR 설치 (선택사항)
pip install tesserocr
```

## 📊 모델 정보
- **YOLO 모델**: UI 요소 검출을 위한 커스텀 훈련 모델
- **화면 유사성 모델**: ResNet 기반 화면 비교 모델
- **지원 요소**: 버튼, 텍스트, 이미지, 입력 필드 등

## 🎯 활용 사례
1. **디자인-개발 QA**: Figma 디자인과 실제 웹 구현 간 일치성 검증
2. **자동 UI 테스팅**: 디자인 기반 자동화된 인터랙션 테스트
3. **접근성 검사**: UI 요소의 위치 및 크기 적합성 검증
4. **반응형 디자인 검증**: 다양한 화면 크기에서의 레이아웃 일치성 확인

## 📝 라이센스
이 프로젝트의 라이센스 정보는 프로젝트 루트의 LICENSE 파일을 참조하세요.