# AI - AUTA 팀 백엔드

광운대학교 캡스톤 AUTA팀 AI 백엔드 저장소입니다.

## 📋 프로젝트 개요

이 프로젝트는 UI/UX 테스트 자동화를 위한 AI 백엔드 시스템입니다. Figma 디자인과 실제 웹 페이지를 비교하여 UI 요소를 매핑하고 평가하는 기능을 제공합니다.

## 🔧 시스템 요구사항

### 운영체제
- Ubuntu 20.04+ (권장)
- 기타 Linux 배포판

### Python 환경
- Python 3.10+
- Conda 또는 venv 가상환경

## 🚀 설치 가이드

### 1. 저장소 클론
```bash
git clone https://github.com/KW-AUTA/AI.git
cd AI
```

### 2. Conda 환경 생성 및 활성화
```bash
# conda 환경 생성
conda create -n auta python=3.10
conda activate auta

# 또는 기존 auta 환경이 있다면
conda activate auta
```

### 3. Python 패키지 설치

#### 방법 1: environment.yml 사용 (권장)
```bash
# conda 환경을 한 번에 생성하고 패키지 설치
conda env create -f environment.yml
conda activate auta
```

#### 방법 2: requirements.txt 사용
```bash
pip install -r requirements.txt
```

### 4. 시스템 종속성 설치

#### 4.1 Google Chrome 설치
```bash
# Chrome GPG 키 추가
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -

# Chrome 저장소 추가
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list

# 패키지 목록 업데이트 및 Chrome 설치
sudo apt update
sudo apt install -y google-chrome-stable
```

#### 4.2 Tesseract OCR 설치
```bash
# Tesseract OCR 및 언어팩 설치
sudo apt install -y tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng

# 설치 확인
tesseract --version
```

#### 4.3 기타 시스템 라이브러리
```bash
# OpenCV 관련 의존성 (이미 설치되어 있을 수 있음)
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

### 5. 환경 변수 설정

#### 5.1 .env 파일 설정
```bash
# .env 파일에 다음 내용 추가
OPENAI_API_KEY=your-openai-api-key-here
KMP_DUPLICATE_LIB_OK=TRUE
OMP_NUM_THREADS=1
```

#### 5.2 OpenAI API 키 설정
1. [OpenAI Platform](https://platform.openai.com/)에서 API 키 발급
2. `.env` 파일의 `OPENAI_API_KEY`에 실제 키 값 입력

## 🏃‍♂️ 실행 방법

### 1. 기본 실행 테스트
```bash
# conda 환경 활성화
conda activate auta

# 애플리케이션 실행 테스트
python main.py
```

### 2. FastAPI 서버 실행
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

서버가 실행되면 `http://localhost:8000`에서 API에 접근할 수 있습니다.

## 📁 프로젝트 구조

```
AI/
├── main.py                 # FastAPI 애플리케이션 진입점
├── routes/                 # API 라우트
│   ├── dto/               # 데이터 전송 객체
│   ├── test_route.py      # 테스트 관련 라우트
│   └── evaluate.py        # UI 평가 라우트
├── service/               # 비즈니스 로직
│   ├── component_test.py  # 컴포넌트 테스트 서비스
│   └── evaluator.py       # UI 평가 서비스
├── yolo/                  # YOLO 및 이미지 처리
│   ├── element_matcher.py # 요소 매칭 알고리즘
│   ├── mapping.py         # UI 매핑 로직
│   ├── web_navigator.py   # 웹 네비게이션
│   └── visualizer.py      # 결과 시각화
├── utils/                 # 유틸리티 함수
├── model/                 # 모델 파일들
├── requirements.txt       # Python 패키지 목록
└── .env                   # 환경 변수 설정
```

### 주요 기술 스택
- **Backend**: FastAPI, Python 3.10
- **AI/ML**: YOLO, PyTorch, OpenAI API
- **Image Processing**: OpenCV, PIL
- **OCR**: Tesseract, tesserocr
- **Web Automation**: Selenium WebDriver
- **Database**: SQLAlchemy


## 📄 라이선스

이 프로젝트는 광운대학교 캡스톤 디자인 과정의 일환으로 개발되었습니다.

## 👥 팀 정보

**AUTA 팀** - 광운대학교 캡스톤 디자인  
AI 기반 UI/UX 테스트 자동화 시스템 개발
