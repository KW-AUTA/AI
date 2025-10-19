# OCR 개선 사항

## 요약

어두운 배경 + 밝은 텍스트 이미지에서 OCR 인식률을 개선했습니다.

## 주요 변경사항

### 1. Feature 유사도 계산 수정
**파일**: `core/matcher.py`

**문제**:
- 코사인 유사도 [-1, 1]을 [0, 1]로 변환하면서 랜덤 벡터도 ~0.5의 유사도를 가짐
- 전혀 다른 이미지가 높은 유사도를 보이는 문제

**해결**:
```python
# 이전: similarity = (similarity + 1) / 2
# 개선: similarity = np.clip(similarity, 0.0, 1.0)
```

**결과**:
- 랜덤 벡터: 0.5 → 0.01 유사도
- 같은 이미지: 0.75 → 1.0 유사도 (변화 없음)

### 2. XOR 케이스 매칭 보호
**파일**: `core/matcher.py:569-579`

**문제**:
- OCR이 한쪽만 실패하면 텍스트 체크를 건너뜀
- Feature 유사도만으로 잘못된 매칭 발생

**해결**:
```python
# 한쪽만 텍스트가 있을 때 (XOR 케이스)
if figma_has ^ web_has:
    if len(longer_text) >= 3 and feat_sim < 0.85:
        # 거부: Feature 유사도가 85% 미만이면 매칭 안 함
        continue
```

**환경변수**: `MIN_FEAT_SIM_XOR=0.85`

### 3. OCR 전처리 개선
**파일**: `core/element_matcher.py:191-284`

**문제**:
- Tesseract가 어두운 배경 + 밝은 텍스트를 인식 못 함

**해결**:
- 배경 밝기 자동 감지 (외곽 영역 평균 밝기)
- 어두운 배경(< 128) 감지 시 `THRESH_BINARY_INV` 사용
- Dilation + Closing 연산으로 외곽선 텍스트 채우기

```python
# 배경 밝기 감지
background_brightness = np.mean([
    top_border, bottom_border,
    left_border, right_border
])

if background_brightness < 128:
    # 어두운 배경 → 반전 처리
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    # Dilation으로 외곽선 채우기
    binary = cv2.dilate(binary, kernel, iterations=2)
```

**결과**: 전처리는 개선되었으나 외곽선 스타일 텍스트는 Tesseract 한계로 여전히 실패

### 4. PaddleOCR 통합 (별도 사용)
**파일**: `test_paddle_standalone.py`

**문제**:
- Tesseract가 외곽선 텍스트 인식 실패
- PaddleOCR은 딥러닝 기반으로 복잡한 텍스트도 인식 가능

**해결**:
- PaddleOCR 설치 및 통합
- macOS에서 YOLO와 커널 충돌 발생으로 **별도 스크립트**로 제공

**사용법**:
```bash
python test_paddle_standalone.py
```

**결과**:
- Tesseract: 0% 인식 (빈 문자열)
- PaddleOCR: 100% 인식 (`Y Combinator Product clarity`)

## 테스트 파일

### 1. `test_feature_similarity.py`
Feature 유사도 계산 개선 테스트
- 랜덤 벡터: ~0.01 유사도
- 같은 벡터: 1.0 유사도

### 2. `test_empty_text_case.py`
빈 텍스트 XOR 케이스 테스트
- 한쪽 텍스트 비어있을 때 동작 확인

### 3. `test_text_threshold.py`
텍스트 유사도 임계값 테스트
- `MIN_TEXT_SIM_BOTH=0.50` 동작 확인

### 4. `test_ocr_improvement.py`
OCR 전처리 개선 테스트
- 어두운 배경 이미지 전처리 확인

### 5. `test_xor_realistic.py`
XOR 케이스 실전 테스트
- Feature 유사도에 따른 매칭 여부 확인

### 6. `test_paddle_standalone.py` ⭐
**PaddleOCR 단독 테스트 (권장)**
- YOLO 없이 PaddleOCR만 사용
- macOS에서 안정적으로 작동
- 어두운 배경 이미지 100% 인식 성공

## 환경변수

```bash
# Feature 유사도 (XOR 케이스)
MIN_FEAT_SIM_XOR=0.85

# 텍스트 유사도 (둘 다 텍스트 있을 때)
MIN_TEXT_SIM_BOTH=0.50

# 크기 유사도
MIN_SIZE_SIM=0.30

# 디버그 모드
SIM_DEBUG=1
```

## 제한사항

### macOS PaddlePaddle 커널 충돌
- **문제**: YOLO(Ultralytics)와 PaddlePaddle이 동시에 로드되면 커널 충돌 발생
- **오류**: `Unsupported kernel argument type`
- **해결**: PaddleOCR을 별도 스크립트(`test_paddle_standalone.py`)로 사용
- **권장**: 프로덕션 환경에서는 Linux 사용 또는 EasyOCR 고려

### Tesseract 한계
- 외곽선 스타일 텍스트 인식 불가
- 어두운 배경 전처리로 일부 개선되었으나 완벽하지 않음
- PaddleOCR/EasyOCR 같은 딥러닝 기반 OCR 권장

## 다음 단계

1. **프로덕션 환경**: Linux에서 PaddleOCR 통합 시도
2. **대안 OCR**: EasyOCR 고려 (PyTorch 기반, 충돌 적음)
3. **전처리 추가 개선**: Transformer 기반 문서 전처리 모델 사용

## 커밋 내역

- Feature 유사도 계산 수정
- XOR 케이스 매칭 보호 추가
- OCR 전처리 개선 (배경 감지 + 반전)
- PaddleOCR 통합 (별도 스크립트)
