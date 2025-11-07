# GENERAL ERROR
G_ERROR_COORDINATE_X = 'X 좌표 오차'
G_ERROR_COORDINATE_Y = 'Y좌표 오차'
G_ERROR_SIZE = '크기 오차'
G_ERROR_TEXT = '텍스트 오차'
G_ERROR_NOT_MATCHED = '매칭 안됨'
G_ERROR_ASPECT_RATIO = '가로세로 비율 오차'
G_ERROR_FEATURE_DISSIMILAR = '시각적 특징 불일치'
G_ERROR_LOW_OVERALL_SCORE = '낮은 전체 스코어'
G_ERROR_RELATIVE_POSITION = '상대 좌표 오차'

# ROUTE ERROR
R_ERROR_ROUTING = '라우트 오차'
R_ERROR_SESSION_INVALID = '브라우저 세션 무효' 

# INTERACTION ERROR
I_ERROR_NAVIGATE_NOT_OVERLAY = '네비게이트 아닌 오버레이'

I_ERROR_BACK_NOT_BACK = '백 아닌 경우 백'
I_ERROR_OVERLAY_NOT_FOUND = '오버레이 없음'
I_ERROR_DIFFERENT_OVERLAY = '다른 오버레이'

I_ERROR_DIFFERENT_BACK = '다른 뒤로가기'

NORMAL = '정상'  # 오차가 없는 경우


# 상세 정보 메시지 함수들
def detail_coordinate_x(diff_px: float) -> str:
    """X 좌표 차이 상세 정보"""
    direction = "오른쪽" if diff_px > 0 else "왼쪽"
    return f"피그마 요소가 X축으로 {abs(diff_px):.1f}px {direction}에 있습니다"


def detail_coordinate_y(diff_px: float) -> str:
    """Y 좌표 차이 상세 정보"""
    direction = "아래" if diff_px > 0 else "위"
    return f"피그마 요소가 Y축으로 {abs(diff_px):.1f}px {direction}에 있습니다"


def detail_size(diff_px: float) -> str:
    """크기 차이 상세 정보"""
    direction = "크게" if diff_px > 0 else "작게"
    return f"피그마 요소가 {abs(diff_px):.1f}px {direction} 되어 있습니다"


def detail_aspect_ratio(figma_ratio: float, web_ratio: float) -> str:
    """가로세로 비율 차이 상세 정보"""
    return f"가로세로 비율이 다릅니다 (피그마: {figma_ratio:.2f}, 웹: {web_ratio:.2f})"


def detail_image_similarity(distance: float, threshold: float = 0.35) -> str:
    """이미지 유사도 상세 정보"""
    similarity_percent = max(0, (1 - distance / threshold) * 100)
    return f"이미지 유사도가 {similarity_percent:.1f}%로 낮습니다 (거리값: {distance:.3f})"


def detail_url_difference(expected: str, actual: str) -> str:
    """URL 차이 상세 정보"""
    return f"예상 URL: {expected}, 실제 URL: {actual}"


def detail_text_difference(figma_text: str, web_text: str) -> str:
    """텍스트 차이 상세 정보"""
    return f"텍스트가 다릅니다 (피그마: '{figma_text}', 웹: '{web_text}')"


def detail_feature_similarity(similarity: float) -> str:
    """시각적 특징 유사도 상세 정보"""
    percent = similarity * 100
    return f"시각적 특징 유사도가 {percent:.1f}%로 낮습니다"


def detail_overall_score(score: float) -> str:
    """전체 매칭 스코어 상세 정보"""
    percent = score * 100
    return f"전체 매칭 스코어가 {percent:.1f}%로 낮습니다"


def detail_relative_position(rel_diff_x: float, rel_diff_y: float) -> str:
    """상대 좌표 차이 상세 정보"""
    x_percent = rel_diff_x * 100
    y_percent = rel_diff_y * 100
    return f"화면 기준 위치가 X축 {x_percent:.1f}%, Y축 {y_percent:.1f}% 차이납니다"
