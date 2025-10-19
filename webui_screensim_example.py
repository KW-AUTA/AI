"""
기존 WebUI screensim 모델 사용 예제
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import gdown

def setup_screensim():
    """WebUI screensim 모델 설정"""
    
    # 1. 필요한 모델 다운로드
    model_path = "screensim-resnet-uda+web350k.torchscript"
    
    if not os.path.exists(model_path):
        print("모델 다운로드 중...")
        # WebUI의 최고 성능 screensim 모델
        model_url = "https://drive.google.com/file/d/1WCofe3JUDT_AJNVLXjVxWsBurLe0wcjQ/view?usp=share_link"
        gdown.download(model_url, model_path, fuzzy=True, use_cookies=False)
        print(f"모델 다운로드 완료: {model_path}")
    
    # 2. 모델 로드
    model = torch.jit.load(model_path)
    model.eval()
    
    # 3. 이미지 변환 설정 (WebUI와 동일)
    img_transforms = transforms.Compose([
        transforms.Resize((256, 128)),  # WebUI screensim 입력 크기
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return model, img_transforms

def calculate_screen_similarity(image1_path, image2_path, model, img_transforms):
    """두 스크린 이미지의 유사도 계산"""
    
    # 1. 이미지 로드 및 전처리
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')
    
    img1_tensor = img_transforms(img1).unsqueeze(0)
    img2_tensor = img_transforms(img2).unsqueeze(0)
    
    # 2. 임베딩 추출
    with torch.no_grad():
        embedding1 = model(img1_tensor)
        embedding2 = model(img2_tensor)
    
    # 3. 유클리드 거리 계산
    distance = torch.linalg.norm(embedding1 - embedding2)
    
    # 4. 유사도 판단 (WebUI 기본 마진)
    margin = (0.2 + 0.5) / 2  # margin_pos=0.2, margin_neg=0.5의 평균
    is_similar = float(distance) < margin
    
    return float(distance), is_similar

def main():
    """사용 예제"""
    
    print("=== WebUI Screensim 사용 예제 ===")
    
    # 1. 모델 설정
    try:
        model, img_transforms = setup_screensim()
        print("모델 로드 완료!")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 2. 예제 이미지 (실제 이미지 경로로 변경 필요)
    image1 = "example1.png"  # 첫 번째 스크린샷 경로
    image2 = "example2.png"  # 두 번째 스크린샷 경로
    
    # 현재 디렉토리의 png 파일 확인
    import glob
    png_files = glob.glob("*.png")
    
    if len(png_files) >= 2:
        image1, image2 = png_files[0], png_files[1]
        print(f"사용할 이미지: {image1}, {image2}")
        
        # 3. 유사도 계산
        try:
            distance, is_similar = calculate_screen_similarity(
                image1, image2, model, img_transforms
            )
            
            print(f"\n=== 결과 ===")
            print(f"거리: {distance:.3f}")
            print(f"유사한 스크린: {is_similar}")
            print(f"기준 마진: {(0.2 + 0.5) / 2:.3f}")
            
        except Exception as e:
            print(f"유사도 계산 실패: {e}")
    else:
        print("테스트할 이미지가 부족합니다 (최소 2개의 PNG 파일 필요)")
        print(f"현재 디렉토리의 PNG 파일: {png_files}")

# 기존 프로젝트에 통합하는 방법
def integrate_with_existing_project():
    """기존 프로젝트 통합 방법"""
    
    integration_code = '''
# yolo/mapping.py에 추가할 수 있는 코드:

import torch
from torchvision import transforms
import gdown
import os

class ScreenSimilarityChecker:
    def __init__(self):
        self.model = None
        self.transforms = None
        self.load_model()
    
    def load_model(self):
        """screensim 모델 로드"""
        model_path = "screensim-resnet-uda+web350k.torchscript"
        
        if not os.path.exists(model_path):
            # 모델 다운로드
            model_url = "https://drive.google.com/file/d/1WCofe3JUDT_AJNVLXjVxWsBurLe0wcjQ/view?usp=share_link"
            gdown.download(model_url, model_path, fuzzy=True, use_cookies=False)
        
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        self.transforms = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def check_layout_similarity(self, figma_img, web_img):
        """레이아웃 유사도 확인"""
        if self.model is None:
            return None, False
        
        # PIL Image로 변환 (필요시)
        if isinstance(figma_img, str):
            figma_img = Image.open(figma_img)
        if isinstance(web_img, str):
            web_img = Image.open(web_img)
        
        # 전처리
        figma_tensor = self.transforms(figma_img).unsqueeze(0)
        web_tensor = self.transforms(web_img).unsqueeze(0)
        
        # 임베딩 계산
        with torch.no_grad():
            figma_embedding = self.model(figma_tensor)
            web_embedding = self.model(web_tensor)
        
        # 거리 계산
        distance = torch.linalg.norm(figma_embedding - web_embedding)
        margin = 0.35  # 조정 가능
        is_similar = float(distance) < margin
        
        return float(distance), is_similar

# mapping 함수에서 사용:
def mapping(figma_img_path, web_img_path, ...):
    # 기존 코드...
    
    # 레이아웃 유사도 체크 추가
    similarity_checker = ScreenSimilarityChecker()
    layout_distance, layout_similar = similarity_checker.check_layout_similarity(
        figma_img_path, web_img_path
    )
    
    if layout_similar:
        print(f"레이아웃이 유사합니다 (거리: {layout_distance:.3f})")
        # 기존 매칭 로직 사용
    else:
        print(f"레이아웃이 다릅니다 (거리: {layout_distance:.3f})")
        # 더 엄격한 매칭 기준 사용하거나 다른 전략 적용
    
    # 나머지 매칭 로직...
'''
    
    print("=== 기존 프로젝트 통합 방법 ===")
    print(integration_code)

if __name__ == "__main__":
    main()
    print("\n" + "="*50)
    integrate_with_existing_project()
