"""
이미지 유틸리티 모듈

시각화 모듈에서 공통으로 사용하는 이미지 처리 함수들을 제공합니다.
"""

import base64
import io
import sys
from typing import Optional
from PIL import Image
import requests


def decode_image_from_reference(img_ref: str) -> Optional[Image.Image]:
    """
    base64 문자열이나 URL에서 이미지를 PIL Image로 변환

    Args:
        img_ref: Base64 인코딩된 문자열 또는 HTTP(S) URL

    Returns:
        PIL Image 객체. 실패 시 None

    Examples:
        >>> # From URL
        >>> img = decode_image_from_reference("https://example.com/image.png")

        >>> # From base64
        >>> img = decode_image_from_reference("data:image/png;base64,iVBORw0KGgo...")

        >>> # From base64 without prefix
        >>> img = decode_image_from_reference("iVBORw0KGgo...")
    """
    if not img_ref or not isinstance(img_ref, str):
        return None

    # HTTP(S) URL인 경우
    if img_ref.startswith("http"):
        return _download_image_from_url(img_ref)

    # Base64 인코딩된 문자열인 경우
    return _decode_base64_image(img_ref)


def _download_image_from_url(url: str) -> Optional[Image.Image]:
    """
    URL에서 이미지 다운로드

    Args:
        url: 이미지 URL

    Returns:
        PIL Image 객체. 실패 시 None
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print(
            f"Warning: Could not download image from URL {url}. Error: {e}",
            file=sys.stderr
        )
        return None
    except Exception as e:
        print(
            f"Warning: Could not open image from URL {url}. Error: {e}",
            file=sys.stderr
        )
        return None


def _decode_base64_image(base64_str: str) -> Optional[Image.Image]:
    """
    Base64 인코딩된 문자열을 PIL Image로 디코딩

    Args:
        base64_str: Base64 인코딩된 문자열 (data URI 프리픽스 포함 가능)

    Returns:
        PIL Image 객체. 실패 시 None
    """
    # data URI 프리픽스 제거
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",", 1)[1]

    # 공백 및 개행 제거
    base64_str = base64_str.replace('\n', '').replace('\r', '').replace(' ', '')

    # Base64 패딩 추가
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)

    try:
        img_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        print(
            f"Warning: Could not decode or open image from base64 string. Error: {e}",
            file=sys.stderr
        )
        return None


class ImageLoader:
    """
    이미지 로딩을 담당하는 클래스

    다양한 소스(URL, base64)에서 이미지를 로드하는 기능을 제공합니다.
    """

    @staticmethod
    def load_from_reference(img_ref: str) -> Optional[Image.Image]:
        """
        참조(URL 또는 base64)에서 이미지 로드

        Args:
            img_ref: 이미지 참조 (URL 또는 base64 문자열)

        Returns:
            PIL Image 객체. 실패 시 None
        """
        return decode_image_from_reference(img_ref)

    @staticmethod
    def load_from_url(url: str) -> Optional[Image.Image]:
        """
        URL에서 이미지 로드

        Args:
            url: 이미지 URL

        Returns:
            PIL Image 객체. 실패 시 None
        """
        return _download_image_from_url(url)

    @staticmethod
    def load_from_base64(base64_str: str) -> Optional[Image.Image]:
        """
        Base64 문자열에서 이미지 로드

        Args:
            base64_str: Base64 인코딩된 문자열

        Returns:
            PIL Image 객체. 실패 시 None
        """
        return _decode_base64_image(base64_str)
