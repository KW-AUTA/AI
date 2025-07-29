import base64
import re

def decode_base64_image(b64_string: str) -> bytes:
    # 'data:image/png;base64,...' 같은 prefix 제거
    header_removed = re.sub(r"^data:image/.+;base64,", "", b64_string.strip())

    # 패딩 보정
    missing_padding = len(header_removed) % 4
    if missing_padding:
        header_removed += '=' * (4 - missing_padding)

    return base64.b64decode(header_removed)