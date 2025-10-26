import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from PIL import ImageDraw, Image
from io import BytesIO
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from utils.figma import (
    extract_ui_elements,
    summarize_elements,
    find_elements_by_ids,
    parse_llm_usability_report,
    extract_problem_ids_from_llm,
    generate_task_from_frame_name,
    generate_persona,
    generate_prompt_with_id,
    load_figma_json_from_s3,
)
from utils.image import decode_base64_image

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

def _fetch_image_bytes(image_url: str) -> bytes:
    if not image_url:
        raise ValueError("image_url is empty")
    if image_url.startswith("data:image/"):
        return decode_base64_image(image_url)
    if image_url.startswith("http://") or image_url.startswith("https://"):
        r = requests.get(image_url, timeout=20)
        r.raise_for_status()
        return r.content
    raise ValueError(f"Unsupported image URL: {image_url[:50]}")

def _safe_get_abs_pos(frame_data: dict):
    pos = frame_data.get("absolutePosition") or {}
    need = ("x", "y", "width", "height")
    return pos if all(k in pos for k in need) else None

async def evaluate_all_frames_logic(figma_url: str):
    logger.info("Figma 평가 로직 시작: %s", figma_url)

    figmaJson = load_figma_json_from_s3(figma_url)
    # ⚠️ 중요한 변경: data만 뽑지 말고, 노드 전체( children 포함 )를 그대로 보관
    frames = [node for node in figmaJson.get("tree", [])]
    logger.info("총 %d개 프레임 감지됨", len(frames))

    results = []
    scores = []
    unique_id = f"{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    save_dir = STATIC_DIR / unique_id
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("이미지 저장 디렉토리 생성됨: %s", save_dir)

    for idx, frame in enumerate(frames):
        frame_data = frame.get("data", {}) or {}
        frame_name = frame_data.get("name", f"Unnamed_{idx}")
        logger.info("프레임 분석 시작: %s", frame_name)
        try:
            # 요소 요약은 children까지 보는 extract_ui_elements를 위해 frame(랩퍼) 자체 전달
            elements = extract_ui_elements(frame)
            summary = summarize_elements(elements)
            task = generate_task_from_frame_name(frame_name)
            persona = generate_persona()
            prompt = generate_prompt_with_id(task, summary, persona)

            image_url = frame_data.get("image")
            if not image_url:
                raise ValueError("No image URL found in frame.data.image")

            logger.info("GPT-4o 평가 요청 시작 (프레임: %s)", frame_name)
            # ✅ Chat Completions + JSON Object 강제 (구버전 SDK도 동작)
            resp = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "넌 UI/UX 평가 전문가야."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            llm_reply = resp.choices[0].message.content  # JSON 문자열
            logger.info("GPT 응답 수신 완료 (프레임: %s)", frame_name)

            # 하이라이트 박스 계산
            problem_ids = extract_problem_ids_from_llm(llm_reply)
            bounding_elements = find_elements_by_ids(frame, problem_ids)

            img_bytes = _fetch_image_bytes(image_url)
            image = Image.open(BytesIO(img_bytes)).convert("RGB")

            abs_pos = _safe_get_abs_pos(frame_data)
            if abs_pos:
                scale_x = image.width / max(1, abs_pos["width"])
                scale_y = image.height / max(1, abs_pos["height"])
                origin_x = abs_pos["x"]
                origin_y = abs_pos["y"]

                draw = ImageDraw.Draw(image)
                for el in bounding_elements:
                    apos = (el.get("absolutePosition") or {})
                    try:
                        x1 = (apos["x"] - origin_x) * scale_x
                        y1 = (apos["y"] - origin_y) * scale_y
                        x2 = (apos["x"] + apos["width"] - origin_x) * scale_x
                        y2 = (apos["y"] + apos["height"] - origin_y) * scale_y
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    except KeyError:
                        continue
            else:
                logger.warning("absolutePosition 누락: 프레임 %s (bbox 스킵)", frame_name)

            filename = f"{uuid.uuid4().hex}.png"
            save_path = save_dir / filename
            image.save(save_path, format="PNG")
            logger.info("하이라이트 이미지 저장 완료: %s", save_path)

            # 점수/요약 파싱
            evaluation = parse_llm_usability_report(llm_reply)

            scores.append(int(evaluation.get("usability_score", 0)))
            results.append({
                "frameSummary": evaluation.get("frame_summary", ""),
                "highlightImageUrl": f"/static/{unique_id}/{filename}",
            })
            logger.info("프레임 평가 완료: %s (점수: %s)", frame_name, evaluation.get("usability_score"))

        except Exception as e:
            logger.error("프레임 평가 실패: %s, 이유: %s", frame_name, e)
            results.append({"frame_name": frame_name, "error": str(e)})
            continue  # 다음 프레임 계속

    overall_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    logger.info("전체 평가 종료. 종합 점수: %s", overall_score)

    return {"usabilityScore": int(overall_score), "evaluations": results}
