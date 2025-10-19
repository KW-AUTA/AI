import os
import uuid
from datetime import datetime
from pathlib import Path
from PIL import ImageDraw, Image
from io import BytesIO

from openai import OpenAI

from utils.figma import (
    extract_ui_elements,
    summarize_elements,
    find_elements_by_ids,
    parse_llm_usability_report,
    extract_problem_ids_from_llm,
    generate_task_from_frame_name,
    generate_persona,
    generate_prompt_with_id,
    load_figma_json_from_s3
)
from utils.image import decode_base64_image

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

async def evaluate_all_frames_logic(figma_url: str):
    figmaJson = load_figma_json_from_s3(figma_url)
    frames = [node["data"] for node in figmaJson.get("tree", [])]

    results = []
    scores = []
    unique_id = f"{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    save_dir = STATIC_DIR / unique_id
    save_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        try:
            elements = extract_ui_elements(frame)
            summary = summarize_elements(elements)
            task = generate_task_from_frame_name(frame.get("name", "frame"))
            persona = generate_persona()
            prompt = generate_prompt_with_id(task, summary, persona)

            image_url = frame.get("image")
            if not image_url:
                raise ValueError("No image URL found in frame")

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "넌 UI/UX 평가 전문가야."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            llm_reply = response.choices[0].message.content

            problem_ids = extract_problem_ids_from_llm(llm_reply)
            bounding_elements = find_elements_by_ids(frame, problem_ids)

            image_data = decode_base64_image(image_url)
            image = Image.open(BytesIO(image_data)).convert("RGB")

            frame_pos = frame["absolutePosition"]
            frame_width = frame_pos["width"]
            frame_height = frame_pos["height"]
            origin_x = frame_pos["x"]
            origin_y = frame_pos["y"]

            scale_x = image.width / frame_width
            scale_y = image.height / frame_height

            draw = ImageDraw.Draw(image)
            for el in bounding_elements:
                abs_pos = el["absolutePosition"]
                x1 = (abs_pos["x"] - origin_x) * scale_x
                y1 = (abs_pos["y"] - origin_y) * scale_y
                x2 = (abs_pos["x"] + abs_pos["width"] - origin_x) * scale_x
                y2 = (abs_pos["y"] + abs_pos["height"] - origin_y) * scale_y
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            filename = f"{uuid.uuid4().hex}.png"
            save_path = save_dir / filename
            image.save(save_path, format="PNG")

            evaluation = parse_llm_usability_report(llm_reply)

            scores.append(evaluation["usability_score"])
            results.append({
                "frameSummary": evaluation["frame_summary"],
                "highlightImageUrl": f"/static/{unique_id}/{filename}"
            })
        except Exception as e:
            print(f"[ERROR] Frame {frame.get('name', 'Unnamed')} 평가 실패: {e}")
            results.append({
                "frame_name": frame.get("name", "Unnamed Frame"),
                "error": str(e)
            })
        break
    overall_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    return {
        "usabilityScore": int(overall_score),
        "evaluations": results
    }
