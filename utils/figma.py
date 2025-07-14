import re
import json
import requests
from typing import List


# 유틸 함수들
def extract_ui_elements(node: dict, depth: int = 0) -> list[dict]:
    elements = []
    allowed_types = {"TEXT", "RECTANGLE", "FRAME", "INSTANCE", "VECTOR"}
    node_type = node.get("type")
    node_name = node.get("name", "").strip()
    if node_type in allowed_types and node_name:
        elements.append({
            "id": node["id"],
            "name": node_name,
            "type": node_type,
            "depth": depth,
            "absoluteX": node.get("absoluteX", 0),
            "absoluteY": node.get("absoluteY", 0),
            "width": node.get("width", 0),
            "height": node.get("height", 0)
        })
    for child in node.get("children", []):
        elements.extend(extract_ui_elements(child, depth + 1))
    return elements

def summarize_elements(elements: list[dict]) -> str:
    lines = []
    for el in elements:
        indent = "  " * el["depth"]
        line = f"{indent}- {el['name']} [{el['type']}] (id: {el['id']})"
        lines.append(line)
    return "\n".join(lines)

def find_elements_by_ids(frame, ids: List[str]) -> List[dict]:
    all_elements = extract_ui_elements(frame)
    return [el for el in all_elements if el["id"] in ids]


def parse_llm_usability_report(reply: str):
    try:
        json_text = re.search(r"\{[\s\S]+\}", reply).group(0)
        data = json.loads(json_text)
        return {
            "usability_score": data.get("usability_score", 0),
            "problem_components": data.get("problem_components", [])
        }
    except Exception as e:
        print(f"[GPT 파싱 오류] {e}")
        return {
            "usability_score": 0,
            "problem_components": []
        }


def extract_problem_ids_from_llm(reply: str) -> List[str]:
    try:
        json_text = re.search(r"\{[\s\S]+\}", reply).group(0)
        json_data = json.loads(json_text)
        components = json_data.get("problem_components", [])
        return [comp["id"] for comp in components if "id" in comp]
    except Exception as e:
        print(f"[GPT 파싱 오류] {e}")
        return []


def generate_task_from_frame_name(frame_name: str) -> str:
    return f"{frame_name} 화면에서 사용자가 작업을 수행하는 데 직관적인지 평가해주세요."


def generate_persona() -> str:
    return "qa를 담당하는 기획자"

def generate_prompt_with_id(task_desc, ui_summary, persona=None):
    prompt = f"""
    당신은 전문적인 UI/UX 분석가이며, 실제 서비스의 사용자 경험을 평가하는 역할을 맡고 있습니다.

    지금부터 제시되는 이미지는 하나의 UI 화면을 나타냅니다.  
    이 이미지를 **시각적으로 직접 분석**하고, 아래에 제공되는 텍스트 기반 UI 요소 정보와 사용자 시나리오를 함께 고려하여 문제점과 개선점을 평가해주세요.

    ---

    [사용자 정보]
    - 주요 사용자 유형: {persona or "직관적인 인터페이스에 익숙한 기획자"}
    - 예상 사용자 행동: {task_desc}

    [이미지 설명]
    - 이 이미지는 실제 UI를 캡처한 것이며, 버튼/텍스트/입력창/카드 등의 구성 요소가 시각적으로 배치되어 있습니다.
    - UI 요소 간의 거리, 정렬, 대비, 시선 흐름 등을 포함한 **전체적인 시각 경험**을 함께 고려해야 합니다.

    [요소 요약 목록]
    각 요소는 name, id, type, 위치, 크기 정보를 포함합니다:

    {ui_summary}

    ---

    다음 기준에 따라 UI를 평가하고, 문제가 있는 요소의 id, name, issue_type, reason을 JSON으로 작성하세요:

    1. **시스템 상태 가시성**: 로딩, 진행 상태, 피드백이 적절히 전달되는가?
    2. **현실과의 일치**: 사용자가 익숙한 개념/아이콘/언어가 사용되었는가?
    3. **사용자 통제와 자유**: 되돌리기, 취소, 빠져나가기 같은 선택의 자유가 제공되는가?
    4. **일관성과 표준**: 동일한 기능은 동일한 디자인인가? 내부/외부 표준을 따르는가?
    5. **오류 예방**: 잘못된 입력이나 실수를 사전에 막아주는 장치가 있는가?
    6. **미니멀리즘**: 과도한 정보, 불필요한 시각 요소 없이 본질적인 정보에 집중했는가?
    7. **시각적 위계와 흐름**: 가장 중요한 정보에 시선이 자연스럽게 도달하는가? 시각적 혼란은 없는가?

    ---

    **요구 출력 형식 (JSON만 출력하세요):**

    ```json
    {{
      "usability_score": int,  // 전체 평가 점수 (0~100)
      "problem_components": [
        {{
          "id": "string",
          "name": "string",
          "issue_type": "string",  // 위 기준 중 하나
          "reason": "string"       // 이미지 기반 시각 분석 + 텍스트 정보 모두를 반영한 판단 근거
        }}
      ]
    }}

    주의사항
    - JSON 외 텍스트는 절대 출력하지 마세요.
    - 이미지 기반 판단을 최우선으로 하되, 요소 텍스트 요약과 사용자 유형/시나리오도 반드시 반영하십시오
    """
    return prompt


def load_figma_json_from_s3(figmaJsonUrl: str):
    response = requests.get(figmaJsonUrl)
    response.raise_for_status()
    figma_json = response.json()
    return figma_json