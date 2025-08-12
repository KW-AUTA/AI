import re
import json
import requests
from typing import List


ALLOWED_TYPES = {"TEXT", "RECTANGLE", "FRAME", "INSTANCE", "VECTOR"}

def _node_shapes(node: dict):
    """
    랩퍼형: {"data": {...}, "children":[...]}
    데이터형: {...}
    """
    if "data" in node:
        data = node.get("data") or {}
        children = node.get("children") or []
    else:
        data = node or {}
        children = node.get("children", []) if isinstance(node, dict) else []
    return data, children

def extract_ui_elements(node: dict, depth: int = 0) -> list[dict]:
    elements = []
    data, children = _node_shapes(node)

    node_type = data.get("type")
    node_name = (data.get("name") or "").strip()

    if isinstance(node_type, str) and node_type in ALLOWED_TYPES and node_name:
        abspos = data.get("absolutePosition") or {}
        elements.append({
            "id": data.get("id"),
            "name": node_name,
            "type": node_type,
            "depth": depth,
            "absolutePosition": {
                "x": abspos.get("x", 0),
                "y": abspos.get("y", 0),
                "width": abspos.get("width", 0),
                "height": abspos.get("height", 0),
            },
        })

    if isinstance(children, list):
        for child in children:
            elements.extend(extract_ui_elements(child, depth + 1))

    return elements

def summarize_elements(elements: list[dict]) -> str:
    lines = []
    for el in elements:
        indent = "  " * el.get("depth", 0)
        line = f"{indent}- {el.get('name','')} [{el.get('type','')}] (id: {el.get('id')})"
        lines.append(line)
    return "\n".join(lines)

def find_elements_by_ids(frame, ids: List[str]) -> List[dict]:
    all_elements = extract_ui_elements(frame)
    idset = set(ids or [])
    return [el for el in all_elements if el.get("id") in idset]

def parse_llm_usability_report(reply: str):
    """
    Chat Completions + response_format=json_object 로 오므로 reply는 JSON 문자열.
    혹시 모를 잡텍스트가 끼면 정규식으로 {}만 추출하여 파싱.
    """
    try:
        try:
            data = json.loads(reply)
        except Exception:
            json_text = re.search(r"\{[\s\S]+\}", reply).group(0)
            data = json.loads(json_text)
        return {
            "usability_score": data.get("usability_score", 0),
            "frame_summary": data.get("frame_summary", ""),
            "problem_components": data.get("problem_components", []),
        }
    except Exception as e:
        print(f"[GPT 파싱 오류] {e}")
        return {"usability_score": 0, "frame_summary": "", "problem_components": []}

def extract_problem_ids_from_llm(reply: str) -> List[str]:
    try:
        try:
            data = json.loads(reply)
        except Exception:
            json_text = re.search(r"\{[\s\S]+\}", reply).group(0)
            data = json.loads(json_text)
        comps = data.get("problem_components", []) or []
        return [c.get("id") for c in comps if isinstance(c, dict) and c.get("id")]
    except Exception as e:
        print(f"[GPT 파싱 오류] {e}")
        return []

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

def generate_prompt_with_id(task_desc, ui_summary, persona=None, max_issues=8):
    persona = persona or "직관적인 인터페이스에 익숙한 기획자"
    return f"""
당신은 전문 UI/UX 분석가다. 단일 화면의 스크린샷(이미지)과 해당 화면의 요소 요약(JSON)을 함께 받아, 휴리스틱 기반으로 문제를 식별하고 점수를 산출한다.

입력은 두 개다:
1) 이미지: 메시지의 별도 파트로 제공됨
2) 요소 요약(JSON): 아래와 같은 배열. 각 항목은 name, id, type, position(x,y), size(width,height) 등을 포함한다.
---
{ui_summary}
---

사용자 맥락:
- 주요 사용자: {persona}
- 예상 사용자 행동(태스크): {task_desc}

평가기준(휴리스틱):
- H1 시스템 상태 가시성
- H2 현실과의 일치
- H3 사용자 통제와 자유
- H4 일관성과 표준
- H5 오류 예방
- H6 미니멀리즘
- H7 시각적 위계와 흐름
(필요 시 Nielsen 나머지도 활용 가능: H8 인지 부하 감소, H9 오류 인지·회복, H10 도움말/문서화)

출력은 JSON 한 객체만. 그 외 어떤 텍스트/마크다운/코드블록도 금지.

스키마:
{{
  "usability_score": 0..100 정수. 계산식 엄수.
  "frame_summary": 한국어 한 문장(최대 120자).
  "problem_components": [
    {{
      "id": ui_summary에 존재하는 id 문자열만. 없으면 해당 이슈를 제거.
      "name": 요소명(최대 50자). 없으면 null.
      "issue_type": ["H1","H2","H3","H4","H5","H6","H7","H8","H9","H10"] 중 하나.
      "severity": ["minor","major","critical"] 중 하나.
      "reason": 한국어 한 문장(최대 160자). 이미지 단서 + 텍스트 단서를 함께 근거로 제시.
      "evidence": {{
        "visual_cues": [짧은 구문 1~3개],    # 대비 부족, 버튼 밀집, 시선 유도 실패 등
        "bbox": [x,y,width,height] 또는 null  # 이미지 내 해당 요소의 위치를 추정. 불가하면 null
      }},
      "suggestion": 개선안 한 문장(최대 120자)
    }}
  ]
}}

제약:
- 이슈는 최대 {max_issues}개. 중복/유사 이슈는 하나로 합치기.
- id는 반드시 ui_summary의 id 중 하나여야 한다. 아니면 그 이슈는 삭제.
- 이유/개선안은 과장 표현 없이 구체적으로. “더 좋게”, “개선 필요” 같은 공허한 문구 금지.
- 이미지가 단서가 없으면 "visual_cues": ["insufficient_visual_evidence"]로 표기.
- 이슈가 없으면 "problem_components": [] 로 출력하고, "frame_summary"는 "주요 위배 없음"으로, "usability_score"는 90~100 범위에서 합리적으로 설정.

점수 계산식(반드시 준수):
1) 기본 100점에서 시작
2) 각 이슈별 감점: minor=3, major=7, critical=12
3) 동일 issue_type이 3개 이상이면 추가 페널티 -3
4) 최종 점수는 0~100 사이로 클램프
5) 이슈가 0개면 최소 90점

절차:
1) 이미지 중심으로 후보 이슈를 식별하고, ui_summary로 id 매핑 검증
2) 중복 병합 후 상위 {max_issues}개만 남김
3) 점수 계산식 적용
4) 아래 JSON만 출력

출력: JSON만. 추가 설명/프리텍스트/코드블록 금지.
"""


def load_figma_json_from_s3(figmaJsonUrl: str):
    response = requests.get(figmaJsonUrl)
    response.raise_for_status()
    figma_json = response.json()
    return figma_json