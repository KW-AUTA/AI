from fastapi import APIRouter
from routes.dto.request import UITestRequest
from service.evaluator import evaluate_all_frames_logic
from fastapi.responses import JSONResponse
from routes.dto.response import UITestResponse
router = APIRouter()

@router.post("/evaluate-ui-with-highlight")
async def evaluate_ui_with_highlight(request: UITestRequest):
    result = await evaluate_all_frames_logic(request.figmaJsonUrl)
    return UITestResponse(**result)