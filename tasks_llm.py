
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from schemas import LLMRequest
import time

router = APIRouter(prefix="/vision", tags=["LLM"])

@router.post("/llm_stream")
async def llm_vision_stream(request: LLMRequest):
    """
    大模型视觉问答（流式响应）
    """
    async def generate():
        # TODO: 调用大模型
        response_text = "This is a simulated response from the Vision LLM."
        for word in response_text.split():
            yield f"data: {word}\n\n"
            time.sleep(0.1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")
