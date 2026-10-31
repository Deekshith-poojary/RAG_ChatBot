from fastapi import APIRouter

from rag_chatbot.api.dependencies import ContainerDep
from rag_chatbot.api.schemas import GenerateRequest, GenerateResponse, OllamaModel

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/models", response_model=list[OllamaModel])
async def list_models(container: ContainerDep) -> list[OllamaModel]:
    """Models available on the configured Ollama endpoint."""
    models = await container.llm.list_models()
    return [OllamaModel(name=m["name"], size_bytes=m.get("size")) for m in models]


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, container: ContainerDep) -> GenerateResponse:
    """Plain LLM completion through the Ollama endpoint (no retrieval)."""
    messages = [{"role": "user", "content": request.prompt}]
    if request.system:
        messages.insert(0, {"role": "system", "content": request.system})
    text = await container.llm.chat(
        messages, model=request.model, temperature=request.temperature, max_tokens=request.max_tokens
    )
    return GenerateResponse(model=request.model or container.llm.model, response=text)
