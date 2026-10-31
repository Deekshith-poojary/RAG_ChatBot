"""FastAPI application factory."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_chatbot import __version__
from rag_chatbot.api.container import build_container
from rag_chatbot.api.routes import chat, documents, llm, system
from rag_chatbot.api.web import mount_web
from rag_chatbot.config import get_settings
from rag_chatbot.errors import DocumentBusyError, DocumentNotFoundError, OllamaError
from rag_chatbot.logging_config import configure_logging

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    container = await asyncio.to_thread(build_container, get_settings())
    app.state.container = container
    warm_up = asyncio.create_task(container.llm.warm_up())
    logger.info("API ready: open http://127.0.0.1:8000/")
    try:
        yield
    finally:
        warm_up.cancel()
        await container.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Chatbot API", version=__version__, lifespan=lifespan)

    for module in (system, documents, chat, llm):
        app.include_router(module.router, prefix=API_PREFIX)

    @app.exception_handler(DocumentNotFoundError)
    async def _not_found(_: Request, exc: DocumentNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Document not found: {exc}"})

    @app.exception_handler(DocumentBusyError)
    async def _busy(_: Request, exc: DocumentBusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(OllamaError)
    async def _ollama(_: Request, exc: OllamaError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    mount_web(app)
    return app


app = create_app()


def run() -> None:
    uvicorn.run("rag_chatbot.api.main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
