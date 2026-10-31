from typing import Annotated

from fastapi import Depends, Request

from rag_chatbot.api.container import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]
