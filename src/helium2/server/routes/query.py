from typing import Annotated

from fastapi import APIRouter, Depends

from helium2 import helium2
from helium2.runtime import HeliumServer
from helium2.runtime.protocol import HeliumRequest, HeliumResponse

router = APIRouter()


def get_started_server():
    yield helium2.get_started_instance()


@router.post("/request", tags=["request"])
async def handle_request(
    request: HeliumRequest, server: Annotated[HeliumServer, Depends(get_started_server)]
) -> HeliumResponse:
    return await server.request(request)
