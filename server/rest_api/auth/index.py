from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from server.rest_api.interface import QueuingInterface
from server.server import SyncServer
from log import get_logger
logger = get_logger(__name__)
router = APIRouter()
# logger = get_logger(__name__)

class AuthResponse(BaseModel):
    uuid: UUID = Field(..., description="UUID of the user")


class AuthRequest(BaseModel):
    password: str = Field(None, description="Admin password provided when starting the typeagent server")


def setup_auth_router(server: SyncServer, interface: QueuingInterface, password: str) -> APIRouter:
    @router.post("/auth", tags=["auth"], response_model=AuthResponse)
    def authenticate_user(request: AuthRequest) -> AuthResponse:
        """
        Authenticates the user and sends response with User related data.

        Currently, this is a placeholder that simply returns a UUID placeholder
        """
        interface.clear()
        try:
            if request.password != password:
                response = server.api_key_to_user(api_key=request.password)
                # raise HTTPException(status_code=400, detail="Incorrect credentials")
            else:
                response = server.authenticate_user()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return AuthResponse(uuid=response)

    return router
