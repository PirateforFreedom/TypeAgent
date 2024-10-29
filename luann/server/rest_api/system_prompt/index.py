import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends,Query,HTTPException
from pydantic import BaseModel, Field

from models.pydantic_models import SystemPromptModel
from server.rest_api.auth_token import get_current_user
from server.rest_api.interface import QueuingInterface
from server.server import SyncServer

router = APIRouter()


class ListSystempromptResponse(BaseModel):
    systemprompt: List[SystemPromptModel] = Field(..., description="List of system prompt configurations.")


class CreateSystempromptRequest(BaseModel):
    text: str = Field(..., description="The system prompt text.")
    name: str = Field(..., description="The name of the system prompt.")

class DeleteSystempromptResponse(BaseModel):
    message: str
    systemprompt_deleted: str
def setup_systemprompt_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/sysprompt", tags=["system prompt"], response_model=ListSystempromptResponse)
    async def list_sysprompt(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()
        systemprompt = server.ms.list_systemprompt(user_id=user_id)
        return ListSystempromptResponse(systemprompt=systemprompt)

    @router.post("/sysprompt", tags=["system prompt"], response_model=SystemPromptModel)
    async def create_sysprompt(
        request: CreateSystempromptRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        new_systemprompt = SystemPromptModel(text=request.text, name=request.name, user_id=user_id,user_status="on")
        systemprompt_id = new_systemprompt.id
        server.ms.add_systemprompt(new_systemprompt)
        return SystemPromptModel(id=systemprompt_id, text=request.text, name=request.name, user_id=user_id,user_status="on")
    @router.delete("/sysprompt/sysprompt_name", tags=["system prompt"], response_model=DeleteSystempromptResponse)
    def delete_sysprompt(
        name: str = Query(..., description="The system nameto be deleted."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        try:
            token = server.ms.get_systemprompt(name=name, user_id=user_id)
            if token is None:
                raise HTTPException(status_code=404, detail=f"system name does not exist")
            server.ms.delete_systemprompt(name=name, user_id=user_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return DeleteSystempromptResponse(message="system name successfully deleted.", systemprompt_deleted=name)
    return router
    

  
