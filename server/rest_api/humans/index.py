import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends,Query,HTTPException
from pydantic import BaseModel, Field

from models.pydantic_models import HumanModel
from server.rest_api.auth_token import get_current_user
from server.rest_api.interface import QueuingInterface
from server.server import SyncServer

router = APIRouter()


class ListHumansResponse(BaseModel):
    humans: List[HumanModel] = Field(..., description="List of human configurations.")


class CreateHumanRequest(BaseModel):
    text: str = Field(..., description="The human text.")
    name: str = Field(..., description="The name of the human.")

class DeleteHumanResponse(BaseModel):
    message: str
    humanname_deleted: str
def setup_humans_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/humans", tags=["humans prompt"], response_model=ListHumansResponse)
    async def list_humans(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()
        humans = server.ms.list_humans(user_id=user_id)
        return ListHumansResponse(humans=humans)

    @router.post("/humans", tags=["humans prompt"], response_model=HumanModel)
    async def create_human(
        request: CreateHumanRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        new_human = HumanModel(text=request.text, name=request.name, user_id=user_id,user_status="on")
        human_id = new_human.id
        server.ms.add_human(new_human)
        return HumanModel(id=human_id, text=request.text, name=request.name, user_id=user_id,user_status="on")
    @router.delete("/humans/human_name", tags=["humans prompt"], response_model=DeleteHumanResponse)
    def delete_human(
        name: str = Query(..., description="The human nameto be deleted."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        try:
            token = server.ms.get_human(name=name, user_id=user_id)
            if token is None:
                raise HTTPException(status_code=404, detail=f"human name does not exist")
            server.ms.delete_human(name=name, user_id=user_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return DeleteHumanResponse(message="human name successfully deleted.", humanname_deleted=name)
    return router
    

  
