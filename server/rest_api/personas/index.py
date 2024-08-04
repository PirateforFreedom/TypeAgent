import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends,Query,HTTPException
from pydantic import BaseModel, Field

from models.pydantic_models import PersonaModel
from server.rest_api.auth_token import get_current_user
from server.rest_api.interface import QueuingInterface
from server.server import SyncServer

router = APIRouter()


class ListPersonasResponse(BaseModel):
    personas: List[PersonaModel] = Field(..., description="List of persona configurations.")


class CreatePersonaRequest(BaseModel):
    text: str = Field(..., description="The persona text.")
    name: str = Field(..., description="The name of the persona.")

class DeletePersonasResponse(BaseModel):
    message: str
    personasname_deleted: str
def setup_personas_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/personas", tags=["personas prompt"], response_model=ListPersonasResponse)
    async def list_personas(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()

        personas = server.ms.list_personas(user_id=user_id)
        return ListPersonasResponse(personas=personas)

    @router.post("/personas", tags=["personas prompt"], response_model=PersonaModel)
    async def create_persona(
        request: CreatePersonaRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        new_persona = PersonaModel(text=request.text, name=request.name, user_id=user_id,user_status="on")
        persona_id = new_persona.id
        server.ms.add_persona(new_persona)
        return PersonaModel(id=persona_id, text=request.text, name=request.name, user_id=user_id,user_status="on")
    @router.delete("/personas/personas_name", tags=["personas prompt"], response_model=DeletePersonasResponse)
    def delete_personas(
        name: str = Query(..., description="The personas nameto be deleted."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        try:
            token = server.ms.get_persona(name=name, user_id=user_id)
            if token is None:
                raise HTTPException(status_code=404, detail=f"personas name does not exist")
            server.ms.delete_persona(name=name, user_id=user_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return DeletePersonasResponse(message="personas name successfully deleted.", personasname_deleted=name)

    return router
