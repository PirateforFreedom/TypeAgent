import uuid
from functools import partial
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from constants import DEFAULT_HUMAN, DEFAULT_PERSONA,DEFAULT_PRESET,DEFAULT_SYSTEMPROMPT
from data_types import Preset  # TODO remove
from models.pydantic_models import PresetModel,HumanModel, PersonaModel
from prompts import gpt_system
from server.rest_api.auth_token import get_current_user
from server.rest_api.interface import QueuingInterface
from server.server import SyncServer
from utils import get_human_text, get_persona_text

router = APIRouter()

"""
Implement the following functions:
* List all available presets
* Create a new preset
* Delete a preset
* TODO update a preset
"""


class ListPresetsResponse(BaseModel):
    presets: List[PresetModel] = Field(..., description="List of available presets.")


class CreatePresetsRequest(BaseModel):
    # TODO is there a cleaner way to create the request from the PresetModel (need to drop fields though)?
    name: str = Field(..., description="The name of the preset.")
    # id: Optional[Union[uuid.UUID, str]] = Field(default_factory=uuid.uuid4, description="The unique identifier of the preset.")
    id: Optional[str] = Field(None, description="The unique identifier of the preset.")
    # user_id: uuid.UUID = Field(..., description="The unique identifier of the user who created the preset.")
    # description: Optional[str] = Field(None, description="The description of the preset.")
    # created_at: datetime = Field(default_factory=get_utc_time, description="The unix timestamp of when the preset was created.")
    # system: str = Field(..., description="The system prompt of the preset.")
    # persona: str = Field(default=get_persona_text(DEFAULT_PERSONA), description="The persona of the preset.")
    # human: str = Field(default=get_human_text(DEFAULT_HUMAN), description="The human of the preset.")
    # system_name: Optional[str] = Field(None, description="The system prompt of the preset.")  # TODO: make optional and allow defaults
    # persona: Optional[str] = Field(default=None, description="The persona of the preset.")
    # human: Optional[str] = Field(default=None, description="The human of the preset.")
    functions_name: List[str] = Field(..., description="The functions schema of the preset.")
    # TODO
    persona_name: Optional[str] = Field(None, description="The name of the persona of the preset.")
    human_name: Optional[str] = Field(None, description="The name of the human of the preset.")
    system_name: Optional[str] = Field(None, description="The name of the system prompt of the preset.")

class CreatePresetResponse(BaseModel):
    preset: PresetModel = Field(..., description="The newly created preset.")


def setup_presets_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)
    @router.get("/presets/{preset_name}", tags=["presets"], response_model=PresetModel)
    async def get_preset(
        preset_name: str,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """Get a preset."""
        try:
            preset = server.get_preset(user_id=user_id, preset_name=preset_name)
            return preset
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.get("/presets", tags=["presets"], response_model=ListPresetsResponse)
    async def list_presets(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """List all presets created by a user."""
        # Clear the interface
        interface.clear()

        try:
            presets = server.list_presets(user_id=user_id)
            return ListPresetsResponse(presets=presets)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.post("/presets", tags=["presets"], response_model=CreatePresetResponse)
    async def create_preset(
        request: CreatePresetsRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """Create a preset."""
        try:
            if isinstance(request.id, str):
                request.id = uuid.UUID(request.id)
            

            # check if preset already exists
            # TODO: move this into a server function to create a preset
            if server.ms.get_preset(name=request.name, user_id=user_id):
                raise HTTPException(status_code=400, detail=f"Preset with name {request.name} already exists.")

            # For system/human/persona - if {system/human-personal}_name is None but the text is provied, then create a new data entry
            if request.system_name:
                # new system provided without name identity
                system_name =request.system_name
                systemp=server.ms.get_systemprompt(name=system_name,user_id=user_id)
                if systemp==None:
                    raise HTTPException(status_code=400, detail=f"system prompt with name {request.system_name} don't exists,please bulid system prompt of name ,try again")

                system =systemp.text
                # TODO: insert into system table
            else:
                system_name =DEFAULT_SYSTEMPROMPT
                systemp=server.ms.get_systemprompt(name=system_name,user_id=user_id)
                system =systemp.text

            if request.human_name:
                # new human provided without name identity
                human_name =request.human_name
                humamodel=server.ms.get_human(name=human_name,user_id=user_id)
                if humamodel==None:
                    raise HTTPException(status_code=400, detail=f"human with name {request.human_name} don't exists,please bulid human ,try again")

                human =humamodel.text
            else:
                human_name =DEFAULT_HUMAN
                humamodel=server.ms.get_human(name=human_name,user_id=user_id)
                human =humamodel.text

            if  request.persona_name:
                # new persona provided without name identity
                persona_name =request.persona_name
                personaodel=server.ms.get_persona(name=persona_name,user_id=user_id)
                if personaodel==None:
                    raise HTTPException(status_code=400, detail=f"persona with name {request.persona_name} don't exists,please bulid persona ,try again")

                persona =personaodel.text
                
            else:
                persona_name =DEFAULT_PERSONA
                personaodel=server.ms.get_persona(name=persona_name,user_id=user_id)
                persona =personaodel.text

            functions_schema=[]
            if len(request.functions_name)>=0:
                # functions_schema=[]
                # new persona provided without name identity
                functions_namelist =request.functions_name
                for functionsit in functions_namelist:
                     onetool=server.ms.get_tool(tool_name=functionsit,user_id=user_id)
                     if onetool==None:
                         continue
                        # raise HTTPException(status_code=400, detail=f"persona with name {request.persona_name} don't exists,please bulid persona ,try again")
                     functions_schema.append(onetool.json_schema)
                
            else:
                listtools=server.ms.list_tools(user_id=user_id)
                for listto in listtools:
                     functions_schema.append(listto.json_schema)
                    
                   



            # new_preset = PresetModel(
            new_preset = Preset(
                user_id=user_id,
                id=request.id if request.id else uuid.uuid4(),
                name=request.name,
                system=system,
                system_name=system_name,
                persona=persona,
                persona_name=persona_name,
                human=human,
                human_name=human_name,
                functions_schema=functions_schema,
                # persona_name=request.persona_name,
                # human_name=request.human_name,
                user_status="on"
            )
            preset = server.create_preset(preset=new_preset)

            # TODO remove once we migrate from Preset to PresetModel
            preset = PresetModel(**vars(preset))

            return CreatePresetResponse(preset=preset)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.delete("/presets/{preset_id}", tags=["presets"])
    async def delete_preset(
        preset_id: uuid.UUID,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """Delete a preset."""
        interface.clear()
        try:
            preset = server.delete_preset(user_id=user_id, preset_id=preset_id)
            return JSONResponse(
                status_code=status.HTTP_200_OK, content={"message": f"Preset preset_id={str(preset.id)} successfully deleted"}
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    return router
