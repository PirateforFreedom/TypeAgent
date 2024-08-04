import uuid
from functools import partial
from typing import List,Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from models.pydantic_models import (
    AgentStateModel,
    EmbeddingConfigModel,
    LLMConfigModel,
    PresetModel,
)
from server.rest_api.auth_token import get_current_user
from server.rest_api.interface import QueuingInterface
from server.server import SyncServer

router = APIRouter()


class ListAgentsResponse(BaseModel):
    num_agents: int = Field(..., description="The number of agents available to the user.")
    listAgentstate:List[AgentStateModel]= Field(..., description="The list agents  state available to the user.")
    # listpreset:List[PresetModel]= Field(..., description="The list preset  that relate to the agent available to the user.")#      also return - presets: List[PresetModel]
    # agents: List[dict] = Field(..., description="List of agent configurations.")
class ListtypeofAgentsResponse(BaseModel):
    typeofagents:List[str] = Field(..., description="List of type of agent.")


class CreateAgentRequest(BaseModel):
    # config: dict = Field(..., description="The agent configuration object.")
    type_agent:Optional[str]= Field( None, description="The agent type object.")
    name:Optional[str]= Field( None, description="The agent name object.")
    preset_id:Optional[uuid.UUID] = Field( None, description="Unique identifier for the preset.")
    persona_memory:Optional[str]= Field(None, description="The person memory type object.")
    human_memory:Optional[str]= Field(None, description="The human memory type object.")

class CreateAgentResponse(BaseModel):
    agent_state: AgentStateModel = Field(..., description="The state of the newly created agent.")
    # preset: PresetModel = Field(..., description="The preset that the agent was created from.")


def setup_agents_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents", tags=["agents"], response_model=ListAgentsResponse)
    def list_agents(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        List all agents associated with a given user.

        This endpoint retrieves a list of all agents and their configurations associated with the specified user ID.
        """
        interface.clear()
        #TODO add more information,now only has agentmodel,preset,later add source,recall memory,archive memory,knowledage base
        listagentstatemodel= server.list_agents(user_id=user_id)
        return ListAgentsResponse(
            listAgentstate=listagentstatemodel,
            num_agents=len(listagentstatemodel)
            # listpreset=listpresetmodel
            )
  
   
    @router.get("/agents/type", tags=["agents"], response_model=ListtypeofAgentsResponse)
    def list_typeofagents(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        List all agents associated with a given user.

        This endpoint retrieves a list of all agents and their configurations associated with the specified user ID.
        """
        interface.clear()
        agents_data = server.list_typeofagents()
        # print(agents_data)
        return ListtypeofAgentsResponse(typeofagents=[typekey for typekey in list(agents_data.keys())])
    @router.post("/agents", tags=["agents"], response_model=CreateAgentResponse)
    def create_agent(
        request: CreateAgentRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Create a new agent with the specified configuration.
        """
        interface.clear()
       
        # try:
        agent_state = server.create_agent(
            user_id=user_id,
            # **request.config
            # TODO turn into a pydantic model
            type_agent=request.type_agent if request.type_agent else None,
            name=request.name if request.name else None,
            preset_id=request.preset_id if request.preset_id and request.preset_id!=None else None,
            # persona_name=request.config["persona_name"] if "persona_name" in request.config else None,
            # human_name=request.config["human_name"] if "human_name" in request.config else None,
            persona_memory=request.persona_memory if request.persona_memory else None,
            human_memory=request.human_memory if request.human_memory else None,
            # llm_config=LLMConfigModel(
            # model=request.config['model'],
            # )
            # function_names=request.config["function_names"].split(",") if "function_names" in request.config else None,
        )
        llm_config = LLMConfigModel(**vars(agent_state.llm_config))
        embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

        # TODO when get_preset returns a PresetModel instead of Preset, we can remove this packing/unpacking line
        # preset= server.get_preset(preset_id=agent_state.preset_id, user_id=user_id)
        # recallmemory=server.get_preset(preset_id=agent_state.preset_id, user_id=user_id)
        # archivememory=server.get_preset(preset_id=agent_state.preset_id, user_id=user_id)

        # print(presetcreated)
        return CreateAgentResponse(
            agent_state=AgentStateModel(
                id=agent_state.id,
                name=agent_state.name,
                type_agent=agent_state.type_agent,
                user_id=agent_state.user_id,
                preset_id=agent_state.preset_id,
                persona_memory=agent_state.persona_memory,
                human_memory=agent_state.human_memory,
                llm_config=llm_config,
                embedding_config=embedding_config,
                state=agent_state.state,
                created_at=int(agent_state.created_at.timestamp()),
                user_status=agent_state.user_status,
                # functions_schema=agent_state.state["functions"],  # TODO: this is very error prone, jsut lookup the preset instead
            ),
            # preset=PresetModel(
            #     name=preset.name,
            #     id=preset.id,
            #     user_id=preset.user_id,
            #     # description=preset.description,
            #     created_at=preset.created_at,
            #     system=preset.system,
            #     persona=preset.persona,
            #     human=preset.human,
            #     functions_schema=preset.functions_schema,
            # ),

        )
        # except Exception as e:
        #    print(str(e))
        #    raise HTTPException(status_code=500, detail=str(e))

    return router
