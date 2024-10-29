import datetime
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union

import requests
from functions.functions import parse_source_code
from functions.schema_generator import generate_schema
from requests import HTTPError
from config import typeagentConfig
from constants import DEFAULT_PRESET,DEFAULT_SYSTEMPROMPT,DEFAULT_HUMAN,DEFAULT_PERSONA
from data_sources.connectors import DataConnector
from data_types import (
    AgentState,
    EmbeddingConfig,
    LLMConfig,
    Preset,
    Source,
    User,
)
from metadata import MetadataStore
from models.pydantic_models import (
    HumanModel,
    JobModel,
    JobStatus,
    PersonaModel,
    PresetModel,
    SourceModel,
    ToolModel,
    SystemPromptModel,
    LLMConfigModel,
    PresetModel,
    AgentStateModel,
    EmbeddingConfigModel,
)

# import pydantic response objects from server.rest_api
from server.rest_api.agents.command import CommandResponse
from server.rest_api.agents.config import GetAgentResponse
from server.rest_api.agents.index import CreateAgentResponse, ListAgentsResponse,CreateAgentRequest
from server.rest_api.agents.memory import (
    GetAgentArchivalMemoryResponse,
    GetAgentCoreMemoryResponse,
    InsertAgentArchivalMemoryResponse,
    UpdateAgentMemoryResponse,
    UpdateAgentMemoryRequest,
    GetAgentRecallMemoryResponse,
)
from server.rest_api.agents.message import (
    GetAgentMessagesResponse,
    UserMessageResponse,
)
from server.rest_api.config.index import ConfigResponse
from server.rest_api.humans.index import ListHumansResponse,DeleteHumanResponse
from server.rest_api.system_prompt.index import ListSystempromptResponse,DeleteSystempromptResponse
from server.rest_api.interface import QueuingInterface
from server.rest_api.models.index import ListModelsResponse
from server.rest_api.personas.index import ListPersonasResponse,DeletePersonasResponse
from server.rest_api.presets.index import (
    CreatePresetResponse,
    CreatePresetsRequest,
    ListPresetsResponse,
)
from server.rest_api.sources.index import ListSourcesResponse
from server.rest_api.tools.index import CreateToolResponse,ListToolsResponse
from server.server import SyncServer


def create_client(base_url: Optional[str] = None, token: Optional[str] = None,user_id:Optional[str] = None):
    if base_url is None and token is None:
        return LocalClient(user_id=user_id)
    else:
        return RESTClient(base_url, token)


class AbstractClient(object):
    def __init__(
        self,
        auto_save: bool = False,
        debug: bool = False,
    ):
        self.auto_save = auto_save
        self.debug = debug

    # agents

    def list_agents(self):
        """List all agents associated with a given user."""
        raise NotImplementedError

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        """Check if an agent with the specified ID or name exists."""
        raise NotImplementedError

    def create_agent(
        self,
        name: Optional[str] = None,
        preset: Optional[str] = None,
        persona: Optional[str] = None,
        human: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> AgentState:
        """Create a new agent with the specified configuration."""
        raise NotImplementedError

    def rename_agent(self, agent_id: uuid.UUID, new_name: str):
        """Rename the agent."""
        raise NotImplementedError

    def delete_agent(self, agent_id: uuid.UUID):
        """Delete the agent."""
        raise NotImplementedError

    def get_agent(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> AgentState:
        raise NotImplementedError

    # presets
    def create_preset(self, preset: Preset):
        raise NotImplementedError

    def delete_preset(self, preset_id: uuid.UUID):
        raise NotImplementedError

    def list_presets(self):
        raise NotImplementedError

    # memory

    def get_agent_memory(self, agent_id: str) -> Dict:
        raise NotImplementedError

    def update_agent_core_memory(self, agent_id: str, human: Optional[str] = None, persona: Optional[str] = None) -> Dict:
        raise NotImplementedError

    # agent interactions

    def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
        raise NotImplementedError

    def run_command(self, agent_id: str, command: str) -> Union[str, None]:
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    # archival memory

    def get_agent_archival_memory(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ):
        """Paginated get for the archival memory for an agent"""
        raise NotImplementedError

    def insert_archival_memory(self, agent_id: uuid.UUID, memory: str):
        """Insert archival memory into the agent."""
        raise NotImplementedError

    def delete_archival_memory(self, agent_id: uuid.UUID, memory_id: uuid.UUID):
        """Delete archival memory from the agent."""
        raise NotImplementedError

    # messages (recall memory)

    def get_messages(
        self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    ):
        """Get messages for the agent."""
        raise NotImplementedError

    def send_message(self, agent_id: uuid.UUID, message: str, role: str, stream: Optional[bool] = False):
        """Send a message to the agent."""
        raise NotImplementedError

    # humans / personas

    def list_humans(self):
        """List all humans."""
        raise NotImplementedError

    def create_human(self, name: str, human: str):
        """Create a human."""
        raise NotImplementedError

    def list_personas(self):
        """List all personas."""
        raise NotImplementedError

    def create_persona(self, name: str, persona: str):
        """Create a persona."""
        raise NotImplementedError

    # tools

    def list_tools(self):
        """List all tools."""
        raise NotImplementedError

    def create_tool(
        self, name: str, file_path: str, source_type: Optional[str] = "python", tags: Optional[List[str]] = None
    ) -> CreateToolResponse:
        """Create a tool."""
        raise NotImplementedError

    # data sources

    def list_sources(self):
        """List loaded sources"""
        raise NotImplementedError

    def delete_source(self):
        """Delete a source and associated data (including attached to agents)"""
        raise NotImplementedError

    def load_file_into_source(self, filename: str, source_id: uuid.UUID):
        """Load {filename} and insert into source"""
        raise NotImplementedError

    def create_source(self, name: str):
        """Create a new source"""
        raise NotImplementedError

    def attach_source_to_agent(self, source_id: uuid.UUID, agent_id: uuid.UUID):
        """Attach a source to an agent"""
        raise NotImplementedError

    def detach_source(self, source_id: uuid.UUID, agent_id: uuid.UUID):
        """Detach a source from an agent"""
        raise NotImplementedError

    # server configuration commands

    def list_models(self):
        """List all models."""
        raise NotImplementedError

    def get_config(self):
        """Get server config"""
        raise NotImplementedError


class RESTClient(AbstractClient):
    def __init__(
        self,
        base_url: str,
        token: str,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.base_url = base_url
        self.headers = {"accept": "application/json", "authorization": f"Bearer {token}"}
        self.token = token

    # agents

    def list_agents(self):
        response = requests.get(f"{self.base_url}/api/agents", headers=self.headers)
        return ListAgentsResponse(**response.json())

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        response = requests.get(f"{self.base_url}/api/agents/{str(agent_id)}/config", headers=self.headers)
        # print(response.text, response.status_code)
        # print(response)
        if response.status_code == 404:
            # not found error
            return False
        elif response.status_code == 200:
            return True
        else:
            raise ValueError(f"Failed to check if agent exists: {response.text}")

    def create_agent(
        self,
        name: Optional[str] = None,
        type_agent: Optional[str] = None,
        preset_id: Optional[str] = None,
        persona_memory: Optional[str] = None, # TODO: this should actually be re-named preset_name
        human_memory: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> AgentState:
        if embedding_config or llm_config:
            raise ValueError("Cannot override embedding_config or llm_config when creating agent via REST API")
        payload = CreateAgentRequest(
            name=name,
            type_agent=type_agent,
            preset_id=preset_id,
            persona_memory=persona_memory,
            human_memory=human_memory,
        )
        

       
        response = requests.post(f"{self.base_url}/api/agents", json=payload.model_dump(), headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Status {response.status_code} - Failed to create agent: {response.text}")
        response_obj = CreateAgentResponse(**response.json())
        return self.get_agent_response_to_state(response_obj)

    def get_agent_response_to_state(self, response: Union[GetAgentResponse, CreateAgentResponse]) -> AgentState:
        # TODO: eventually remove this conversion
        # llm_config = LLMConfig(
        #     model=response.agent_state.llm_config.model,
        #     model_endpoint_type=response.agent_state.llm_config.model_endpoint_type,
        #     model_endpoint=response.agent_state.llm_config.model_endpoint,
        #     model_wrapper=response.agent_state.llm_config.model_wrapper,
        #     context_window=response.agent_state.llm_config.context_window,
        # )
        # embedding_config = EmbeddingConfig(
        #     embedding_endpoint_type=response.agent_state.embedding_config.embedding_endpoint_type,
        #     embedding_endpoint=response.agent_state.embedding_config.embedding_endpoint,
        #     embedding_model=response.agent_state.embedding_config.embedding_model,
        #     embedding_dim=response.agent_state.embedding_config.embedding_dim,
        #     embedding_chunk_size=response.agent_state.embedding_config.embedding_chunk_size,
        # )
        agent_state = AgentState(
            id=response.agent_state.id,
            name=response.agent_state.name,
            type_agent=response.agent_state.type_agent,
            user_id=response.agent_state.user_id,
            preset_id=response.agent_state.preset_id,
            persona_memory=response.agent_state.persona_memory,
            human_memory=response.agent_state.human_memory,
            llm_config=response.agent_state.llm_config,
            embedding_config=response.agent_state.embedding_config,
            state=response.agent_state.state,
            # load datetime from timestampe
            created_at=datetime.datetime.fromtimestamp(response.agent_state.created_at, tz=datetime.timezone.utc),
        )
        return agent_state

    def rename_agent(self, agent_id: uuid.UUID, new_name: str):
        response = requests.patch(f"{self.base_url}/api/agents/{str(agent_id)}/rename", json={"agent_name": new_name}, headers=self.headers)
        assert response.status_code == 200, f"Failed to rename agent: {response.text}"
        response_obj = GetAgentResponse(**response.json())
        return self.get_agent_response_to_state(response_obj)

    def delete_agent(self, agent_id: uuid.UUID):
        """Delete the agent."""
        response = requests.delete(f"{self.base_url}/api/agents/{str(agent_id)}", headers=self.headers)
        # assert response.status_code == 200, f"Failed to delete agent: {response.text}"
        if response.status_code != 200:
            raise HTTPError(response.json())
        return response.json()

    def get_agent(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> AgentState:
        response = requests.get(f"{self.base_url}/api/agents/{str(agent_id)}/config", headers=self.headers)
        assert response.status_code == 200, f"Failed to get agent: {response.text}"
        response_obj = GetAgentResponse(**response.json())
        return self.get_agent_response_to_state(response_obj)

    
    def get_preset(self, name: str) -> PresetModel:
        response = requests.get(f"{self.base_url}/api/presets/{name}", headers=self.headers)
        assert response.status_code == 200, f"Failed to get preset: {response.text}"
        return PresetModel(**response.json())

    def create_preset(
        self,
        name: str,
        functions_name: List[str],
        id:Optional[str]=None,
        system_name: Optional[str] = None,
        persona_name: Optional[str] = None,
        human_name: Optional[str] = None,
    ) -> PresetModel:
        """Create an agent preset
        :param name: Name of the preset
        :type name: str
        :param system: System prompt (text)
        :type system: str
        :param persona: Persona prompt (text)
        :type persona: Optional[str]
        :param human: Human prompt (text)
        :type human: Optional[str]
        :param tools: List of tools to connect, defaults to None
        :type tools: Optional[List[Tool]], optional
        :param default_tools: Whether to automatically include default tools, defaults to True
        :type default_tools: bool, optional
        :return: Preset object
        :rtype: PresetModel
        """
        # provided tools
        # schema = []
        # if tools:
        #     for tool in tools:
        #         print("CUSOTM TOOL", tool.json_schema)
        #         schema.append(tool.json_schema)

        # # include default tools
        # default_preset = self.get_preset(name=DEFAULT_PRESET)
        # if default_tools:
        #     # TODO
        #     # from memgpt.functions.functions import load_function_set
        #     # load_function_set()
        #     # return
        #     for function in default_preset.functions_schema:
        #         schema.append(function)

        payload = CreatePresetsRequest(
            name=name,
            id=id,
            system_name=system_name,
            persona_name=persona_name,
            human_name=human_name,
            functions_name=functions_name,
        )
        # print(schema)
        # print(human_name, persona_name, system_name, name)
        # print(payload.model_dump())
        response = requests.post(f"{self.base_url}/api/presets", json=payload.model_dump(), headers=self.headers)
        assert response.status_code == 200, f"Failed to create preset: {response.text}"
        return CreatePresetResponse(**response.json())
        # return CreatePresetResponse(**response.json()).preset
    def delete_preset(self, preset_id: uuid.UUID):
        response = requests.delete(f"{self.base_url}/api/presets/{str(preset_id)}", headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return response.json()
        # assert response.status_code == 200, f"Failed to delete preset: {response.text}"

    def list_presets(self) -> List[PresetModel]:
        response = requests.get(f"{self.base_url}/api/presets", headers=self.headers)
        return ListPresetsResponse(**response.json()).presets

    # memory
    
    def get_agent_corememory(self, agent_id: uuid.UUID) -> GetAgentCoreMemoryResponse:
        response = requests.get(f"{self.base_url}/api/agents/{agent_id}/corememory", headers=self.headers)
        return GetAgentCoreMemoryResponse(**response.json())

    def update_agent_core_memory(self, agent_id: str, humman_memory:str,persona_memory:str) -> UpdateAgentMemoryResponse:

        new_memory_contents =UpdateAgentMemoryRequest(
            human=humman_memory,
            persona=persona_memory,)

      
        response = requests.post(f"{self.base_url}/api/agents/{agent_id}/corememory", json=new_memory_contents.model_dump(), headers=self.headers)
        return UpdateAgentMemoryResponse(**response.json())

    # agent interactions

    def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
        return self.send_message(agent_id, message, role="user")

    def run_command(self, agent_id: str, command: str) -> Union[str, None]:
        response = requests.post(f"{self.base_url}/api/agents/{str(agent_id)}/command", json={"command": command}, headers=self.headers)
        return CommandResponse(**response.json())

    # def save(self):
    #     raise NotImplementedError

    # archival memory

    def get_agent_archival_memory(
        self, agent_id: uuid.UUID,
    ):
        """Paginated get for the archival memory for an agent"""
      
        
        response = requests.get(f"{self.base_url}/api/agents/{str(agent_id)}/archival/all", headers=self.headers)
        assert response.status_code == 200, f"Failed to get archival memory: {response.text}"
        return GetAgentArchivalMemoryResponse(**response.json())

    def insert_archival_memory(self, agent_id: uuid.UUID, memory: str) -> GetAgentArchivalMemoryResponse:
        response = requests.post(f"{self.base_url}/api/agents/{agent_id}/archival", json={"content": memory}, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to insert archival memory: {response.text}")
        # print(response.json())
        return InsertAgentArchivalMemoryResponse(**response.json())

    def delete_archival_memory(self, agent_id: uuid.UUID, memory_id: uuid.UUID):
        response = requests.delete(f"{self.base_url}/api/agents/{agent_id}/archival?id={memory_id}", headers=self.headers)
        assert response.status_code == 200, f"Failed to delete archival memory: {response.text}"

        if response.status_code != 200:
            raise HTTPError(response.json())
        return response.json()

    # messages (recall memory)
    def get_agent_recallmemory(self, agent_id: uuid.UUID) -> GetAgentRecallMemoryResponse:
        response = requests.get(f"{self.base_url}/api/agents/{agent_id}/recallmemory", headers=self.headers)
        return GetAgentRecallMemoryResponse(**response.json())
    def get_messages(
        self, agent_id: uuid.UUID, start: Optional[int] = None, count: Optional[int] = None
    ) -> GetAgentMessagesResponse:
        params = {"start": start, "count": count}
        response = requests.get(f"{self.base_url}/api/agents/{agent_id}/messages", params=params, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to get messages: {response.text}")
        return GetAgentMessagesResponse(**response.json())

    def send_message(self, agent_id: uuid.UUID, message: str, role: str, stream: Optional[bool] = False) -> UserMessageResponse:
        data = {"message": message, "role": role, "stream": stream}
        response = requests.post(f"{self.base_url}/api/agents/{agent_id}/messages", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to send message: {response.text}")
        return UserMessageResponse(**response.json())

    
    
    
    
    # humans / personas/ system prompt
    

    def list_systemprompt(self) -> ListSystempromptResponse:
        response = requests.get(f"{self.base_url}/api/sysprompt", headers=self.headers)
        return ListSystempromptResponse(**response.json())

    def create_systemprompt(self, name: str, systemprompt: str) -> SystemPromptModel:
        data = {"name": name, "text": systemprompt}
        response = requests.post(f"{self.base_url}/api/sysprompt", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create human: {response.text}")

        # print(response.json())
        return SystemPromptModel(**response.json())
    def delete_systemprompt(self, name:str):
        params = {"name": str(name)}
        response = requests.delete(f"{self.base_url}/api/sysprompt/sysprompt_name", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return DeleteSystempromptResponse(**response.json())

    def list_humans(self) -> ListHumansResponse:
        response = requests.get(f"{self.base_url}/api/humans", headers=self.headers)
        return ListHumansResponse(**response.json())

    def create_human(self, name: str, human: str) -> HumanModel:
        data = {"name": name, "text": human}
        response = requests.post(f"{self.base_url}/api/humans", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create human: {response.text}")

        # print(response.json())
        return HumanModel(**response.json())
    def delete_human(self, name:str):
        params = {"name": str(name)}
        response = requests.delete(f"{self.base_url}/api/humans/human_name", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return DeleteHumanResponse(**response.json())

    def delete_personas(self, name:str):
        params = {"name": str(name)}
        response = requests.delete(f"{self.base_url}/api/personas/personas_name", params=params, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError(response.json())
        return DeletePersonasResponse(**response.json())
    def list_personas(self) -> ListPersonasResponse:
        response = requests.get(f"{self.base_url}/api/personas", headers=self.headers)
        return ListPersonasResponse(**response.json())

    def create_persona(self, name: str, persona: str) -> PersonaModel:
        data = {"name": name, "text": persona}
        response = requests.post(f"{self.base_url}/api/personas", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create persona: {response.text}")
        # print(response.json())
        return PersonaModel(**response.json())

    # sources

    def list_sources(self):
        """List loaded sources"""
        response = requests.get(f"{self.base_url}/api/sources", headers=self.headers)
        response_json = response.json()
        return ListSourcesResponse(**response_json)

    def delete_source(self, source_id: uuid.UUID):
        """Delete a source and associated data (including attached to agents)"""
        response = requests.delete(f"{self.base_url}/api/sources/{str(source_id)}", headers=self.headers)
        assert response.status_code == 200, f"Failed to delete source: {response.text}"
        return response.json()

    def get_job_status(self, job_id: uuid.UUID):
        response = requests.get(f"{self.base_url}/api/sources/status/{str(job_id)}", headers=self.headers)

        if response.status_code != 404:
             return JobModel(**response.json())
        else:
            return None

    def load_file_into_source(self, filename: str, source_id: uuid.UUID, blocking=True):
        """Load {filename} and insert into source"""
        files = {"file": open(filename, "rb")}

        # create job
        response = requests.post(f"{self.base_url}/api/sources/{source_id}/upload", files=files, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to upload file to source: {response.text}")

        job = JobModel(**response.json())
        if blocking:
            # wait until job is completed
            while True:
                job = self.get_job_status(job.id)
                if job.status == JobStatus.completed:
                    break
                elif job.status == JobStatus.failed:
                    raise ValueError(f"Job failed: {job.metadata}")
                time.sleep(1)
        return job

    def create_source(self, name: str) -> SourceModel:
        """Create a new source"""
        payload = {"name": name}
        response = requests.post(f"{self.base_url}/api/sources", json=payload, headers=self.headers)
        response_json = response.json()
        response_obj = SourceModel(**response_json)
        return response_obj

    def attach_source_to_agent(self, source_id: uuid.UUID, agent_id: uuid.UUID):
        """Attach a source to an agent"""
        params = {"agent_id": agent_id}
        response = requests.post(f"{self.base_url}/api/sources/{source_id}/attach", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to attach source to agent: {response.text}"
        return response.json()

    def detach_source_from_agent(self, source_id: uuid.UUID, agent_id: uuid.UUID):
        """Detach a source from an agent"""
        params = {"agent_id": str(agent_id)}
        response = requests.post(f"{self.base_url}/api/sources/{source_id}/detach", params=params, headers=self.headers)
        assert response.status_code == 200, f"Failed to detach source from agent: {response.text}"
        return response.json()

    # server configuration commands

    def list_models(self) -> ListModelsResponse:
        response = requests.get(f"{self.base_url}/api/models", headers=self.headers)
        return ListModelsResponse(**response.json())

    def get_config(self) -> ConfigResponse:
        response = requests.get(f"{self.base_url}/api/config", headers=self.headers)
        return ConfigResponse(**response.json())


 # tools (currently only available for admin)
    def create_tool(self, name: str, file_path: str, source_type: Optional[str] = "python", tags: Optional[List[str]] = None) -> ToolModel:
        """Add a tool implemented in a file path"""
        source_code = open(file_path, "r").read()
        data = {"name": name, "source_code": source_code, "source_type": source_type, "tags": tags}
        response = requests.post(f"{self.base_url}/api/tools", json=data, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to create tool: {response.text}")
        return ToolModel(**response.json())

    def list_tools(self) -> ListToolsResponse:
        response = requests.get(f"{self.base_url}/api/tools", headers=self.headers)
        return ListToolsResponse(**response.json())

    def delete_tool(self, name: str):
        response = requests.delete(f"{self.base_url}/api/tools/{name}", headers=self.headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to delete tool: {response.text}")
        return response.json()

    def get_tool(self, name: str):
        response = requests.get(f"{self.base_url}/api/tools/{name}", headers=self.headers)
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise ValueError(f"Failed to get tool: {response.text}")
        return ToolModel(**response.json())
    # tools

    # def create_tool(
    #     self, name: str, file_path: str, source_type: Optional[str] = "python", tags: Optional[List[str]] = None
    # ) -> CreateToolResponse:
    #     """Add a tool implemented in a file path"""
    #     source_code = open(file_path, "r").read()
    #     data = {"name": name, "source_code": source_code, "source_type": source_type, "tags": tags}
    #     response = requests.post(f"{self.base_url}/api/tools", json=data, headers=self.headers)
    #     if response.status_code != 200:
    #         raise ValueError(f"Failed to create tool: {response.text}")
    #     return CreateToolResponse(**response.json())

    # def list_tools(self) -> ListToolsResponse:
    #     response = requests.get(f"{self.base_url}/api/tools", headers=self.headers)
    #     return ListToolsResponse(**response.json())


class LocalClient(AbstractClient):#TODO refactor code ,all the code and logic must be modified later ,because these code  and function don't work out
    def __init__(
        self,
        auto_save: bool = False,
        user_id: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initializes a new instance of Client class.
        :param auto_save: indicates whether to automatically save after every message.
        :param quickstart: allows running quickstart on client init.
        :param config: optional config settings to apply after quickstart
        :param debug: indicates whether to display debug messages.
        """
        self.auto_save = auto_save

        # determine user_id (pulled from local config)
        config = typeagentConfig.load()
        if user_id:
            self.user_id = uuid.UUID(user_id)
        else:
            self.user_id = uuid.UUID(config.anon_clientid)

        # create user if does not exist
        ms = MetadataStore(config)
        self.user = User(id=self.user_id)
        if ms.get_user(self.user_id):
            # update user
            ms.update_user(self.user)
        else:
            ms.create_user(self.user)

        # create preset records in metadata store
        # from presets.presets import add_default_presets

        # add_default_presets(self.user_id, ms)

        self.interface = QueuingInterface(debug=debug)
        # self.server = SyncServer(default_interface=self.interface)
        self.server = SyncServer(default_interface_factory=lambda: self.interface)
    



    # agent
    def list_agents(self):
        self.interface.clear()
        return self.server.list_agents(user_id=self.user_id)

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        if not (agent_id or agent_name):
            raise ValueError(f"Either agent_id or agent_name must be provided")
        if agent_id and agent_name:
            raise ValueError(f"Only one of agent_id or agent_name can be provided")
        existingagents = self.list_agents()
        # print(existingagents)
        if agent_id:
            return  agent_id in [str(agent.id) for agent in existingagents]
        else:
            return agent_name in [agent.name for agent in existingagents]

    def create_agent(
        self,
        name: Optional[str] = None,
        type_agent: Optional[str] = None,
        preset_id: Optional[str] = None,
        persona_memory: Optional[str] = None, # TODO: this should actually be re-named preset_name
        human_memory: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> CreateAgentResponse:
        if name and self.agent_exists(agent_name=name):
            raise ValueError(f"Agent with name {name} already exists (user_id={self.user_id})")

        self.interface.clear()
        agent_state = self.server.create_agent(
            user_id=self.user_id,
            # **request.config
            # TODO turn into a pydantic model
            type_agent=type_agent if type_agent else None,
            name=name if name else None,
            preset_id=preset_id if preset_id and preset_id!=None else None,
            # persona_name=request.config["persona_name"] if "persona_name" in request.config else None,
            # human_name=request.config["human_name"] if "human_name" in request.config else None,
            persona_memory=persona_memory if persona_memory else None,
            human_memory=human_memory if human_memory else None,
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
            

        )

    def rename_agent(self, agent_id: uuid.UUID, new_name: str):
        self.interface.clear()
        try:
            agent_state = self.server.rename_agent(user_id=self.user_id, agent_id=agent_id, new_agent_name=new_name)
            # get sources
            # attached_sources = server.list_attached_sources(agent_id=agent_id)
        except Exception as e:
            raise ValueError("wrong name for update agent name")
        llm_config = LLMConfigModel(**vars(agent_state.llm_config))
        embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

        return GetAgentResponse(
            agent_state=AgentStateModel(
                id=agent_state.id,
                name=agent_state.name,
                user_id=agent_state.user_id,
                type_agent=agent_state.type_agent,
                preset_id=agent_state.preset_id,
                persona_memory=agent_state.persona_memory,
                human_memory=agent_state.human_memory,
                llm_config=llm_config,
                embedding_config=embedding_config,
                state=agent_state.state,
                created_at=int(agent_state.created_at.timestamp()),
                functions_schema=agent_state.state["functions"],  # TODO: this is very error prone, jsut lookup the preset instead
                user_status=agent_state.user_status,
            ),
            # last_run_at=None,  # TODO
            # sources=attached_sources,
        )
    def delete_agent(self, agent_id: uuid.UUID):
        """Delete the agent."""
        self.interface.clear()
        try:
            self.server.delete_agent(user_id=self.user_id, agent_id=agent_id)
            return f"Agent agent_id={agent_id} successfully deleted"
        except Exception as e:
            raise ValueError("delete agent is wrong")
    
    
    
    def get_agent(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> GetAgentResponse:
        self.interface.clear()
        

        agent_state = self.server.get_agent_config(user_id=self.user_id, agent_id=agent_id)
        # get sources
        # attached_sources = server.list_attached_sources(agent_id=agent_id)

        # configs
        llm_config = LLMConfigModel(**vars(agent_state.llm_config))
        embedding_config = EmbeddingConfigModel(**vars(agent_state.embedding_config))

        return GetAgentResponse(
            agent_state=AgentStateModel(
                id=agent_state.id,
                name=agent_state.name,
                user_id=agent_state.user_id,
                type_agent=agent_state.type_agent,
                preset_id=agent_state.preset_id,
                persona_memory=agent_state.persona_memory,
                human_memory=agent_state.human_memory,
                llm_config=llm_config,
                embedding_config=embedding_config,
                state=agent_state.state,
                created_at=int(agent_state.created_at.timestamp()),
                functions_schema=agent_state.state["functions"],  # TODO: this is very error prone, jsut lookup the preset instead
                user_status=agent_state.user_status,
            ),
            # last_run_at=None,  # TODO
            # sources=attached_sources,
        )
    
    #preset
    def create_preset(self,
        name: str,
        functions_name: List[str],
        id:Optional[str]=None,
        system_name: Optional[str] = None,
        persona_name: Optional[str] = None,
        human_name: Optional[str] = None,
        ) -> PresetModel:
        try:
            if isinstance(id, str):
                id = uuid.UUID(id)
            

            # check if preset already exists
            # TODO: move this into a server function to create a preset
            if self.server.ms.get_preset(name=name, user_id=self.user_id):
                raise ValueError(f"Preset with name {name} already exists.")

            # For system/human/persona - if {system/human-personal}_name is None but the text is provied, then create a new data entry
            if system_name:
                # new system provided without name identity
                system_name =system_name
                systemp=self.server.ms.get_systemprompt(name=system_name,user_id=self.user_id)
                if systemp==None:
                    raise ValueError(f"sytem prompt is none")

                system =systemp.text
                # TODO: insert into system table
            else:
                system_name =DEFAULT_SYSTEMPROMPT
                systemp=self.server.ms.get_systemprompt(name=system_name,user_id=self.user_id)
                system =systemp.text

            if human_name:
                # new human provided without name identity
                human_name =human_name
                humamodel=self.server.ms.get_human(name=human_name,user_id=self.user_id)
                if humamodel==None:
                    raise ValueError(f"human with name {human_name} don't exists,please bulid human ,try again")

                human =humamodel.text
            else:
                human_name =DEFAULT_HUMAN
                humamodel=self.server.ms.get_human(name=human_name,user_id=self.user_id)
                human =humamodel.text

            if  persona_name:
                # new persona provided without name identity
                persona_name =persona_name
                personaodel=self.server.ms.get_persona(name=persona_name,user_id=self.user_id)
                if personaodel==None:
                    raise ValueError(f"persona with name {persona_name} don't exists,please bulid persona ,try again")

                persona =personaodel.text
                
            else:
                persona_name =DEFAULT_PERSONA
                personaodel=self.server.ms.get_persona(name=persona_name,user_id=self.user_id)
                persona =personaodel.text

            functions_schema=[]
            if len(functions_name)>=0:
                # functions_schema=[]
                # new persona provided without name identity
                functions_namelist =functions_name
                for functionsit in functions_namelist:
                     onetool=self.server.ms.get_tool(tool_name=functionsit,user_id=self.user_id)
                     if onetool==None:
                         continue
                        # raise HTTPException(status_code=400, detail=f"persona with name {request.persona_name} don't exists,please bulid persona ,try again")
                     functions_schema.append(onetool.json_schema)
                
            else:
                listtools=self.server.ms.list_tools(user_id=self.user_id)
                for listto in listtools:
                     functions_schema.append(listto.json_schema)
                    
                   



            # new_preset = PresetModel(
            new_preset = Preset(
                user_id=self.user_id,
                id=id if id else uuid.uuid4(),
                name=name,
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
            preset = self.server.create_preset(preset=new_preset)

            # TODO remove once we migrate from Preset to PresetModel
            preset = PresetModel(**vars(preset))

            return CreatePresetResponse(preset=preset)
        except Exception as e:
            raise ValueError(f"create preset is wrong,please check it ")
       
    def delete_preset(self, preset_id: uuid.UUID):
         self.interface.clear()
       
         preset = self.server.delete_preset(user_id=self.user_id, preset_id=preset_id)
         return "sucessfully delete preset" 

    def list_presets(self) -> List[PresetModel]:
        return self.server.list_presets(user_id=self.user_id)
    
    def get_preset(self, name: str) -> PresetModel:
        preset = self.server.get_preset(user_id=self.user_id, preset_name=self.preset_name)
        return preset

    
    # humans / personas
    def list_humans(self, user_id: uuid.UUID):
        return self.server.list_humans(user_id=user_id if user_id else self.user_id)
    def get_human(self, name: str, user_id: uuid.UUID):
        return self.server.ms.get_human(name=name, user_id=user_id)
    def add_human(self, human: HumanModel):
        return self.server.ms.add_human(human=human)
    def update_human(self, human: HumanModel):
        return self.server.ms.update_human(human=human)
    def delete_human(self, name: str, user_id: uuid.UUID):
        return self.server.ms.delete_human(name, user_id)
    # tools
    # def create_tool(
    #     self,
    #     func,
    #     name: Optional[str] = None,
    #     update: Optional[bool] = True,  # TODO: actually use this
    #     tags: Optional[List[str]] = None,
    # ):
    #     """
    #     Create a tool.
    #     Args:
    #         func (callable): The function to create a tool for.
    #         tags (Optional[List[str]], optional): Tags for the tool. Defaults to None.
    #         update (bool, optional): Update the tool if it already exists. Defaults to True.
    #     Returns:
    #         tool (ToolModel): The created tool.
    #     """
    #     # TODO: check if tool already exists
    #     # TODO: how to load modules?
    #     # parse source code/schema
    #     source_code = parse_source_code(func)
    #     json_schema = generate_schema(func, name)
    #     source_type = "python"
    #     tool_name = json_schema["name"]

    #     # check if already exists:
    #     existing_tool = self.server.ms.get_tool(tool_name)
    #     existing_tool = self.server.ms.get_tool(tool_name, self.user_id)
    #     if existing_tool:
    #         if update:
    #             # update existing tool
    #             existing_tool.source_code = source_code
    #             existing_tool.source_type = source_type
    #             existing_tool.tags = tags
    #             existing_tool.json_schema = json_schema
    #             self.server.ms.update_tool(existing_tool)
    #             return self.server.ms.get_tool(tool_name)
    #             return self.server.ms.get_tool(tool_name, self.user_id)
    #         else:
    #             raise ValueError(f"Tool {name} already exists and update=False")

    #     tool = ToolModel(name=tool_name, source_code=source_code, source_type=source_type, tags=tags, json_schema=json_schema)
    #     tool = ToolModel(
    #         name=tool_name, source_code=source_code, source_type=source_type, tags=tags, json_schema=json_schema, user_id=self.user_id
    #     )
    #     self.server.ms.add_tool(tool)
    #     return self.server.ms.get_tool(tool_name)
    #     return self.server.ms.get_tool(tool_name, self.user_id)
    def create_tool(
        self,
        func,
        name: Optional[str] = None,
        update: Optional[bool] = True,  # TODO: actually use this
        tags: Optional[List[str]] = None,
    ):
        """
        Create a tool.
        Args:
            func (callable): The function to create a tool for.
            tags (Optional[List[str]], optional): Tags for the tool. Defaults to None.
            update (bool, optional): Update the tool if it already exists. Defaults to True.
        Returns:
            tool (ToolModel): The created tool.
        """

        # TODO: check if tool already exists
        # TODO: how to load modules?
        # parse source code/schema
        source_code = parse_source_code(func)
        json_schema = generate_schema(func, name)
        source_type = "python"
        tool_name = json_schema["name"]

        # check if already exists:
        existing_tool = self.server.ms.get_tool(tool_name=tool_name,user_id=self.user_id)
        if existing_tool:
            if update:
                # update existing tool
                existing_tool.source_code = source_code
                existing_tool.source_type = source_type
                existing_tool.tags = tags
                existing_tool.json_schema = json_schema
                self.server.ms.update_tool(existing_tool)
                return self.server.ms.get_tool(tool_name)
            else:
                raise ValueError(f"Tool {name} already exists and update=False")

        tool = ToolModel(name=tool_name, source_code=source_code, source_type=source_type, tags=tags, json_schema=json_schema,user_id=self.user_id,user_status="on")
        self.server.ms.add_tool(tool)
        return self.server.ms.get_tool(tool_name)
    def list_tools(self):
        """List available tools.
        Returns:
            tools (List[ToolModel]): A list of available tools.
        """
        # return self.server.ms.list_tools()
        return self.server.ms.list_tools(user_id=self.user_id)

    def get_tool(self, name: str):
        return self.server.ms.get_tool(name, user_id=self.user_id)

    def delete_tool(self, name: str):
        return self.server.ms.delete_tool(name, user_id=self.user_id)
   # core memory
    def get_agent_corememory(self, agent_id: uuid.UUID) :
        self.interface.clear()
        return self.server.get_agent_corememory(user_id=self.user_id, agent_id=agent_id)

    def update_agent_core_memory(self, agent_id: uuid.UUID, new_memory_contents: Dict):
        self.interface.clear()
        return self.server.update_agent_core_memory(user_id=self.user_id, agent_id=agent_id, new_memory_contents=new_memory_contents)
   # archival memory


    # recall memory

    # agent interactions
    # messages
    def send_message(self, agent_id: uuid.UUID, message: str, role: str, stream: Optional[bool] = False) -> UserMessageResponse:
        if stream:
            # TODO: implement streaming with stream=True/False
            raise NotImplementedError
        self.interface.clear()
        # self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        usage = self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        if self.auto_save:
            self.save()
        else:
            return UserMessageResponse(messages=self.interface.to_list(), usage=usage)
    def user_message(self, agent_id: uuid.UUID, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
        self.interface.clear()
        # self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        # usage = self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        usage = self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        if self.auto_save:
            self.save()
        else:
            # return self.interface.to_list()
            # return UserMessageResponse(messages=self.interface.to_list())
            return UserMessageResponse(messages=self.interface.to_list(), usage=usage)

    def run_command(self, agent_id: uuid.UUID, command: str) -> Union[str, None]:
        self.interface.clear()
        return self.server.run_command(user_id=self.user_id, agent_id=agent_id, command=command)

   
      # data sources
    def load_data(self, connector: DataConnector, source_name: str):
        self.server.load_data(user_id=self.user_id, connector=connector, source_name=source_name)

    def create_source(self, name: str):
        self.server.create_source(user_id=self.user_id, name=name)

    def attach_source_to_agent(self, source_id: uuid.UUID, agent_id: uuid.UUID):
        self.server.attach_source_to_agent(user_id=self.user_id, source_id=source_id, agent_id=agent_id)

    # def delete_agent(self, agent_id: uuid.UUID):
    #     self.server.delete_agent(user_id=self.user_id, agent_id=agent_id)

    # def get_agent_archival_memory(
    # def list_agents(self):
    #     self.interface.clear()
    #     return self.server.list_agents(user_id=self.user_id)

    # def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
    #     if not (agent_id or agent_name):
    #         raise ValueError(f"Either agent_id or agent_name must be provided")
    #     if agent_id and agent_name:
    #         raise ValueError(f"Only one of agent_id or agent_name can be provided")
    #     existing = self.list_agents()
    #     if agent_id:
    #         return agent_id in [agent["id"] for agent in existing["agents"]]
    #     else:
    #         return agent_name in [agent["name"] for agent in existing["agents"]]

    # def create_agent(
    #     self,
    #     name: Optional[str] = None,
    #     preset: Optional[str] = None,
    #     persona: Optional[str] = None,
    #     human: Optional[str] = None,
    #     type_agent: Optional[str] = None,
    # ) -> AgentState:
    #     if name and self.agent_exists(agent_name=name):
    #         raise ValueError(f"Agent with name {name} already exists (user_id={self.user_id})")

    #     self.interface.clear()
    #     agent_state = self.server.create_agent(
    #         user_id=self.user_id,
    #         name=name,
    #         preset=preset,
    #         persona=persona,
    #         human=human,
    #         type_agent=type_agent
    #     )
    #     return agent_state

    # def create_preset(self, preset: Preset) -> Preset:
    #     if preset.user_id is None:
    #         preset.user_id = self.user_id
    #     preset = self.server.create_preset(preset=preset)
    #     return preset

    # def delete_preset(self, preset_id: uuid.UUID):
    #     preset = self.server.delete_preset(preset_id=preset_id, user_id=self.user_id)

    # def list_presets(self) -> List[PresetModel]:
    #     return self.server.list_presets(user_id=self.user_id)

    # def get_agent_config(self, agent_id: str) -> AgentState:
    #     self.interface.clear()
    #     return self.server.get_agent_config(user_id=self.user_id, agent_id=agent_id)

    # def get_agent_memory(self, agent_id: str) -> Dict:
    #     self.interface.clear()
    #     return self.server.get_agent_memory(user_id=self.user_id, agent_id=agent_id)

    # def update_agent_core_memory(self, agent_id: str, new_memory_contents: Dict) -> Dict:
    #     self.interface.clear()
    #     return self.server.update_agent_core_memory(user_id=self.user_id, agent_id=agent_id, new_memory_contents=new_memory_contents)

    # def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:
    #     self.interface.clear()
    #     self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
    #     if self.auto_save:
    #         self.save()
    #     else:
    #         # return self.interface.to_list()
    #         return UserMessageResponse(messages=self.interface.to_list())

    # def run_command(self, agent_id: str, command: str) -> Union[str, None]:
    #     self.interface.clear()
    #     return self.server.run_command(user_id=self.user_id, agent_id=agent_id, command=command)

    # def save(self):
    #     self.server.save_agents()

    # def load_data(self, connector: DataConnector, source_name: str):
    #     self.server.load_data(user_id=self.user_id, connector=connector, source_name=source_name)

    # def create_source(self, name: str):
    #     self.server.create_source(user_id=self.user_id, name=name)

    # def attach_source_to_agent(self, source_id: uuid.UUID, agent_id: uuid.UUID):
    #     self.server.attach_source_to_agent(user_id=self.user_id, source_id=source_id, agent_id=agent_id)

    # def delete_agent(self, agent_id: uuid.UUID):
    #     self.server.delete_agent(user_id=self.user_id, agent_id=agent_id)

    # def get_agent_archival_memory(
    #     self, agent_id: uuid.UUID, before: Optional[uuid.UUID] = None, after: Optional[uuid.UUID] = None, limit: Optional[int] = 1000
    # ):
    #     _, archival_json_records = self.server.get_agent_archival_cursor(
    #         user_id=self.user_id,
    #         agent_id=agent_id,
    #         after=after,
    #         before=before,
    #         limit=limit,
    #     )
    #     return archival_json_records
