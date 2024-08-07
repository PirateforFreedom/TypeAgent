import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from autogen.agentchat import (
    Agent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
)

import typeagent.constants as constants
import typeagent.system as system
import typeagent.utils as utils
from typeagent.agent import Agent as typeagentAgent
from typeagent.agent import save_agent
from typeagent.agent_store.storage import StorageConnector, StorageType
from typeagent.autogen.interface import AutoGenInterface
from typeagent.cli.cli_load import load_directory, load_vector_database
from typeagent.config import typeagentConfig
from typeagent.credentials import typeagentCredentials
from typeagent.data_types import EmbeddingConfig, LLMConfig, User
from typeagent.metadata import MetadataStore
from typeagent.utils import get_human_text, get_persona_text


class typeagentConversableAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        agent: typeagentAgent,
        skip_verify: bool = False,
        auto_save: bool = False,
        concat_other_agent_messages: bool = False,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
    ):
        """A wrapper around a typeagent agent that implements the AutoGen ConversibleAgent functions

        This allows the typeagent agent to be used in an AutoGen groupchat
        """
        super().__init__(name, llm_config=False)
        self.agent = agent
        self.skip_verify = skip_verify
        self.auto_save = auto_save

        self.concat_other_agent_messages = concat_other_agent_messages
        self.register_reply([Agent, None], typeagentConversableAgent._generate_reply_for_user_message)
        self.messages_processed_up_to_idx = 0
        self._default_auto_reply = default_auto_reply

        self._is_termination_msg = is_termination_msg if is_termination_msg is not None else (lambda x: x == "TERMINATE")

        config = typeagentConfig.load()
        self.ms = MetadataStore(config)

    def save(self):
        """Save the underlying typeagent agent to the database"""
        try:
            save_agent(agent=self.agent, ms=self.ms)
        except Exception as e:
            print(f"Failed to save typeagent AutoGen agent\n{self.agent}\nError: {str(e)}")
            raise

    def load(self, name: str, type: str, **kwargs):
        # call load function based on type
        if type == "directory":
            load_directory(name=name, **kwargs)
        elif type == "webpage":
            load_webpage(name=name, **kwargs)
        elif type == "database":
            load_database(name=name, **kwargs)
        elif type == "vector_database":
            load_vector_database(name=name, **kwargs)
        else:
            raise ValueError(f"Invalid data source type {type}")

    def attach(self, data_source: str):
        # attach new data
        config = typeagentConfig.load()
        source_connector = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id=self.agent.agent_state.user_id)
        self.agent.attach_source(data_source, source_connector, ms=self.ms)

    def load_and_attach(self, name: str, type: str, force=False, **kwargs):
        # check if data source already exists
        data_source_options = self.ms.list_sources(user_id=self.agent.agent_state.user_id)
        data_source_options = [s.name for s in data_source_options]

        kwargs["user_id"] = self.agent.agent_state.user_id

        if name in data_source_options and not force:
            print(f"Data source {name} already exists. Use force=True to overwrite.")
            self.attach(name)
        else:
            self.load(name, type, **kwargs)
            self.attach(name)

    def format_other_agent_message(self, msg):
        if "name" in msg:
            user_message = f"{msg['name']}: {msg['content']}"
        else:
            user_message = msg["content"]
        return user_message

    def find_last_user_message(self):
        last_user_message = None
        for msg in self.agent.messages:
            if msg["role"] == "user":
                last_user_message = msg["content"]
        return last_user_message

    def find_new_messages(self, entire_message_list):
        """Extract the subset of messages that's actually new"""
        return entire_message_list[self.messages_processed_up_to_idx :]

    @staticmethod
    def _format_autogen_message(autogen_message):
        # {'content': "...", 'name': '...', 'role': 'user'}
        if not isinstance(autogen_message, dict) or ():
            print(f"Warning: AutoGen message was not a dict -- {autogen_message}")
            user_message = system.package_user_message(autogen_message)
        elif "content" not in autogen_message or "name" not in autogen_message or "name" not in autogen_message:
            print(f"Warning: AutoGen message was missing fields -- {autogen_message}")
            user_message = system.package_user_message(autogen_message)
        else:
            user_message = system.package_user_message(user_message=autogen_message["content"], name=autogen_message["name"])

        return user_message

    def _generate_reply_for_user_message(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        assert isinstance(
            self.agent.interface, AutoGenInterface
        ), f"typeagent AutoGen Agent is using the wrong interface - {self.agent.interface}"
        self.agent.interface.reset_message_list()

        new_messages = self.find_new_messages(messages)
        new_messages_count = len(new_messages)
        if new_messages_count > 1:
            if self.concat_other_agent_messages:
                # Combine all the other messages into one message
                user_message = "\n".join([self.format_other_agent_message(m) for m in new_messages])
            else:
                # Extend the typeagent message list with multiple 'user' messages, then push the last one with agent.step()
                self.agent.append_to_messages(new_messages[:-1])
                user_message = new_messages[-1]
        elif new_messages_count == 1:
            user_message = new_messages[0]
        else:
            return True, self._default_auto_reply

        # Package the user message
        # user_message = system.package_user_message(user_message)
        user_message = self._format_autogen_message(user_message)

        # Send a single message into typeagent
        while True:
            (
                new_messages,
                heartbeat_request,
                function_failed,
                token_warning,
                tokens_accumulated,
            ) = self.agent.step(user_message, first_message=False, skip_verify=self.skip_verify)
            # Skip user inputs if there's a memory warning, function execution failed, or the agent asked for control
            if token_warning:
                user_message = system.get_token_limit_warning()
            elif function_failed:
                user_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
            elif heartbeat_request:
                user_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
            else:
                break

        # Stop the conversation
        if self._is_termination_msg(new_messages[-1]["content"]):
            return True, None

        # Pass back to AutoGen the pretty-printed calls typeagent made to the interface
        pretty_ret = typeagentConversableAgent.pretty_concat(self.agent.interface.message_list)
        self.messages_processed_up_to_idx += new_messages_count

        # If auto_save is on, save after every full step
        if self.auto_save:
            self.save()

        return True, pretty_ret

    @staticmethod
    def pretty_concat(messages):
        """AutoGen expects a single response, but typeagent may take many steps.

        To accommodate AutoGen, concatenate all of typeagent's steps into one and return as a single message.
        """
        ret = {"role": "assistant", "content": ""}
        lines = []
        for m in messages:
            lines.append(f"{m}")
        ret["content"] = "\n".join(lines)

        # prevent error in LM Studio caused by scenarios where typeagent didn't say anything
        if ret["content"] in ["", "\n"]:
            ret["content"] = "..."

        return ret


def update_config_from_dict(config_object: Union[LLMConfig, EmbeddingConfig], config_dict: dict) -> bool:
    """Utility method used in the agent creation process for AutoGen

    Update the attributes of a configuration object based on a dictionary.

    :param config_object: The configuration object to be updated.
    :param config_dict: The dictionary containing new values for the configuration.
    """
    was_modified = False
    for attr in dir(config_object):
        # Filter out private attributes and methods
        if not attr.startswith("_") and not callable(getattr(config_object, attr)):
            if attr in config_dict:
                # Cast the value to the type of the attribute in config_object
                attr_type = type(getattr(config_object, attr))
                try:
                    setattr(config_object, attr, attr_type(config_dict[attr]))
                    was_modified = True
                except TypeError:
                    print(f"Type mismatch for attribute {attr}, cannot cast {config_dict[attr]} to {attr_type}")

    return was_modified


def load_autogen_typeagent_agent(
    agent_config: dict,
    skip_verify: bool = False,
    auto_save: bool = False,
    interface: bool = None,
    interface_kwargs: dict = {},
    default_auto_reply: Optional[Union[str, Dict, None]] = "",
    is_termination_msg: Optional[Callable[[Dict], bool]] = None,
) -> typeagentConversableAgent:
    """Load a typeagent agent into a wrapped ConversableAgent class"""
    if "name" not in agent_config:
        raise ValueError("Must provide 'name' in agent_config to load an agent")

    interface = AutoGenInterface(**interface_kwargs) if interface is None else interface

    config = typeagentConfig.load()
    # Create the default user, or load the specified user
    ms = MetadataStore(config)
    if "user_id" not in agent_config:
        user_id = uuid.UUID(config.anon_clientid)
        user = ms.get_user(user_id=user_id)
        if user is None:
            ms.create_user(User(id=user_id))
            user = ms.get_user(user_id=user_id)
            if user is None:
                raise ValueError(f"Failed to create default user {str(user_id)} in database.")
    else:
        user_id = uuid.UUID(agent_config["user_id"])
        user = ms.get_user(user_id=user_id)

    # Make sure that the agent already exists
    agent_state = ms.get_agent(agent_name=agent_config["name"], user_id=user.id)
    if agent_state is None:
        raise ValueError(f"Couldn't find an agent named {agent_config['name']} in the agent database")

    # Create the agent object directly from the loaded state (not via preset creation)
    try:
        typeagent_agent = typeagentAgent(agent_state=agent_state, interface=interface)
    except Exception:
        print(f"Failed to create an agent object from agent state =\n{agent_state}")
        raise

    # If the user provided new config information, write it out to the agent
    # E.g. if the user is trying to load the same agent, but on a new LLM backend
    llm_config_was_modified = update_config_from_dict(typeagent_agent.agent_state.llm_config, agent_config)
    embedding_config_was_modified = update_config_from_dict(typeagent_agent.agent_state.embedding_config, agent_config)
    if llm_config_was_modified or embedding_config_was_modified:
        save_agent(agent=typeagent_agent, ms=ms)

    # After creating the agent, we then need to wrap it in a ConversableAgent so that it can be plugged into AutoGen
    autogen_typeagent_agent = typeagentConversableAgent(
        name=agent_state.name,
        agent=typeagent_agent,
        default_auto_reply=default_auto_reply,
        is_termination_msg=is_termination_msg,
        skip_verify=skip_verify,
        auto_save=auto_save,
    )
    return autogen_typeagent_agent


def create_autogen_typeagent_agent(
    agent_config: dict,
    skip_verify: bool = False,
    auto_save: bool = False,
    interface: bool = None,
    interface_kwargs: dict = {},
    default_auto_reply: Optional[Union[str, Dict, None]] = "",
    is_termination_msg: Optional[Callable[[Dict], bool]] = None,
) -> typeagentConversableAgent:
    """
    See AutoGenInterface.__init__ for available options you can pass into
    `interface_kwargs`.  For example, typeagent's inner monologue and functions are
    off by default so that they are not visible to the other agents. You can
    turn these on by passing in
    ```
    interface_kwargs={
        "debug": True,  # to see all typeagent activity
        "show_inner_thoughts: True  # to print typeagent inner thoughts "globally"
                                    # (visible to all AutoGen agents)
    }
    ```
    """
    interface = AutoGenInterface(**interface_kwargs) if interface is None else interface

    config = typeagentConfig.load()
    llm_config = config.default_llm_config
    embedding_config = config.default_embedding_config

    # Overwrite parts of the LLM and embedding configs that were passed into the config dicts
    update_config_from_dict(llm_config, agent_config)
    update_config_from_dict(embedding_config, agent_config)

    # Create the default user, or load the specified user
    ms = MetadataStore(config)
    if "user_id" not in agent_config:
        user_id = uuid.UUID(config.anon_clientid)
        user = ms.get_user(user_id=user_id)
        if user is None:
            ms.create_user(User(id=user_id))
            user = ms.get_user(user_id=user_id)
            if user is None:
                raise ValueError(f"Failed to create default user {str(user_id)} in database.")
    else:
        user_id = uuid.UUID(agent_config["user_id"])
        user = ms.get_user(user_id=user_id)

    try:
        preset_obj = ms.get_preset(name=agent_config["preset"] if "preset" in agent_config else config.preset, user_id=user.id)
        if preset_obj is None:
            # create preset records in metadata store
            from typeagent.presets.presets import add_default_presets

            add_default_presets(user.id, ms)
            # try again
            preset_obj = ms.get_preset(name=agent_config["preset"] if "preset" in agent_config else config.preset, user_id=user.id)
            if preset_obj is None:
                print("Couldn't find presets in database, please run `typeagent configure`")
                sys.exit(1)

        # Overwrite fields in the preset if they were specified
        # TODO make sure that the human/persona aren't filenames but actually real values
        preset_obj.human = agent_config["human"] if "human" in agent_config else get_human_text(config.human)
        preset_obj.persona = agent_config["persona"] if "persona" in agent_config else get_persona_text(config.persona)

        typeagent_agent = typeagentAgent(
            interface=interface,
            name=agent_config["name"] if "name" in agent_config else None,
            created_by=user.id,
            preset=preset_obj,
            llm_config=llm_config,
            embedding_config=embedding_config,
            # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
            first_message_verify_mono=True if (llm_config.model is not None and "gpt-4" in llm_config.model) else False,
        )
        # Save agent in database immediately after writing
        save_agent(agent=typeagent_agent, ms=ms)
    except ValueError as e:
        raise ValueError(f"Failed to create agent from provided information:\n{agent_config}\n\nError: {str(e)}")

    # After creating the agent, we then need to wrap it in a ConversableAgent so that it can be plugged into AutoGen
    autogen_typeagent_agent = typeagentConversableAgent(
        name=typeagent_agent.agent_state.name,
        agent=typeagent_agent,
        default_auto_reply=default_auto_reply,
        is_termination_msg=is_termination_msg,
        skip_verify=skip_verify,
        auto_save=auto_save,
    )
    return autogen_typeagent_agent


def create_typeagent_autogen_agent_from_config(
    name: str,
    system_message: Optional[str] = "You are a helpful AI Assistant.",
    is_termination_msg: Optional[Callable[[Dict], bool]] = None,
    max_consecutive_auto_reply: Optional[int] = None,
    human_input_mode: Optional[str] = "ALWAYS",
    function_map: Optional[Dict[str, Callable]] = None,
    code_execution_config: Optional[Union[Dict, bool]] = None,
    llm_config: Optional[Union[Dict, bool]] = None,
    # config setup for non-typeagent agents:
    nontypeagent_llm_config: Optional[Union[Dict, bool]] = None,
    default_auto_reply: Optional[Union[str, Dict, None]] = "",
    interface_kwargs: Dict = None,
    skip_verify: bool = False,
    auto_save: bool = False,
) -> typeagentConversableAgent:
    """Same function signature as used in base AutoGen, but creates a typeagent agent

    Construct AutoGen config workflow in a clean way.
    """
    if not isinstance(llm_config, dict):
        llm_config = None
    llm_config = llm_config["config_list"][0]

    if interface_kwargs is None:
        interface_kwargs = {}

    # The "system message" in AutoGen becomes the persona in typeagent
    persona_desc = utils.get_persona_text(constants.DEFAULT_PERSONA) if system_message == "" else system_message
    # The user profile is based on the input mode
    if human_input_mode == "ALWAYS":
        user_desc = ""
    elif human_input_mode == "TERMINATE":
        user_desc = "Work by yourself, the user won't reply until you output `TERMINATE` to end the conversation."
    else:
        user_desc = "Work by yourself, the user won't reply. Elaborate as much as possible."

    # If using azure or openai, save the credentials to the config
    config = typeagentConfig.load()
    credentials = typeagentCredentials.load()

    if (
        llm_config["model_endpoint_type"] in ["azure", "openai"]
        or llm_config["model_endpoint_type"] != config.default_llm_config.model_endpoint_type
    ):
        # we load here to make sure we don't override existing values
        # all we want to do is add extra credentials

        if llm_config["model_endpoint_type"] == "azure":
            credentials.azure_key = llm_config["azure_key"]
            credentials.azure_endpoint = llm_config["azure_endpoint"]
            credentials.azure_version = llm_config["azure_version"]
            llm_config.pop("azure_key")
            llm_config.pop("azure_endpoint")
            llm_config.pop("azure_version")

        elif llm_config["model_endpoint_type"] == "openai":
            credentials.openai_key = llm_config["openai_key"]
            llm_config.pop("openai_key")

        credentials.save()

    # Create an AgentConfig option from the inputs
    llm_config.pop("name", None)
    llm_config.pop("persona", None)
    llm_config.pop("human", None)
    agent_config = dict(
        name=name,
        persona=persona_desc,
        human=user_desc,
        **llm_config,
    )

    if function_map is not None or code_execution_config is not None:
        raise NotImplementedError

    autogen_typeagent_agent = create_autogen_typeagent_agent(
        agent_config,
        default_auto_reply=default_auto_reply,
        is_termination_msg=is_termination_msg,
        interface_kwargs=interface_kwargs,
        skip_verify=skip_verify,
        auto_save=auto_save,
    )

    if human_input_mode != "ALWAYS":
        coop_agent1 = create_autogen_typeagent_agent(
            agent_config,
            default_auto_reply=default_auto_reply,
            is_termination_msg=is_termination_msg,
            interface_kwargs=interface_kwargs,
            skip_verify=skip_verify,
            auto_save=auto_save,
        )
        if default_auto_reply != "":
            coop_agent2 = UserProxyAgent(
                "User_proxy",
                human_input_mode="NEVER",
                default_auto_reply=default_auto_reply,
            )
        else:
            coop_agent2 = create_autogen_typeagent_agent(
                agent_config,
                default_auto_reply=default_auto_reply,
                is_termination_msg=is_termination_msg,
                interface_kwargs=interface_kwargs,
                skip_verify=skip_verify,
                auto_save=auto_save,
            )

        groupchat = GroupChat(
            agents=[autogen_typeagent_agent, coop_agent1, coop_agent2],
            messages=[],
            max_round=12 if max_consecutive_auto_reply is None else max_consecutive_auto_reply,
        )
        assert nontypeagent_llm_config is not None
        manager = GroupChatManager(name=name, groupchat=groupchat, llm_config=nontypeagent_llm_config)
        return manager

    else:
        return autogen_typeagent_agent
