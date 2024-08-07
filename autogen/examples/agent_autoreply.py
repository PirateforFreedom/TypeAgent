"""Example of how to add typeagent into an AutoGen groupchat

Based on the official AutoGen example here: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb

Begin by doing:
  pip install "pyautogen[teachable]"
  pip install pytypeagent
  or
  pip install -e . (inside the typeagent home directory)
"""

import os

import autogen

from typeagent.autogen.typeagent_agent import create_typeagent_autogen_agent_from_config
from typeagent.constants import DEFAULT_PRESET, LLM_MAX_TOKENS

LLM_BACKEND = "openai"
# LLM_BACKEND = "azure"
# LLM_BACKEND = "local"

if LLM_BACKEND == "openai":
    # For demo purposes let's use gpt-4
    model = "gpt-4"

    openai_api_key = os.getenv("OPENAI_API_KEY")
    assert openai_api_key, "You must set OPENAI_API_KEY or set LLM_BACKEND to 'local' to run this example"

    # This config is for AutoGen agents that are not powered by typeagent
    config_list = [
        {
            "model": model,
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ]

    # This config is for AutoGen agents that powered by typeagent
    config_list_typeagent = [
        {
            "model": model,
            "context_window": LLM_MAX_TOKENS[model],
            "preset": DEFAULT_PRESET,
            "model_wrapper": None,
            # OpenAI specific
            "model_endpoint_type": "openai",
            "model_endpoint": "https://api.openai.com/v1",
            "openai_key": openai_api_key,
        },
    ]

elif LLM_BACKEND == "azure":
    # Make sure that you have access to this deployment/model on your Azure account!
    # If you don't have access to the model, the code will fail
    model = "gpt-4"

    azure_openai_api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_version = os.getenv("AZURE_OPENAI_VERSION")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    assert (
        azure_openai_api_key is not None and azure_openai_version is not None and azure_openai_endpoint is not None
    ), "Set all the required OpenAI Azure variables (see: https://typeagent.readme.io/docs/endpoints#azure-openai)"

    # This config is for AutoGen agents that are not powered by typeagent
    config_list = [
        {
            "model": model,
            "api_type": "azure",
            "api_key": azure_openai_api_key,
            "api_version": azure_openai_version,
            # NOTE: on versions of pyautogen < 0.2.0, use "api_base"
            # "api_base": azure_openai_endpoint,
            "base_url": azure_openai_endpoint,
        }
    ]

    # This config is for AutoGen agents that powered by typeagent
    config_list_typeagent = [
        {
            "model": model,
            "context_window": LLM_MAX_TOKENS[model],
            "preset": DEFAULT_PRESET,
            "model_wrapper": None,
            # Azure specific
            "model_endpoint_type": "azure",
            "azure_key": azure_openai_api_key,
            "azure_endpoint": azure_openai_endpoint,
            "azure_version": azure_openai_version,
        },
    ]

elif LLM_BACKEND == "local":
    # Example using LM Studio on a local machine
    # You will have to change the parameters based on your setup

    # Non-typeagent agents will still use local LLMs, but they will use the ChatCompletions endpoint
    config_list = [
        {
            "model": "NULL",  # not needed
            # NOTE: on versions of pyautogen < 0.2.0 use "api_base", and also uncomment "api_type"
            # "api_base": "http://localhost:1234/v1",
            # "api_type": "open_ai",
            "base_url": "http://localhost:1234/v1",  # ex. "http://127.0.0.1:5001/v1" if you are using webui, "http://localhost:1234/v1/" if you are using LM Studio
            "api_key": "NULL",  #  not needed
        },
    ]

    # typeagent-powered agents will also use local LLMs, but they need additional setup (also they use the Completions endpoint)
    config_list_typeagent = [
        {
            "preset": DEFAULT_PRESET,
            "model": None,  # only required for Ollama, see: https://typeagent.readme.io/docs/ollama
            "context_window": 8192,  # the context window of your model (for Mistral 7B-based models, it's likely 8192)
            "model_wrapper": "chatml",  # chatml is the default wrapper
            "model_endpoint_type": "lmstudio",  # can use webui, ollama, llamacpp, etc.
            "model_endpoint": "http://localhost:1234",  # the IP address of your LLM backend
        },
    ]

else:
    raise ValueError(LLM_BACKEND)


# If USE_typeagent is False, then this example will be the same as the official AutoGen repo
# (https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb)
# If USE_typeagent is True, then we swap out the "coder" agent with a typeagent agent
USE_typeagent = True

llm_config = {"config_list": config_list, "seed": 42}
llm_config_typeagent = {"config_list": config_list_typeagent, "seed": 42}

# Set to True if you want to print typeagent's inner workings.
DEBUG = False
interface_kwargs = {
    "debug": DEBUG,
    "show_inner_thoughts": DEBUG,
    "show_function_outputs": DEBUG,
}

# The user agent
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
    human_input_mode="TERMINATE",  # needed?
    default_auto_reply="You are going to figure all out by your own. "
    "Work by yourself, the user won't reply until you output `TERMINATE` to end the conversation.",
)

if not USE_typeagent:
    # In the AutoGen example, we create an AssistantAgent to play the role of the coder
    coder = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config,
        system_message=f"I am a 10x engineer, trained in Python. I was the first engineer at Uber "
        f"(which I make sure to tell everyone I work with).",
        human_input_mode="TERMINATE",
        default_auto_reply="...",  # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
    )

else:
    # In our example, we swap this AutoGen agent with a typeagent agent
    # This typeagent agent will have all the benefits of typeagent, ie persistent memory, etc.
    coder = create_typeagent_autogen_agent_from_config(
        "typeagent_coder",
        llm_config=llm_config_typeagent,
        nontypeagent_llm_config=llm_config,
        system_message=f"I am a 10x engineer, trained in Python. I was the first engineer at Uber "
        f"(which I make sure to tell everyone I work with).",
        human_input_mode="TERMINATE",
        interface_kwargs=interface_kwargs,
        default_auto_reply="...",  # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
        skip_verify=False,  # NOTE: you should set this to True if you expect your typeagent AutoGen agent to call a function other than send_message on the first turn
    )

# Begin the group chat with a message from the user
user_proxy.initiate_chat(
    coder,
    message="I want to design an app to make me one million dollars in one month. " "Tell me all the details, then try out every steps.",
)
