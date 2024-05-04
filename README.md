

<p align="center">
  <img src="./assets/logotest5.png" alt="TypeAgent logo"></a>
</p>

<div align="center">

 <strong>TypeAgent allows you to create various types of  LLM agents types include Memgpt, OpenDevin, SWE, AIOS,Concordia,Generative Agents etc</strong>

</div>





TypeAgent makes it easy to build and deploy stateful LLM agents with support for: 
*  various types of  LLM agents,include Memgpt, OpenDevin, SWE, AIOS,Concordia,Generative Agents etc
* Long term memory/state managemet
* Connections to [external data sources] (e.g. PDF files) for RAG 
* Defining and calling [custom tools] (e.g. [google search])
* observation ,action,learn architecture

You can also use TypeAgent to depoy agents as a *service*. You can use a TypeAgent server to run a multi-user, multi-agent application on top of supported LLM providers.



## Installation & Setup   
Install TypeAgent:
```sh
pip install -U pytypeagent
```

## Quickstart (CLI)  
You can create and chat with a TypeAgent agent by running `typeagent run` in your CLI. The `run` command supports the following optional flags (see the [CLI documentation] for the full list of flags):
* `--agent`: (str) Name of agent to create or to resume chatting with.
* `--first`: (str) Allow user to sent the first message.
* `--debug`: (bool) Show debug logs (default=False)
* `--no-verify`: (bool) Bypass message verification (default=False)
* `--yes`/`-y`: (bool) Skip confirmation prompt and use defaults (default=False)

You can view the list of available in-chat commands (e.g. `/memory`, `/exit`) in the [CLI documentation].


## Quickstart (Server)  


**1:** Run with the CLI:
1. Run typeagent configure
2. Run `typeagent server`
3. Go to `localhost:8283` in the browser to view the developer portal 

Once the server is running, you can use the [Python client] or [REST API] to connect to  `localhost:8283` (if you're running with the CLI) to create users, agents, and more. The service requires authentication with a TypeAgent admin password, which can be set with running `export TYPEAGENT_SERVER_PASS=password`. 


## Supported Agent Type 
TypeAgent is designed to be Agent Type  agnostic. The following Agent Type are supported: 

| Type                | support status    |
|---------------------|-----------------  |
| Memgpt              | ✅               |
| OpenDevin_Planer    | ✅               |
| OpenDevin_SWE       | ❌               |
| OpenDevin_CodeAct   | ❌               | 
| OpenDevin_Micro     | ❌               | 
| OpenDevin_Monologue | ❌               | 
| Devika              | ❌               |
| Perplexica          | ❌               |
| Concordia           | ❌               | 
| Charlie Mnemonic    | ❌               |
| AIOS                | ❌               |
| Open Interpreter    | ❌               |
| SalesGPT            | ❌               |
| SWE                 | ❌               |
| Generative Agents   | ❌               |

The implementation methods of all the above types of agents are different. Now they are integrated into the unified interface through TypeAgent. 

Later, different types of agents will be added one after another


## Supported Endpoints & Backends 
TypeAgent is designed to be model and provider agnostic. The following LLM and embedding endpoints are supported: 

| Provider            | LLM Endpoint    | Embedding Endpoint |
|---------------------|-----------------|--------------------|
| OpenAI              | ✅               | ✅                  |
| Azure OpenAI        | ✅               | ✅                  |
| Google AI (Gemini)  | ✅               | ❌                  |
| Anthropic (Claude)  | ✅               | ❌                  |
| Groq                | ✅ (alpha release) | ❌                  |
| Cohere API          | ✅               | ❌                  |
| vLLM                | ✅               | ❌                  |
| Ollama              | ✅               | ❌                  |
| LM Studio           | ✅               | ❌                  |
| koboldcpp           | ✅               | ❌                  |
| oobabooga web UI    | ✅               | ❌                  |
| llama.cpp           | ✅               | ❌                  |
| HuggingFace TEI     | ❌               | ✅                  |

When using TypeAgent with open LLMs (such as those downloaded from HuggingFace), the performance of TypeAgent will be highly dependent on the LLM's function calling ability ,Language understanding and reasoning skills.
## Documentation
See full documentation at:


## Comments

- Our codebase for the TypeAgent builds heavily on [MemGPT codebase](https://github.com/cpacker/MemGPT?tab=readme-ov-file)
Thanks for open-sourcing! ,Our code will be open soon.....

- The implementation of the all type of agent is refer to respective source code,Thanks for open-sourcing!  and integrated into MemGPT structure

## Legal notices
By using TypeAgent and related TypeAgent services (such as the TypeAgent endpoint ), you agree to our [privacy policy](PRIVACY.md) and [terms of service](TERMS.md).


## Roadmap
You can view (and comment on!) the TypeAgent developer roadmap on GitHub: 
