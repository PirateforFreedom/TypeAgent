
## We are coding to add the opendevieplan agent to typeagent.Note:below typeagent python library is not working now,maybe later is ok.
## Current progress：TypeAgent V1.0  90%，TypeAgent V1.0 docs 65%


<p align="center">
  <img src="./assets/logotest5.png" alt="TypeAgent logo"></a>
</p>

<div align="center">

 <strong>TypeAgent allows you to create various types of  LLM agents,agent types include Memgpt, OpenDevin, SWE, AIOS,Concordia,Generative Agents etc,
 every type of agent has a unique architecture which has unique functionality </strong>
</div>





TypeAgent makes it easy to build and deploy stateful,various types of  LLM agents with support for: 
* Various types of  LLM agents,Integrated in unified TypeAgent framework, seamless switching various types of llm agents.
* LLM Type-architecture-functionality architecture which ensure every type of agent has unique feature
* Long term memory/state managemet,short term memory,knowledge base long term memory
* Basic RAG workflow for knowledge base which created by external data sources (e.g. PDF files)
* Intelligent switching knowledge base
* Defining and calling custom tools
* Changing personas settings and role-playing

You can also use TypeAgent to deploy agents as a *service*. You can use a TypeAgent server to run a multi-user, multi-agent application on top of supported type of agents  and supported LLM providers.



## Installation & Setup   
Install TypeAgent:
```sh
pip install -U pytypeagent
```

## Quickstart (CLI)  
You can create and chat with a TypeAgent agent by running `typeagent run` in your CLI. The `run` command supports the following optional flags (see the [CLI documentation] for the full list of flags):
* `--agent`: (str) Name of agent to create or to resume chatting with.
* `--type`:(str) type of agent to create or to resume chatting with,default type is memgpt
* `--first`: (str) Allow user to sent the first message.
* `--debug`: (bool) Show debug logs (default=False)

You can view the list of available in-chat commands (e.g. `/memory`, `/exit`) in the [CLI documentation].


## Quickstart (Server)  


**1:** Run with the CLI:
1. Run `typeagent configure`
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
| Transformer Agents  | ❌               |
| Langchain Agents    | ❌               |

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
| Ollama              | ✅               | ✅                  |
| LM Studio           | ✅               | ❌                  |
| koboldcpp           | ✅               | ❌                  |
| oobabooga web UI    | ✅               | ❌                  |
| llama.cpp           | ✅               | ❌                  |
| HuggingFace TEI     | ❌               | ✅                  |

When using TypeAgent with open LLMs (such as those downloaded from HuggingFace), the performance of TypeAgent will be highly dependent on the LLM's function calling ability ,Language understanding and reasoning skills.
## Documentation
See full documentation at:Coming soon


## Comments

- This project is a Leisure time hobby，If you are interested in the project ,you can make a issue, i am working on that .......
- when TypeAgent V1.0 code and docs are finished ,i will open source .
- 这个项目是业余爱好，刚开始，如果有小伙伴感兴趣，可以加入，但是要会Python
- Our codebase for the TypeAgent builds heavily on [MemGPT codebase](https://github.com/cpacker/MemGPT?tab=readme-ov-file)
Thanks for open-sourcing! ,Our code will be open soon.....

- The implementation of the all type of agent refer to respective source code,Thanks for open-sourcing!  and integrated into MemGPT structure

## Legal notices
By using TypeAgent and related TypeAgent services (such as the TypeAgent endpoint ), you agree to our [privacy policy](PRIVACY.md) and [terms of service](TERMS.md).


## Roadmap
You can view (and comment on!) the TypeAgent developer roadmap on GitHub: Coming soon
