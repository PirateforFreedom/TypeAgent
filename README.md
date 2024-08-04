




<p align="center">
  <img src="./assets/logotest5.png" alt="TypeAgent logo"></a>
</p>

<div align="center">

 <strong>TypeAgent allows you to create a LLM agent,which has complete memory module (long-term memory, short-term memory) and knowledge module（Various knowledge bases）,eventually become an agent with both EQ and IQ</strong>
</div>





TypeAgent makes it easy to build and deploy stateful LLM agents with support for: 
* Long term memory/state managemet,short term memory,knowledge base long term memory
* Basic RAG workflow for knowledge base which created by external data sources (e.g. PDF files)
* Intelligent switching knowledge base
* Defining and calling custom tools
* Changing personas settings and role-playing

You can also use TypeAgent to deploy agents as a *service*. You can use a TypeAgent server to run a multi-user, multi-agent application on top of supported LLM providers.



## Quickstart (Server)  

0. Clone the repo
1. Run `typeagent configure`
2. Run `typeagent server`
3. Go to `localhost:8283` in the browser to view the developer portal

Once the server is running, you can use the [REST API] to connect to  `localhost:8283` (if you're running with the CLI) to create users, agents, and more. The service requires authentication with a TypeAgent admin password, which can be set with running `export TYPEAGENT_SERVER_PASS=password`. 


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


## TODO LIST

- Add typeagent client
- Add other type agent
- test vectordb and other llms

## Comments

- This project is a Leisure time hobby，If you are interested in the project ,you can make a issue, i am working on that .......
- when TypeAgent V1.0 code and docs are finished ,i will open source .
- 这个项目是业余爱好，刚开始，如果有小伙伴感兴趣，可以加入，但是要会Python
- Our codebase for the TypeAgent builds heavily on [MemGPT codebase](https://github.com/cpacker/MemGPT?tab=readme-ov-file)
Thanks for open-sourcing! ,Our code will be open soon.....
- The difference of MemGPT and Typeagent is that typeagent optimizes the entire memgpt code structure and creatively adds a complete memory module and knowledge base module
  
## Roadmap
goal: EQ and IQ AGENT
