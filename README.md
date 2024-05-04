

<p align="center">
  <img src="./assets/logotest5.png" alt="TypeAgent logo"></a>
</p>

<div align="center">

 <strong>TypeAgent allows you to create various types of  LLM agents types include Memgpt, OpenDevin, SWE, AIOS,Concordia,Generative Agents etc</strong>

[![Documentation](https://img.shields.io/github/v/release/cpacker/MemGPT?label=Documentation&logo=readthedocs&style=flat-square)](https://memgpt.readme.io/docs)

</div>

TypeAgent makes it easy to build and deploy stateful LLM agents with support for: 
*  various types of  LLM agents,include Memgpt, OpenDevin, SWE, AIOS,Concordia,Generative Agents etc
* Long term memory/state management 
* Connections to [external data sources](https://memgpt.readme.io/docs/data_sources) (e.g. PDF files) for RAG 
* Defining and calling [custom tools](https://memgpt.readme.io/docs/functions) (e.g. [google search](https://github.com/cpacker/MemGPT/blob/main/examples/google_search.py))

You can also use TypeAgent to depoy agents as a *service*. You can use a TypeAgent server to run a multi-user, multi-agent application on top of supported LLM providers.

<img width="1000" alt="image" src="https://github.com/cpacker/MemGPT/assets/8505980/1096eb91-139a-4bc5-b908-fa585462da09">


## Installation & Setup   
Install TypeAgent:
```sh
pip install -U pytypeagent
```
To use TypeAgent with OpenAI, set the environment variable `OPENAI_API_KEY` to your OpenAI key then run: 
```
typeagent quickstart --backend openai
```
To use TypeAgent with a free hosted endpoint, you run run: 
```
typeagent quickstart --backend typeagent
```
For more advanced configuration options or to use a different [LLM backend](https://memgpt.readme.io/docs/endpoints) or [local LLMs](https://memgpt.readme.io/docs/local_llm), run `typeagent configure`. 

## Quickstart (CLI)  
You can create and chat with a TypeAgent agent by running `typeagent run` in your CLI. The `run` command supports the following optional flags (see the [CLI documentation](https://memgpt.readme.io/docs/quickstart) for the full list of flags):
* `--agent`: (str) Name of agent to create or to resume chatting with.
* `--first`: (str) Allow user to sent the first message.
* `--debug`: (bool) Show debug logs (default=False)
* `--no-verify`: (bool) Bypass message verification (default=False)
* `--yes`/`-y`: (bool) Skip confirmation prompt and use defaults (default=False)

You can view the list of available in-chat commands (e.g. `/memory`, `/exit`) in the [CLI documentation](https://memgpt.readme.io/docs/quickstart).

## Dev portal (alpha build)
TypeAgent provides a developer portal that enables you to easily create, edit, monitor, and chat with your TypeAgent agents. The easiest way to use the dev portal is to install MemGPT via **docker** (see instructions below).

<img width="1000" alt="image" src="https://github.com/cpacker/MemGPT/assets/5475622/071117c5-46a7-4953-bc9d-d74880e66258">

## Quickstart (Server)  


**Option 2:** Run with the CLI:
1. Run `memgpt server`
2. Go to `localhost:8283` in the browser to view the developer portal 

Once the server is running, you can use the [Python client](https://memgpt.readme.io/docs/admin-client) or [REST API](https://memgpt.readme.io/reference/api) to connect to `memgpt.localhost` (if you're running with docker compose) or `localhost:8283` (if you're running with the CLI) to create users, agents, and more. The service requires authentication with a MemGPT admin password, which can be set with running `export MEMGPT_SERVER_PASS=password`. 


## Supported Endpoints & Backends 
MemGPT is designed to be model and provider agnostic. The following LLM and embedding endpoints are supported: 

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

When using MemGPT with open LLMs (such as those downloaded from HuggingFace), the performance of MemGPT will be highly dependent on the LLM's function calling ability. You can find a list of LLMs/models that are known to work well with MemGPT on the [#model-chat channel on Discord](https://discord.gg/9GEQrxmVyE), as well as on [this spreadsheet](https://docs.google.com/spreadsheets/d/1fH-FdaO8BltTMa4kXiNCxmBCQ46PRBVp3Vn6WbPgsFs/edit?usp=sharing).

## Documentation
See full documentation at: https://memgpt.readme.io

## Support
For issues and feature requests, please [open a GitHub issue](https://github.com/cpacker/MemGPT/issues) or message us on our `#support` channel on [Discord](https://discord.gg/9GEQrxmVyE).

## Legal notices
By using MemGPT and related MemGPT services (such as the MemGPT endpoint or hosted service), you agree to our [privacy policy](PRIVACY.md) and [terms of service](TERMS.md).

## Roadmap
You can view (and comment on!) the MemGPT developer roadmap on GitHub: https://github.com/cpacker/MemGPT/issues/1200.

