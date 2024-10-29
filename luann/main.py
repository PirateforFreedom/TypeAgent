import json
import os
import sys
import traceback

import questionary
import requests
import typer
from rich.console import Console

import agent as agent
import errors as errors
import system as system
from agent_store.storage import StorageConnector, StorageType

# import benchmark
#from benchmark.benchmark import bench
from cli.cli import (
    delete_agent,
    migrate,
    open_folder,
    quickstart,
    run,
    server,
    version,
)
from cli.cli_config import add, configure, delete, list
from cli.cli_load import app as load_app
from config import typeagentConfig
from constants import (
    FUNC_FAILED_HEARTBEAT_MESSAGE,
    JSON_ENSURE_ASCII,
    JSON_LOADS_STRICT,
    REQ_HEARTBEAT_MESSAGE,
)
from metadata import MetadataStore

# from interface import CLIInterface as interface  # for printing to terminal
from streaming_interface import AgentRefreshStreamingInterface

# interface = interface()

app = typer.Typer(pretty_exceptions_enable=False)
app.command(name="run")(run)
app.command(name="version")(version)
app.command(name="configure")(configure)
app.command(name="list")(list)
app.command(name="add")(add)
app.command(name="delete")(delete)
app.command(name="server")(server)
app.command(name="folder")(open_folder)
# app.command(name="quickstart")(quickstart) later add official typeagent backend ,now delated
# load data commands
app.add_typer(load_app, name="load") 
# migration command
# app.command(name="migrate")(migrate)# this is the last part to be check
# benchmark command
#app.command(name="benchmark")(bench)
# delete agents
app.command(name="delete-agent")(delete_agent)

app()




