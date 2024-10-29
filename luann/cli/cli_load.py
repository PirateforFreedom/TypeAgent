"""
This file contains functions for loading data into typeagent's archival storage.

Data can be loaded with the following command, once a load function is defined:
```
typeagent load <data-connector-type> --name <dataset-name> [ADDITIONAL ARGS]
```

"""
import os
import uuid
from typing import Annotated, List, Optional
import threading
import typer
import tempfile
from constants import KNOWLEDGE_BASE_DIR
from agent_store.storage import StorageConnector, StorageType
from config import typeagentConfig
from data_sources.connectors import (
    DirectoryConnector,
    # VectorDBConnector,
    load_data,
)
from data_types import Source
from metadata import MetadataStore
from models.pydantic_models import (
    DocumentModel,
    JobModel,
    JobStatus,
    PassageModel,
    SourceModel,
)
app = typer.Typer()

# NOTE: not supported due to llama-index breaking things (please reach out if you still need it)
# @app.command("index")
# def load_index(
#    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
#    dir: Annotated[Optional[str], typer.Option(help="Path to directory containing index.")] = None,
#    user_id: Annotated[Optional[uuid.UUID], typer.Option(help="User ID to associate with dataset.")] = None,
# ):
#    """Load a LlamaIndex saved VectorIndex into typeagent"""
#    if user_id is None:
#        config = typeagentConfig.load()
#        user_id = uuid.UUID(config.anon_clientid)
#
#    try:
#        # load index data
#        storage_context = StorageContext.from_defaults(persist_dir=dir)
#        loaded_index = load_index_from_storage(storage_context)
#
#        # hacky code to extract out passages/embeddings (thanks a lot, llama index)
#        embed_dict = loaded_index._vector_store._data.embedding_dict
#        node_dict = loaded_index._docstore.docs
#
#        # create storage connector
#        config = typeagentConfig.load()
#        if user_id is None:
#            user_id = uuid.UUID(config.anon_clientid)
#
#        passages = []
#        for node_id, node in node_dict.items():
#            vector = embed_dict[node_id]
#            node.embedding = vector
#            # assume embedding are the same as config
#            passages.append(
#                Passage(
#                    text=node.text,
#                    embedding=np.array(vector),
#                    embedding_dim=config.default_embedding_config.embedding_dim,
#                    embedding_model=config.default_embedding_config.embedding_model,
#                )
#            )
#            assert config.default_embedding_config.embedding_dim == len(
#                vector
#            ), f"Expected embedding dimension {config.default_embedding_config.embedding_dim}, got {len(vector)}"
#
#        if len(passages) == 0:
#            raise ValueError(f"No passages found in index {dir}")
#
#        insert_passages_into_source(passages, name, user_id, config)
#    except ValueError as e:
#        typer.secho(f"Failed to load index from provided information.\n{e}", fg=typer.colors.RED)
from data_sources.connectors import DataConnector
from typing import Callable, List, Optional, Tuple, Union
from utils import create_uuid_from_string
from embeddings import embedding_model
from data_types import (
    AgentState,
    EmbeddingConfig,
    LLMConfig,
    Message,
    Preset,
    Source,
    Token,
    User,
    Document,
    Passage,
)
def load_data(
        self,
        user_id: uuid.UUID,
        connector: DataConnector,
        source_name: str,
        config = typeagentConfig,
        # user_id = uuid.UUID(config.anon_clientid)
        ms = MetadataStore,
    ) -> Tuple[int, int]:
        """Load data from a DataConnector into a source for a specified user_id"""
        # TODO: this should be implemented as a batch job or at least async, since it may take a long time

        # load data from a data source into the document store
        source =ms.get_source(source_name=source_name, user_id=user_id)
        if source is None:
            raise ValueError(f"Data source {source_name} does not exist for user {user_id}")

        # get the data connectors
        passage_store = StorageConnector.get_storage_connector(StorageType.KNOWLEDGE_BASE_PASSAGES, config, user_id=user_id)

        """Load data from a connector (generates documents and passages) into a specified source_id, associatedw with a user_id."""
    
        passages = []
        # embedding_to_document_name = {}
        passage_count = 0
        document_count = 0
        
        for document_text, document_metadata in connector.generate_documents():
            doctempid=create_uuid_from_string(f"{str(source.id)}_{document_text}"),
        # insert document into storage
            documentone = DocumentModel(
            id=doctempid[0],
            text=document_text,
            metadata_=document_metadata,
            source_name=source.name,
            user_id=source.user_id,
            source_id=source.id,
            user_status="on",
            )
            document_count += 1
            # print(str(doctempid[0]))
            # print(document_text)
            # print(document_metadata)
            ms.add_Document(documentone)
            documentonelocal=Document(
                text = document_text,
                metadata = document_metadata,
                id=doctempid[0],

            )
        # generate passages
            for passage_text, passage_metadata in connector.generate_passages([documentonelocal], chunk_size=config.default_embedding_config.embedding_chunk_size):

            # for some reason, llama index parsers sometimes return empty strings
                if len(passage_text) == 0:
                    typer.secho(
                    f"Warning: embedding text returned empty string, skipping insert of passage with metadata '{passage_metadata}' into VectorDB. You can usually ignore this warning.",
                    fg=typer.colors.YELLOW,
                   )
                    continue

            # get embedding
                try:
                   
                   print("passage_text:")
                   print(passage_text)
                   embed_model = embedding_model(config.default_embedding_config)

                   embedding = embed_model.embed_documents([passage_text])
                except Exception as e:
                    typer.secho(
                    f"Warning: Failed to get embedding for {passage_text} (error: {str(e)}), skipping insert into VectorDB.",
                    fg=typer.colors.YELLOW,
                    )
                    continue
                passagetempid=create_uuid_from_string(f"{str(source.id)}_{passage_text}")
                print(str(passagetempid))
                passage = PassageModel(
                   id=passagetempid,
                   text=passage_text,
                   doc_id=documentonelocal.id,
                   metadata_=passage_metadata,
                   user_id=source.user_id,
                   source_name=source.name,
                   source_id=source.id,
                   embedding_model=source.embedding_model,
                   embedding=embedding[0],
                   user_status="on",
                 )
                
                 
                ms.add_passages(passage)
           
                passageloca=Passage(
                    id=passagetempid,
                    text=passage_text,
                    doc_id=documentonelocal.id,
                    metadata_=passage_metadata,
                    user_id=source.user_id,
                    data_source=source.name,
                    embedding_model=source.embedding_model,
                    embedding=embedding[0],
                )
                passages.append(passageloca)
         
                if len(passages) >= 100:
                # insert passages into passage store
                    passage_store.insert_many(passages)

                    passage_count += len(passages)
                    passages = []

            # break
        if len(passages) > 0:
        # insert passages into passage store
            passage_store.insert_many(passages)
            passage_count += len(passages)
        
        return passage_count, document_count



@app.command("directory")
def load_directory(
    filepathdirectory: Annotated[str, typer.Option(help="path of dataset to load.")],
    source_name: Annotated[Optional[str], typer.Option(help="upload data to a source")] = None,
    filename: Annotated[str, typer.Option(help="name of file ")] = None,
):
    try:
        config = typeagentConfig.load()
        user_id = uuid.UUID(config.anon_clientid)
        ms = MetadataStore(config)
        source =ms.get_source(source_name=source_name, user_id=user_id)
        text=""
        fullfilepath=filepathdirectory+"\\"+filename
        with open(fullfilepath, "r") as f:
            text = f.read()
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, filename)
            with open(file_path, "w") as buffer:
                buffer.write(text)
            file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
            with open(file_path, "w") as f:
                   f.write(text)
    
            connector = DirectoryConnector(input_directory=tmpdirname)

            # TODO: pre-compute total number of passages?

            # load the data into the source via the connector
            num_passages, num_documents =load_data(user_id=user_id, source_name=source.name, connector=connector,config=config,ms=ms)
        # return job
        
            print(f"Loaded {num_passages} passages and {num_documents} documents from {source.name}")
    except Exception as e:
        typer.secho(f"Failed to load data from provided information.\n{e}", fg=typer.colors.RED)
        ms.delete_source(source_id=source.id)

   


