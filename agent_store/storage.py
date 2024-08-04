""" These classes define storage connectors.

We originally tried to use Llama Index VectorIndex, but their limited API was extremely problematic.
"""
from sqlalchemy import (
    BIGINT,
    BINARY,
    CHAR,
    JSON,
    Column,
    DateTime,
    String,
    TypeDecorator,
    and_,
    asc,
    create_engine,
    desc,
    func,
    or_,
    select,
    text,
)
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, mapped_column, sessionmaker
from sqlalchemy.orm.session import close_all_sessions
from sqlalchemy.sql import func
from sqlalchemy_json import MutableJson
from constants import MAX_EMBEDDING_DIM
from abc import abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union
from data_types import Message, Passage, Record, RecordType, ToolCall
from config import typeagentConfig
from data_types import Document, Message, Passage, Record, RecordType
from utils import printd
import base64
import os
import uuid
import numpy as np
# ENUM representing table types in typeagent
# each table corresponds to a different table schema  (specified in data_types.py)
class StorageType:
    ARCHIVAL_MEMORY = "archival_memory"  # recall memory table: typeagent_agent_{agent_id}
    RECALL_MEMORY = "recall_memory"  # archival memory table: typeagent_agent_recall_{agent_id}
    KNOWLEDGE_BASE="knowledge_base"
    KNOWLEDGE_BASE_PASSAGES = "knowledge_base_passages"  # TODO
    KNOWLEDGE_BASE_DOCUMENTS = "knowledge_base_documents"  # TODO


# table names used by typeagent

# agent tables
RECALL_TABLE_NAME = "recall_memory_agent"  # agent memory
ARCHIVAL_TABLE_NAME = "archival_memory_agent"  # agent memory

# external data source tables
PASSAGE_TABLE_NAME = "passages"  # chunked/embedded passages (from source)
DOCUMENT_TABLE_NAME = "documents"  # original documents (from source)
KNOWLEDGE_BASE = "knowledge_base"  # original documents (from source)
class CommonUUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR())

    def process_bind_param(self, value, dialect):
        if dialect.name == "postgresql" or value is None:
            return value
        else:
            return str(value)  # Convert UUID to string for SQLite

    def process_result_value(self, value, dialect):
        if dialect.name == "postgresql" or value is None:
            return value
        else:
            return uuid.UUID(value)


class CommonVector(TypeDecorator):
    """Common type for representing vectors in SQLite"""

    impl = BINARY
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(BINARY())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        # Ensure value is a numpy array
        if isinstance(value, list):
            value = np.array(value, dtype=np.float32)
        # Serialize numpy array to bytes, then encode to base64 for universal compatibility
        return base64.b64encode(value.tobytes())

    def process_result_value(self, value, dialect):
        if not value:
            return value
        # Check database type and deserialize accordingly
        if dialect.name == "sqlite":
            # Decode from base64 and convert back to numpy array
            value = base64.b64decode(value)
        # For PostgreSQL, value is already in bytes
        return np.frombuffer(value, dtype=np.float32)


# Custom serialization / de-serialization for JSON columns


class ToolCallColumn(TypeDecorator):
    """Custom type for storing List[ToolCall] as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            return [vars(v) for v in value]
        return value

    def process_result_value(self, value, dialect):
        if value:
            return [ToolCall(**v) for v in value]
        return value


Base = declarative_base()


def get_db_model(
    config: typeagentConfig,
    table_name: str,
    storage_type: StorageType,
    user_id: uuid.UUID,
    agent_id: Optional[uuid.UUID] = None,
    dialect="postgresql",
):
    # Define a helper function to create or get the model class
    def create_or_get_model(class_name, base_model, table_name):
        if class_name in globals():
            return globals()[class_name]
        Model = type(class_name, (base_model,), {"__tablename__": table_name, "__table_args__": {"extend_existing": True}})
        globals()[class_name] = Model
        return Model

    if storage_type == StorageType.ARCHIVAL_MEMORY:
        pass
        # # create schema for archival memory
        # class PassageModel(Base):
        #     """Defines data model for storing Passages (consisting of text, embedding)"""

        #     __abstract__ = True  # this line is necessary

        #     # Assuming passage_id is the primary key
        #     # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        #     id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
        #     # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        #     user_id = Column(CommonUUID, nullable=False)
        #     text = Column(String)
        #     doc_id = Column(CommonUUID)
        #     agent_id = Column(CommonUUID)
        #     data_source = Column(String)  # agent_name if agent, data_source name if from data source

        #     # vector storage
        #     if dialect == "sqlite":
        #         embedding = Column(CommonVector)
        #     # else:
        #     #     from pgvector.sqlalchemy import Vector

        #         # embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
        #     embedding_dim = Column(BIGINT)
        #     embedding_model = Column(String)

        #     metadata_ = Column(MutableJson)

        #     # Add a datetime column, with default value as the current time
        #     created_at = Column(DateTime(timezone=True))

        #     def __repr__(self):
        #         return f"<Passage(passage_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

        #     def to_record(self):
        #         return Passage(
        #             text=self.text,
        #             embedding=self.embedding,
        #             embedding_dim=self.embedding_dim,
        #             embedding_model=self.embedding_model,
        #             doc_id=self.doc_id,
        #             user_id=self.user_id,
        #             id=self.id,
        #             data_source=self.data_source,
        #             agent_id=self.agent_id,
        #             metadata_=self.metadata_,
        #             created_at=self.created_at,
        #         )

        # """Create database model for table_name"""
        # class_name = f"{table_name.capitalize()}Model" + dialect
        # return create_or_get_model(class_name, PassageModel, table_name)

    elif storage_type == StorageType.RECALL_MEMORY:

        class MessageModel(Base):
            """Defines data model for storing Message objects"""

            __abstract__ = True  # this line is necessary

            # Assuming message_id is the primary key
            # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
            id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
            # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            user_id = Column(CommonUUID, nullable=False)
            agent_id = Column(CommonUUID, nullable=False)

            # openai info
            role = Column(String, nullable=False)
            text = Column(String)  # optional: can be null if function call
            model = Column(String)  # optional: can be null if LLM backend doesn't require specifying
            name = Column(String)  # optional: multi-agent only

            # tool call request info
            # if role == "assistant", this MAY be specified
            # if role != "assistant", this must be null
            # TODO align with OpenAI spec of multiple tool calls
            tool_calls = Column(ToolCallColumn)

            # tool call response info
            # if role == "tool", then this must be specified
            # if role != "tool", this must be null
            tool_call_id = Column(String)

            # vector storage
            if dialect == "sqlite":
                embedding = Column(CommonVector)
            # else:
                # from pgvector.sqlalchemy import Vector

                # embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
            embedding_dim = Column(BIGINT)
            embedding_model = Column(String)

            # Add a datetime column, with default value as the current time
            created_at = Column(DateTime(timezone=True))

            def __repr__(self):
                return f"<Message(message_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

            def to_record(self):
                return Message(
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    role=self.role,
                    name=self.name,
                    text=self.text,
                    model=self.model,
                    tool_calls=self.tool_calls,
                    tool_call_id=self.tool_call_id,
                    embedding=self.embedding,
                    embedding_dim=self.embedding_dim,
                    embedding_model=self.embedding_model,
                    created_at=self.created_at,
                    id=self.id,
                )

        """Create database model for table_name"""
        class_name = f"{table_name.capitalize()}Model" + dialect
        return create_or_get_model(class_name, MessageModel, table_name)
    elif storage_type == StorageType.KNOWLEDGE_BASE:

        pass
    elif storage_type == StorageType.KNOWLEDGE_BASE_DOCUMENTS:

        # create schema for archival memory
        class DocumentModel(Base):
            """Defines data model for storing Documents (consisting of text, embedding)"""

            __abstract__ = True  # this line is necessary
            

            # id=create_uuid_from_string(f"{str(source.id)}_{document_text}"),
            # text=document_text,
            # metadata=document_metadata,
            # data_source=source.name,
            # user_id=source.user_id,


            # Assuming passage_id is the primary key
            # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
            id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
            # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
            user_id = Column(CommonUUID, nullable=False)
            text = Column(String)
            source_id= Column(CommonUUID)
            # agent_id= Column(CommonUUID)
            data_source = Column(String)  # agent_name if agent, data_source name if from data source

            # vector storage
            if dialect == "sqlite":
                embedding = Column(CommonVector)
            # else:
            #     from pgvector.sqlalchemy import Vector

                # embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
            embedding_dim = Column(BIGINT)
            embedding_model = Column(String)

            metadata_ = Column(MutableJson)

            # Add a datetime column, with default value as the current time
            created_at = Column(DateTime(timezone=True))

            def __repr__(self):
                return f"<Passage(passage_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

            def to_record(self):
                return Passage(
                    text=self.text,
                    embedding=self.embedding,
                    embedding_dim=self.embedding_dim,
                    embedding_model=self.embedding_model,
                    doc_id=self.doc_id,
                    user_id=self.user_id,
                    id=self.id,
                    data_source=self.data_source,
                    agent_id=self.agent_id,
                    metadata_=self.metadata_,
                    created_at=self.created_at,
                )

        """Create database model for table_name"""
        class_name = f"{table_name.capitalize()}Model" + dialect
        return create_or_get_model(class_name, DocumentModel, table_name)

        # pass
    elif storage_type == StorageType.KNOWLEDGE_BASE_PASSAGES:

        pass
    else:
        raise ValueError(f"storage type {storage_type} not implemented")



class StorageConnector:
    """Defines a DB connection that is user-specific to access data: Documents, Passages, Archival/Recall Memory"""

    # type: Type[Record]

    def __init__(
        self,
        storage_type: Union[StorageType.ARCHIVAL_MEMORY, StorageType.RECALL_MEMORY, StorageType.KNOWLEDGE_BASE, StorageType.KNOWLEDGE_BASE_DOCUMENTS,StorageType.KNOWLEDGE_BASE_PASSAGES],
        config: typeagentConfig,
        user_id,
        agent_id=None,
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.storage_type = storage_type

        # get object type
        if storage_type == StorageType.ARCHIVAL_MEMORY:
            # self.type = Passage
            self.table_name = ARCHIVAL_TABLE_NAME
        elif storage_type == StorageType.RECALL_MEMORY:
            # self.type = Message
            self.table_name = RECALL_TABLE_NAME
        elif storage_type == StorageType.KNOWLEDGE_BASE:
            # self.type = Document
            self.table_name == KNOWLEDGE_BASE
        elif storage_type == StorageType.KNOWLEDGE_BASE_DOCUMENTS:
            # self.type = Passage
            self.table_name = DOCUMENT_TABLE_NAME
        elif storage_type == StorageType.KNOWLEDGE_BASE_PASSAGES:
            # self.type = Passage
            self.table_name = PASSAGE_TABLE_NAME
        else:
            raise ValueError(f"Table type {storage_type} not implemented")
        printd(f"Using table name {self.table_name}")

        # setup base filters for agent-specific tables
        if self.storage_type == StorageType.ARCHIVAL_MEMORY or self.storage_type == StorageType.RECALL_MEMORY:
            # agent-specific table
            # assert agent_id is not None, "Agent ID must be provided for agent-specific tables"
            self.filters = {"user_id": self.user_id, "agent_id": self.agent_id}
        elif self.storage_type == StorageType.KNOWLEDGE_BASE_DOCUMENTS or self.storage_type == StorageType.KNOWLEDGE_BASE or self.storage_type == StorageType.KNOWLEDGE_BASE_PASSAGES:
            # setup base filters for user-specific tables
            # assert agent_id is None, "Agent ID must not be provided for user-specific tables"
            self.filters = {"user_id": self.user_id}
        else:
            raise ValueError(f"Table type {storage_type} not implemented")

    @staticmethod
    def get_storage_connector(
        storage_type: Union[StorageType.ARCHIVAL_MEMORY, StorageType.RECALL_MEMORY, StorageType.KNOWLEDGE_BASE, StorageType.KNOWLEDGE_BASE_DOCUMENTS,StorageType.KNOWLEDGE_BASE_PASSAGES],
        config: typeagentConfig,
        user_id,
        agent_id=None,
    ):
        

        if storage_type == StorageType.ARCHIVAL_MEMORY:
            # self.type = Passage
            storage_engine = config.archival_memory_storage_type
        elif storage_type == StorageType.RECALL_MEMORY:
            # self.type = Message
            storage_engine = config.recall_memory_storage_type
        elif storage_type == StorageType.KNOWLEDGE_BASE:
            # self.type = Document
            storage_engine = config.knowledge_base_storage_type
        elif storage_type == StorageType.KNOWLEDGE_BASE_DOCUMENTS:
            storage_engine = config.recall_memory_storage_type
        elif storage_type == StorageType.KNOWLEDGE_BASE_PASSAGES:
            # self.type = Passage
            storage_engine = config.knowledge_base_storage_type
        else:
            raise ValueError(f"storage type {storage_type} not implemented")
       

        if storage_engine == "postgres":
            from agent_store.sqldb.sqldbconnector import PostgresStorageConnector

            return PostgresStorageConnector(storage_type, config, user_id, agent_id)
        elif storage_engine == "chroma":
            from agent_store.vectorsdb.chroma import ChromaStorageConnector

            return ChromaStorageConnector(storage_type, config, user_id, agent_id)
        elif storage_type == "qdrant":
            from agent_store.vectorsdb.qdrant import QdrantStorageConnector

            return QdrantStorageConnector(storage_type, config, user_id, agent_id)
        # TODO: add back
        # elif storage_type == "lancedb":
        #    from agent_store.db import LanceDBConnector

        #    return LanceDBConnector(agent_config=agent_config, table_type=table_type)

        elif storage_type == "sqlite":
            from agent_store.sqldb.sqldbconnector import SQLLiteStorageConnector

            return SQLLiteStorageConnector(storage_type, config, user_id, agent_id)
        elif storage_type == "milvus":
            from agent_store.vectorsdb.milvus import MilvusStorageConnector
            return MilvusStorageConnector(storage_type, config, user_id, agent_id)
        else:
            raise NotImplementedError(f"Storage type {storage_type} not implemented")

    @staticmethod
    def get_archival_storage_connector(user_id, agent_id):
        config = typeagentConfig.load()
        return StorageConnector.get_storage_connector(StorageType.ARCHIVAL_MEMORY, config, user_id, agent_id)
    @staticmethod
    def get_knowledge_Base_storage_connector(user_id, agent_id):
        config = typeagentConfig.load()
        return StorageConnector.get_storage_connector(StorageType.KNOWLEDGE_BASE_PASSAGES, config, user_id, agent_id)

    @staticmethod
    def get_recall_storage_connector(user_id, agent_id):
        config = typeagentConfig.load()
        return StorageConnector.get_storage_connector(StorageType.RECALL_MEMORY, config, user_id, agent_id)

    @abstractmethod
    def get_filters(self, filters: Optional[Dict] = {}) -> Union[Tuple[list, dict], dict]:
        pass

    @abstractmethod
    def get_all_paginated(self, filters: Optional[Dict] = {}, page_size: int = 1000) -> Iterator[List[RecordType]]:
        pass

    @abstractmethod
    def get_all(self, filters: Optional[Dict] = {}, limit=10) -> List[RecordType]:
        pass

    @abstractmethod
    def get(self, id: uuid.UUID) -> Optional[RecordType]:
        pass

    @abstractmethod
    def size(self, filters: Optional[Dict] = {}) -> int:
        pass

    @abstractmethod
    def insert(self, record: RecordType):
        pass

    @abstractmethod
    def insert_many(self, records: List[RecordType], show_progress=False):
        pass

    @abstractmethod
    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[RecordType]:
        pass

    @abstractmethod
    def query_date(self, start_date, end_date):
        pass

    @abstractmethod
    def query_text(self, query):
        pass

    @abstractmethod
    def delete_table(self):
        pass

    @abstractmethod
    def delete(self, filters: Optional[Dict] = {}):
        pass

    @abstractmethod
    def save(self):
        pass
