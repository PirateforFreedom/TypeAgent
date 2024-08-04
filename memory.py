import datetime
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from metadata import MetadataStore
from config import typeagentConfig
import numpy
import typer
from constants import MESSAGE_SUMMARY_REQUEST_ACK, MESSAGE_SUMMARY_WARNING_FRAC
from data_types import AgentState, Message, Passage,ToolCall
from embeddings import embedding_model, generate_passages, query_embedding
from llm_api.llm_api_tools import create
from prompts.gpt_summarize import SYSTEM as SUMMARY_PROMPT_SYSTEM
from utils import (
    count_tokens,
    extract_date_from_timestamp,
    get_local_time,
    printd,
    validate_date_format,
    create_uuid_from_string,
)
from models.pydantic_models import (
    HumanModel,
    JobModel,
    JobStatus,
    PersonaModel,
    ToolModel,
    SystemPromptModel,
    SourceModel,
    DocumentModel,
    PassageModel,
    RecallMemoryModel,
    ArchivalMemoryModel,
)

# from llama_index import Document
# from llama_index.node_parser import SimpleNodeParser


class CoreMemory(object):
    """Held in-context inside the system message

    Core Memory: Refers to the system block, which provides essential, foundational context to the AI.
    This includes the persona information, essential user details,
    and any other baseline data you deem necessary for the AI's basic functioning.
    """

    def __init__(self, persona=None, human=None, persona_char_limit=None, human_char_limit=None, archival_memory_exists=True):
        self.persona = persona
        self.human = human
        self.persona_char_limit = persona_char_limit
        self.human_char_limit = human_char_limit

        # affects the error message the AI will see on overflow inserts
        self.archival_memory_exists = archival_memory_exists

    def __repr__(self) -> str:
        return f"\n### CORE MEMORY ###" + f"\n=== Persona ===\n{self.persona}" + f"\n\n=== Human ===\n{self.human}"

    def to_dict(self):
        return {
            "persona": self.persona,
            "human": self.human,
        }

    @classmethod
    def load(cls, state):
        return cls(state["persona"], state["human"])

    def edit_persona(self, new_persona):
        if self.persona_char_limit and len(new_persona) > self.persona_char_limit:
            error_msg = f"Edit failed: Exceeds {self.persona_char_limit} character limit (requested {len(new_persona)})."
            if self.archival_memory_exists:
                error_msg = f"{error_msg} Consider summarizing existing core memories in 'persona' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            raise ValueError(error_msg)

        self.persona = new_persona
        return len(self.persona)

    def edit_human(self, new_human):
        if self.human_char_limit and len(new_human) > self.human_char_limit:
            error_msg = f"Edit failed: Exceeds {self.human_char_limit} character limit (requested {len(new_human)})."
            if self.archival_memory_exists:
                error_msg = f"{error_msg} Consider summarizing existing core memories in 'human' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            raise ValueError(error_msg)

        self.human = new_human
        return len(self.human)

    def edit(self, field, content):
        if field == "persona":
            return self.edit_persona(content)
        elif field == "human":
            return self.edit_human(content)
        else:
            raise KeyError(f'No memory section named {field} (must be either "persona" or "human")')

    def edit_append(self, field, content, sep="\n"):
        if field == "persona":
            new_content = self.persona + sep + content
            return self.edit_persona(new_content)
        elif field == "human":
            new_content = self.human + sep + content
            return self.edit_human(new_content)
        else:
            raise KeyError(f'No memory section named {field} (must be either "persona" or "human")')

    def edit_replace(self, field, old_content, new_content):
        if len(old_content) == 0:
            raise ValueError("old_content cannot be an empty string (must specify old_content to replace)")

        if field == "persona":
            if old_content in self.persona:
                new_persona = self.persona.replace(old_content, new_content)
                return self.edit_persona(new_persona)
            else:
                raise ValueError("Content not found in persona (make sure to use exact string)")
        elif field == "human":
            if old_content in self.human:
                new_human = self.human.replace(old_content, new_content)
                return self.edit_human(new_human)
            else:
                raise ValueError("Content not found in human (make sure to use exact string)")
        else:
            raise KeyError(f'No memory section named {field} (must be either "persona" or "human")')


def _format_summary_history(message_history: List[Message]):
    # TODO use existing prompt formatters for this (eg ChatML)
    return "\n".join([f"{m.role}: {m.text}" for m in message_history])


def summarize_messages(
    agent_state: AgentState,
    message_sequence_to_summarize: List[Message],
    insert_acknowledgement_assistant_message: bool = True,
):
    """Summarize a message sequence using GPT"""
    # we need the context_window
    context_window = agent_state.llm_config.context_window

    summary_prompt = SUMMARY_PROMPT_SYSTEM
    summary_input = _format_summary_history(message_sequence_to_summarize)
    summary_input_tkns = count_tokens(summary_input)
    if summary_input_tkns > MESSAGE_SUMMARY_WARNING_FRAC * context_window:
        trunc_ratio = (MESSAGE_SUMMARY_WARNING_FRAC * context_window / summary_input_tkns) * 0.8  # For good measure...
        cutoff = int(len(message_sequence_to_summarize) * trunc_ratio)
        summary_input = str(
            [summarize_messages(agent_state, message_sequence_to_summarize=message_sequence_to_summarize[:cutoff])]
            + message_sequence_to_summarize[cutoff:]
        )

    dummy_user_id = uuid.uuid4()
    dummy_agent_id = uuid.uuid4()
    message_sequence = []
    message_sequence.append(Message(user_id=dummy_user_id, agent_id=dummy_agent_id, role="system", text=summary_prompt))
    if insert_acknowledgement_assistant_message:
        message_sequence.append(Message(user_id=dummy_user_id, agent_id=dummy_agent_id, role="assistant", text=MESSAGE_SUMMARY_REQUEST_ACK))
    message_sequence.append(Message(user_id=dummy_user_id, agent_id=dummy_agent_id, role="user", text=summary_input))

    response = create(
        llm_config=agent_state.llm_config,
        user_id=agent_state.user_id,
        messages=message_sequence,
    )

    printd(f"summarize_messages gpt reply: {response.choices[0]}")
    reply = response.choices[0].message.content
    return reply


class ArchivalMemory(ABC):
    @abstractmethod
    def insert(self, memory_string: str):
        """Insert new archival memory

        :param memory_string: Memory string to insert
        :type memory_string: str
        """

    @abstractmethod
    def search(self, query_string, count=None, start=None) -> Tuple[List[str], int]:
        """Search archival memory

        :param query_string: Query string
        :type query_string: str
        :param count: Number of results to return (None for all)
        :type count: Optional[int]
        :param start: Offset to start returning results from (None if 0)
        :type start: Optional[int]

        :return: Tuple of (list of results, total number of results)
        """

    @abstractmethod
    def __repr__(self) -> str:
        pass


class RecallMemory(ABC):
    @abstractmethod
    def text_search(self, query_string, count=None, start=None):
        """Search messages that match query_string in recall memory"""

    @abstractmethod
    def date_search(self, start_date, end_date, count=None, start=None):
        """Search messages between start_date and end_date in recall memory"""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def insert(self, message: Message):
        """Insert message into recall memory"""


class DummyRecallMemory(RecallMemory):
    """Dummy in-memory version of a recall memory database (eg run on MongoDB)

    Recall memory here is basically just a full conversation history with the user.
    Queryable via string matching, or date matching.

    Recall Memory: The AI's capability to search through past interactions,
    effectively allowing it to 'remember' prior engagements with a user.
    """

    def __init__(self, message_database=None, restrict_search_to_summaries=False):
        self._message_logs = [] if message_database is None else message_database  # consists of full message dicts

        # If true, the pool of messages that can be queried are the automated summaries only
        # (generated when the conversation window needs to be shortened)
        self.restrict_search_to_summaries = restrict_search_to_summaries

    def __len__(self):
        return len(self._message_logs)

    def __repr__(self) -> str:
        # don't dump all the conversations, just statistics
        system_count = user_count = assistant_count = function_count = other_count = 0
        for msg in self._message_logs:
            role = msg["message"]["role"]
            if role == "system":
                system_count += 1
            elif role == "user":
                user_count += 1
            elif role == "assistant":
                assistant_count += 1
            elif role == "function":
                function_count += 1
            else:
                other_count += 1
        memory_str = (
            f"Statistics:"
            + f"\n{len(self._message_logs)} total messages"
            + f"\n{system_count} system"
            + f"\n{user_count} user"
            + f"\n{assistant_count} assistant"
            + f"\n{function_count} function"
            + f"\n{other_count} other"
        )
        return f"\n### RECALL MEMORY ###" + f"\n{memory_str}"

    def insert(self, message):
        raise NotImplementedError("This should be handled by the PersistenceManager, recall memory is just a search layer on top")

    def text_search(self, query_string, count=None, start=None):
        # in the dummy version, run an (inefficient) case-insensitive match search
        message_pool = [d for d in self._message_logs if d["message"]["role"] not in ["system", "function"]]

        printd(
            f"recall_memory.text_search: searching for {query_string} (c={count}, s={start}) in {len(self._message_logs)} total messages"
        )
        matches = [
            d for d in message_pool if d["message"]["content"] is not None and query_string.lower() in d["message"]["content"].lower()
        ]
        printd(f"recall_memory - matches:\n{matches[start:start+count]}")

        # start/count support paging through results
        if start is not None and count is not None:
            return matches[start : start + count], len(matches)
        elif start is None and count is not None:
            return matches[:count], len(matches)
        elif start is not None and count is None:
            return matches[start:], len(matches)
        else:
            return matches, len(matches)

    def date_search(self, start_date, end_date, count=None, start=None):
        message_pool = [d for d in self._message_logs if d["message"]["role"] not in ["system", "function"]]

        # First, validate the start_date and end_date format
        if not validate_date_format(start_date) or not validate_date_format(end_date):
            raise ValueError("Invalid date format. Expected format: YYYY-MM-DD")

        # Convert dates to datetime objects for comparison
        start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        # Next, match items inside self._message_logs
        matches = [
            d
            for d in message_pool
            if start_date_dt <= datetime.datetime.strptime(extract_date_from_timestamp(d["timestamp"]), "%Y-%m-%d") <= end_date_dt
        ]

        # start/count support paging through results
        start = int(start) if start is None else start
        count = int(count) if count is None else count
        if start is not None and count is not None:
            return matches[start : start + count], len(matches)
        elif start is None and count is not None:
            return matches[:count], len(matches)
        elif start is not None and count is None:
            return matches[start:], len(matches)
        else:
            return matches, len(matches)


class BaseRecallMemory(RecallMemory):
    """Recall memory based on base functions implemented by storage connectors"""

    def __init__(self, agent_state, restrict_search_to_summaries=False):
        # If true, the pool of messages that can be queried are the automated summaries only
        # (generated when the conversation window needs to be shortened)
        self.restrict_search_to_summaries = restrict_search_to_summaries
        # from agent_store.storage import StorageConnector

        self.agent_state = agent_state
        self.config = typeagentConfig.load()
        self.recallmemory_ms = MetadataStore(self.config)
        # create embedding model
        # self.embed_model = embedding_model(agent_state.embedding_config)
        # self.embedding_chunk_size = agent_state.embedding_config.embedding_chunk_size

        # create storage backend
        # self.storage = StorageConnector.get_recall_storage_connector(user_id=agent_state.user_id, agent_id=agent_state.id)
        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}


    def get_one(self,id:uuid.UUID):
        results = self.recallmemory_ms.get_recallmemory(id=id)
        

        print(results.tool_calls)
        if results.tool_calls:
                tmeptoolcalls=[
                    ToolCall(id =tool_call["id"],tool_call_type = tool_call["type"],function = tool_call["function"]) for tool_call in results.tool_calls
                    ],
        else:
                tmeptoolcalls=None
        

       
        recallmemorymessage=Message(
                id=results.id,
                 user_id=results.user_id,
                 agent_id=results.agent_id,
                 role=results.role,
                 text=results.text,
                 model=results.model,
                 name=results.name,
                 tool_calls=tmeptoolcalls,
                 tool_call_id=results.tool_call_id,
                 embedding=results.embedding,
                 embedding_model=results.embedding_model,
                 embedding_dim=1,# TODO later  add real embedding dim or remove
                 created_at=results.created_at,
                #  user_status="on",
            )
        # results_json = [message.to_openai_dict() for message in results]
        return recallmemorymessage
    


    def get_all(self, start=0, count=None):
        results = self.storage.get_all(start, count)
        results_json = [message.to_openai_dict() for message in results]
        return results_json, len(results)

    def text_search(self, query_string, count=None, start=None):
        # results = self.storage.query_text(query_string, count, start)
        results = self.recallmemory_ms.query_text_recallmemory(query_string, count, start)
        results_json=[]
        for message in results:

             if message.tool_calls:
                   tmeptoolcalls=[
                    ToolCall(id =tool_call["id"],tool_call_type = tool_call["type"],function = tool_call["function"]) for tool_call in results.tool_calls
                    ],
             else:
                tmeptoolcalls=None
             tempmes=Message(
                id=message.id,
                 user_id=message.user_id,
                 agent_id=message.agent_id,
                 role=message.role,
                 text=message.text,
                 model=message.model,
                 name=message.name,
                 tool_calls=tmeptoolcalls,
                 tool_call_id=message.tool_call_id,
                 embedding=message.embedding,
                 embedding_model=message.embedding_model,
                 embedding_dim=1,# TODO later  add real embedding dim or remove
                 created_at=message.created_at,
                #  user_status="on",
               ) 
             results_json.append(tempmes.to_json())
        # results_json = [message.to_openai_dict() for message in results]
        # results_json = [message.to_openai_dict_search_results() for message in results]
        return results_json, len(results)

    def date_search(self, start_date, end_date, count=None, start=None):
        results = self.recallmemory_ms.query_date_recallmemory(start_date, end_date, count, start)
        results_json=[]
        for message in results:

             if message.tool_calls:
                   tmeptoolcalls=[
                    ToolCall(id =tool_call["id"],tool_call_type = tool_call["type"],function = tool_call["function"]) for tool_call in results.tool_calls
                    ],
             else:
                tmeptoolcalls=None
             tempmes=Message(
                id=message.id,
                 user_id=message.user_id,
                 agent_id=message.agent_id,
                 role=message.role,
                 text=message.text,
                 model=message.model,
                 name=message.name,
                 tool_calls=tmeptoolcalls,
                 tool_call_id=message.tool_call_id,
                 embedding=message.embedding,
                 embedding_model=message.embedding_model,
                 embedding_dim=1,# TODO later  add real embedding dim or remove
                 created_at=message.created_at,
                #  user_status="on",
               ) 
             results_json.append(tempmes.to_json())
        # results_json = [message.to_openai_dict() for message in results]
        
        return results_json, len(results)

    def __repr__(self) -> str:
        total = self.recallmemory_ms.size_RecallMemory(user_id=self.agent_state.user_id,agent_id=self.agent_state.id)
        system_count = self.recallmemory_ms.size_RecallMemory(user_id=self.agent_state.user_id,agent_id=self.agent_state.id,filters={"role": "system"})
        user_count = self.recallmemory_ms.size_RecallMemory(user_id=self.agent_state.user_id,agent_id=self.agent_state.id,filters={"role": "user"})
        assistant_count = self.recallmemory_ms.size_RecallMemory(user_id=self.agent_state.user_id,agent_id=self.agent_state.id,filters={"role": "assistant"})
        function_count = self.recallmemory_ms.size_RecallMemory(user_id=self.agent_state.user_id,agent_id=self.agent_state.id,filters={"role": "function"})
        other_count = total - (system_count + user_count + assistant_count + function_count)

        memory_str = (
            f"Statistics:"
            + f"\n{total} total messages"
            + f"\n{system_count} system"
            + f"\n{user_count} user"
            + f"\n{assistant_count} assistant"
            + f"\n{function_count} function"
            + f"\n{other_count} other"
        )
        return f"\n### RECALL MEMORY ###" + f"\n{memory_str}"

    def insert(self, message: Message):
        if message.tool_call_id :
                tool_call_id=message.tool_call_id
        else:
                tool_call_id=None
            
        if message.tool_calls:
                tmeptoolcalls=[tool_call.to_dict() for tool_call in message.tool_calls]
        else:
                tmeptoolcalls=None

        if message.embedding:
                tmepembedding=(numpy.round(message.embedding.astype(numpy.float64), 1)).tolist()
        else:
                tmepembedding=None
        recallmemory=RecallMemoryModel(
                id=message.id,
                 user_id=message.user_id,
                 agent_id=message.agent_id,
                 role=message.role,
                 text=message.text,
                 model=message.model,
                 name=message.name,
                 tool_calls=tmeptoolcalls,
                 tool_call_id=tool_call_id,
                 embedding=tmepembedding,
                 embedding_model=message.embedding_model,
                 embedding_dim=str(message.embedding_dim),
                 created_at=message.created_at,
                 user_status="on",
            )
        self.recallmemory_ms.add_recallmemory(recallmemory)
    def update(self, message: Message):
        if message.tool_call_id :
                tool_call_id=message.tool_call_id
        else:
                tool_call_id=None
            
        if message.tool_calls:
                tmeptoolcalls=[tool_call.to_dict() for tool_call in message.tool_calls]
        else:
                tmeptoolcalls=None

        if message.embedding:
                tmepembedding=(numpy.round(message.embedding.astype(numpy.float64), 1)).tolist()
        else:
                tmepembedding=None
        recallmemory=RecallMemoryModel(
                id=message.id,
                 user_id=message.user_id,
                 agent_id=message.agent_id,
                 role=message.role,
                 text=message.text,
                 model=message.model,
                 name=message.name,
                 tool_calls=tmeptoolcalls,
                 tool_call_id=tool_call_id,
                 embedding=tmepembedding,
                 embedding_model=message.embedding_model,
                 embedding_dim=str(message.embedding_dim),
                 created_at=message.created_at,
                 user_status="on",
            )
        self.recallmemory_ms.update_recallmemroy(user_id=message.user_id,agent_id=message.agent_id,message_id=message.id,recallmemoy=recallmemory)

    def insert_many(self, messages: List[Message]):
        for messageone in messages:
            
            if messageone.tool_call_id :
                tool_call_id=messageone.tool_call_id
            else:
                tool_call_id=None
            
            if messageone.tool_calls:
                tmeptoolcalls=[tool_call.to_dict() for tool_call in messageone.tool_calls]
            else:
                tmeptoolcalls=None

            if messageone.embedding:
                tmepembedding=(numpy.round(messageone.embedding.astype(numpy.float64), 1)).tolist()
            else:
                tmepembedding=None
            recallmemory=RecallMemoryModel(
                id=messageone.id,
                 user_id=messageone.user_id,
                 agent_id=messageone.agent_id,
                 role=messageone.role,
                 text=messageone.text,
                 model=messageone.model,
                 name=messageone.name,
                 tool_calls=tmeptoolcalls,
                 tool_call_id=tool_call_id,
                 embedding=tmepembedding,
                 embedding_model=messageone.embedding_model,
                 embedding_dim=str(messageone.embedding_dim),
                 created_at=messageone.created_at,
                 user_status="on",
            )
            self.recallmemory_ms.add_recallmemory(recallmemory)
        # self.storage.insert_many(messages)

    def save(self):
        self.storage.save()

    def __len__(self):

        print(str(self.agent_state.user_id))
        print(str(self.agent_state.id))
        recallmemsum=self.recallmemory_ms.get_recallmemorybyagentid(user_id=self.agent_state.user_id,agent_id=self.agent_state.id)
        print(recallmemsum)
        if recallmemsum:
            return len(recallmemsum)

        return 0


class EmbeddingArchivalMemory(ArchivalMemory):
    """Archival memory with embedding based search"""

    def __init__(self, agent_state: AgentState, top_k: Optional[int] = 100):
        """Init function for archival memory

        :param archival_memory_database: name of dataset to pre-fill archival with
        :type archival_memory_database: str
        """
        from agent_store.storage import StorageConnector

        self.top_k = top_k
        self.agent_state = agent_state

        # create embedding model
        self.embed_model = embedding_model(agent_state.embedding_config)
        self.embedding_chunk_size = agent_state.embedding_config.embedding_chunk_size
        assert self.embedding_chunk_size, f"Must set {agent_state.embedding_config.embedding_chunk_size}"
        self.config = typeagentConfig.load()
        self.archivememory_ms = MetadataStore(self.config)
        # create storage backend
        self.storage = StorageConnector.get_archival_storage_connector(user_id=agent_state.user_id, agent_id=agent_state.id)
        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def create_passage(self, text, embedding):
        return Passage(
            user_id=self.agent_state.user_id,
            agent_id=self.agent_state.id,
            text=text,
            embedding=embedding,
            embedding_dim=self.agent_state.embedding_config.embedding_dim,
            embedding_model=self.agent_state.embedding_config.embedding_model,
        )

    def save(self):
        """Save the index to disk"""
        self.storage.save()

    def get_allachivememroy(self,user_id: uuid.UUID, agent_id: uuid.UUID)-> List[ArchivalMemoryModel]:
        return self.archivememory_ms.get_all_archivememory(user_id=user_id, agent_id=agent_id)

    def insert(self, memory_string, user_id: uuid.UUID, agent_id: uuid.UUID) ->List[uuid.UUID]:
        """Embed and save memory string"""

        if not isinstance(memory_string, str):
            raise TypeError("memory must be a string")

        try:
            passages = []
            passages_ids=[]
            passage_count=0
            for passage_text, passage_metadata in generate_passages(documents=memory_string, chunk_size=self.config.default_embedding_config.embedding_chunk_size):

            # for some reason, llama index parsers sometimes return empty strings
                if len(passage_text) == 0:
                    # print("passage text is 0")
                    typer.secho(
                    f"Warning: embedding text returned empty string, skipping insert of passage with metadata '{passage_metadata}' into VectorDB. You can usually ignore this warning.",
                    fg=typer.colors.YELLOW,
                   )
                    continue

            # get embedding
                try:
                   
                #    print("passage_text:")
                #    print(passage_text)
                   embed_model = embedding_model(self.config.default_embedding_config)

                   embedding = embed_model.embed_documents([passage_text])
                except Exception as e:
                    typer.secho(
                    f"Warning: Failed to get embedding for {passage_text} (error: {str(e)}), skipping insert into VectorDB.",
                    fg=typer.colors.YELLOW,
                    )
                    continue
                passagetempid=create_uuid_from_string(f"{str(agent_id)}_{passage_text}")
                # passages_ids
                print(str(passagetempid))
                archivepassage = ArchivalMemoryModel(
                   user_id=user_id,
                   id=passagetempid,
                   agent_id=agent_id,
                   text=passage_text,
                   embedding=embedding[0],
                   embedding_model=self.config.default_embedding_config.embedding_model,
                   user_status="on",
                 )
                
                 
                self.archivememory_ms.add_ArchivalMemory(archivepassage)
                passages_ids.append(passagetempid)
                passageloca=Passage(
                    id=passagetempid,
                    text=passage_text,
                    doc_id=agent_id,
                    metadata_=passage_metadata,
                    user_id=user_id,
                    data_source="archive_memory",
                    embedding_model=self.config.default_embedding_config.embedding_model,
                    embedding=embedding[0],
                )
                passages.append(passageloca)
         
                if len(passages) >= 100:
                # insert passages into passage store
                     self.storage.insert_many(passages)
                     passage_count += len(passages)
                     passages = []
            if len(passages) > 0:
        # insert passages into passage store
                self.storage.insert_many(passages)
                passage_count += len(passages)
            return passages_ids
            # for text in parse_and_chunk_text(memory_string, self.embedding_chunk_size):
            #     embedding = self.embed_model.get_text_embedding(text)
            #     # fixing weird bug where type returned isn't a list, but instead is an object
            #     # eg: embedding={'object': 'list', 'data': [{'object': 'embedding', 'embedding': [-0.0071973633, -0.07893023,
            #     if isinstance(embedding, dict):
            #         try:
            #             embedding = embedding["data"][0]["embedding"]
            #         except (KeyError, IndexError):
            #             # TODO as a fallback, see if we can find any lists in the payload
            #             raise TypeError(
            #                 f"Got back an unexpected payload from text embedding function, type={type(embedding)}, value={embedding}"
            #             )
            #     passages.append(self.create_passage(text, embedding))

            # # grab the return IDs before the list gets modified
            # ids = [str(p.id) for p in passages]

            # # insert passages
            # self.storage.insert_many(passages)

            # if return_ids:
            #     return ids
            # else:
            #     return True

        except Exception as e:
            print("Archival insert error", e)
            raise e

    def search(self, query_string, count=None, start=None):
        """Search query string"""
        if not isinstance(query_string, str):
            return TypeError("query must be a string")

        try:
            if query_string not in self.cache:
                # self.cache[query_string] = self.retriever.retrieve(query_string)
                query_vec = query_embedding(self.embed_model, query_string)
                self.cache[query_string] = self.storage.query(query_string, query_vec, top_k=self.top_k)

            start = int(start if start else 0)
            count = int(count if count else self.top_k)
            end = min(count + start, len(self.cache[query_string]))

            results = self.cache[query_string][start:end]
            results = [{"timestamp": get_local_time(), "content": node.text} for node in results]
            return results, len(results)
        except Exception as e:
            print("Archival search error", e)
            raise e

    def __repr__(self) -> str:
        limit = 10
        passages = []
        for passage in list(self.storage.get_all(limit=limit)):  # TODO: only get first 10
            passages.append(str(passage.text))
        memory_str = "\n".join(passages)
        return f"\n### ARCHIVAL MEMORY ###" + f"\n{memory_str}" + f"\nSize: {self.storage.size()}"

    def __len__(self):
        return self.storage.size()
class EmbeddingKnoledgeBase(ArchivalMemory):
    """Archival memory with embedding based search"""

    def __init__(self, agent_state: AgentState, top_k: Optional[int] = 100):
        """Init function for archival memory

        :param archival_memory_database: name of dataset to pre-fill archival with
        :type archival_memory_database: str
        """
        from agent_store.storage import StorageConnector

        self.top_k = top_k
        self.agent_state = agent_state

        # create embedding model
        self.embed_model = embedding_model(agent_state.embedding_config)
        self.embedding_chunk_size = agent_state.embedding_config.embedding_chunk_size
        assert self.embedding_chunk_size, f"Must set {agent_state.embedding_config.embedding_chunk_size}"
        self.config = typeagentConfig.load()
        self.archivememory_ms = MetadataStore(self.config)
        # create storage backend
        self.storage = StorageConnector.get_knowledge_Base_storage_connector(user_id=agent_state.user_id, agent_id=agent_state.id)
        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def create_passage(self, text, embedding):
        return Passage(
            user_id=self.agent_state.user_id,
            agent_id=self.agent_state.id,
            text=text,
            embedding=embedding,
            embedding_dim=self.agent_state.embedding_config.embedding_dim,
            embedding_model=self.agent_state.embedding_config.embedding_model,
        )

    def save(self):
        """Save the index to disk"""
        self.storage.save()

    def get_allachivememroy(self,user_id: uuid.UUID, agent_id: uuid.UUID)-> List[ArchivalMemoryModel]:
        return self.archivememory_ms.get_all_archivememory(user_id=user_id, agent_id=agent_id)

    def insert(self, memory_string, user_id: uuid.UUID, agent_id: uuid.UUID) ->List[uuid.UUID]:
        """Embed and save memory string"""

        if not isinstance(memory_string, str):
            raise TypeError("memory must be a string")

        try:
            passages = []
            passages_ids=[]
            passage_count=0
            for passage_text, passage_metadata in generate_passages(documents=memory_string, chunk_size=self.config.default_embedding_config.embedding_chunk_size):

            # for some reason, llama index parsers sometimes return empty strings
                if len(passage_text) == 0:
                    # print("passage text is 0")
                    typer.secho(
                    f"Warning: embedding text returned empty string, skipping insert of passage with metadata '{passage_metadata}' into VectorDB. You can usually ignore this warning.",
                    fg=typer.colors.YELLOW,
                   )
                    continue

            # get embedding
                try:
                   
                #    print("passage_text:")
                #    print(passage_text)
                   embed_model = embedding_model(self.config.default_embedding_config)

                   embedding = embed_model.embed_documents([passage_text])
                except Exception as e:
                    typer.secho(
                    f"Warning: Failed to get embedding for {passage_text} (error: {str(e)}), skipping insert into VectorDB.",
                    fg=typer.colors.YELLOW,
                    )
                    continue
                passagetempid=create_uuid_from_string(f"{str(agent_id)}_{passage_text}")
                # passages_ids
                print(str(passagetempid))
                archivepassage = ArchivalMemoryModel(
                   user_id=user_id,
                   id=passagetempid,
                   agent_id=agent_id,
                   text=passage_text,
                   embedding=embedding[0],
                   embedding_model=self.config.default_embedding_config.embedding_model,
                   user_status="on",
                 )
                
                 
                self.archivememory_ms.add_ArchivalMemory(archivepassage)
                passages_ids.append(passagetempid)
                passageloca=Passage(
                    id=passagetempid,
                    text=passage_text,
                    doc_id=agent_id,
                    metadata_=passage_metadata,
                    user_id=user_id,
                    data_source="archive_memory",
                    embedding_model=self.config.default_embedding_config.embedding_model,
                    embedding=embedding[0],
                )
                passages.append(passageloca)
         
                if len(passages) >= 100:
                # insert passages into passage store
                     self.storage.insert_many(passages)
                     passage_count += len(passages)
                     passages = []
            if len(passages) > 0:
        # insert passages into passage store
                self.storage.insert_many(passages)
                passage_count += len(passages)
            return passages_ids
            # for text in parse_and_chunk_text(memory_string, self.embedding_chunk_size):
            #     embedding = self.embed_model.get_text_embedding(text)
            #     # fixing weird bug where type returned isn't a list, but instead is an object
            #     # eg: embedding={'object': 'list', 'data': [{'object': 'embedding', 'embedding': [-0.0071973633, -0.07893023,
            #     if isinstance(embedding, dict):
            #         try:
            #             embedding = embedding["data"][0]["embedding"]
            #         except (KeyError, IndexError):
            #             # TODO as a fallback, see if we can find any lists in the payload
            #             raise TypeError(
            #                 f"Got back an unexpected payload from text embedding function, type={type(embedding)}, value={embedding}"
            #             )
            #     passages.append(self.create_passage(text, embedding))

            # # grab the return IDs before the list gets modified
            # ids = [str(p.id) for p in passages]

            # # insert passages
            # self.storage.insert_many(passages)

            # if return_ids:
            #     return ids
            # else:
            #     return True

        except Exception as e:
            print("Archival insert error", e)
            raise e

    def search(self, query_string, count=None, start=None):
        """Search query string"""
        if not isinstance(query_string, str):
            return TypeError("query must be a string")

        try:
            if query_string not in self.cache:
                # self.cache[query_string] = self.retriever.retrieve(query_string)
                query_vec = query_embedding(self.embed_model, query_string)
                self.cache[query_string] = self.storage.query(query_string, query_vec, top_k=self.top_k)

            start = int(start if start else 0)
            count = int(count if count else self.top_k)
            end = min(count + start, len(self.cache[query_string]))

            results = self.cache[query_string][start:end]
            results = [{"timestamp": get_local_time(), "content": node.text} for node in results]
            return results, len(results)
        except Exception as e:
            print("Archival search error", e)
            raise e

    def __repr__(self) -> str:
        limit = 10
        passages = []
        for passage in list(self.storage.get_all(limit=limit)):  # TODO: only get first 10
            passages.append(str(passage.text))
        memory_str = "\n".join(passages)
        return f"\n### ARCHIVAL MEMORY ###" + f"\n{memory_str}" + f"\nSize: {self.storage.size()}"

    def __len__(self):
        return self.storage.size()
