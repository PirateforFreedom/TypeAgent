import configparser
import os
from dataclasses import dataclass
from typing import Optional

from config import get_field, set_field
from constants import typeagent_DIR

SUPPORTED_AUTH_TYPES = ["bearer_token", "api_key"]


@dataclass
class typeagentCredentials:
    # credentials for typeagent
    credentials_path: str = os.path.join(typeagent_DIR, "credentials")

    # openai config
    openai_auth_type: str = "bearer_token"
    openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # gemini config
    google_ai_key: Optional[str] = None
    google_ai_service_endpoint: Optional[str] = None

    # anthropic config
    anthropic_key: Optional[str] = None

    # cohere config
    cohere_key: Optional[str] = None

    # azure config
    azure_auth_type: str = "api_key"
    azure_key: Optional[str] = None
    # base llm / model
    azure_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    # embeddings
    azure_embedding_api_key: Optional[str] = None
    azure_embedding_version: Optional[str] = None
    azure_embedding_endpoint: Optional[str] = None
    azure_embedding_deployment: Optional[str] = None

    # custom llm API config
    openllm_auth_type: Optional[str] = None
    openllm_key: Optional[str] = None

    # defulat llm api config
    # 

    @classmethod
    def load(cls) -> "typeagentCredentials":
        config = configparser.ConfigParser()

        # allow overriding with env variables
        if os.getenv("typeagent_CREDENTIALS_PATH"):
            credentials_path = os.getenv("typeagent_CREDENTIALS_PATH")
        else:
            credentials_path = typeagentCredentials.credentials_path

        if os.path.exists(credentials_path):
            # read existing credentials
            config.read(credentials_path)
            config_dict = {
                # openai
                "openai_auth_type": get_field(config, "openai", "auth_type"),
                "openai_key": get_field(config, "openai", "key"),
                # azure
                "azure_auth_type": get_field(config, "azure", "auth_type"),
                "azure_key": get_field(config, "azure", "key"),
                "azure_version": get_field(config, "azure", "version"),
                "azure_endpoint": get_field(config, "azure", "endpoint"),
                "azure_deployment": get_field(config, "azure", "deployment"),
                "azure_embedding_version": get_field(config, "azure", "embedding_version"),
                "azure_embedding_endpoint": get_field(config, "azure", "embedding_endpoint"),
                "azure_embedding_deployment": get_field(config, "azure", "embedding_deployment"),
                "azure_embedding_api_key": get_field(config, "azure", "embedding_api_key"),
                # gemini
                "google_ai_key": get_field(config, "google_ai", "key"),
                "google_ai_service_endpoint": get_field(config, "google_ai", "service_endpoint"),
                # anthropic
                "anthropic_key": get_field(config, "anthropic", "key"),
                # cohere
                "cohere_key": get_field(config, "cohere", "key"),
                # open llm
                "openllm_auth_type": get_field(config, "openllm", "auth_type"),
                "openllm_key": get_field(config, "openllm", "key"),
                # path
                "credentials_path": credentials_path,
            }
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
            return cls(**config_dict)

        # create new config
        config = cls(credentials_path=credentials_path)
        config.save()  # save updated config
        return config

    def save(self):
        pass

        config = configparser.ConfigParser()
        # openai config
        set_field(config, "openai", "auth_type", self.openai_auth_type)
        set_field(config, "openai", "key", self.openai_key)

        # azure config
        set_field(config, "azure", "auth_type", self.azure_auth_type)
        set_field(config, "azure", "key", self.azure_key)
        set_field(config, "azure", "version", self.azure_version)
        set_field(config, "azure", "endpoint", self.azure_endpoint)
        set_field(config, "azure", "deployment", self.azure_deployment)
        set_field(config, "azure", "embedding_version", self.azure_embedding_version)
        set_field(config, "azure", "embedding_endpoint", self.azure_embedding_endpoint)
        set_field(config, "azure", "embedding_deployment", self.azure_embedding_deployment)
        set_field(config, "azure", "embedding_api_key", self.azure_embedding_api_key)

        # gemini
        set_field(config, "google_ai", "key", self.google_ai_key)
        set_field(config, "google_ai", "service_endpoint", self.google_ai_service_endpoint)

        # anthropic
        set_field(config, "anthropic", "key", self.anthropic_key)

        # cohere
        set_field(config, "cohere", "key", self.cohere_key)

        # openllm config
        set_field(config, "openllm", "auth_type", self.openllm_auth_type)
        set_field(config, "openllm", "key", self.openllm_key)

        if not os.path.exists(typeagent_DIR):
            os.makedirs(typeagent_DIR, exist_ok=True)
        with open(self.credentials_path, "w", encoding="utf-8") as f:
            config.write(f)

    @staticmethod
    def exists():
        # allow overriding with env variables
        if os.getenv("typeagent_CREDENTIALS_PATH"):
            credentials_path = os.getenv("typeagent_CREDENTIALS_PATH")
        else:
            credentials_path = typeagentCredentials.credentials_path

        assert not os.path.isdir(credentials_path), f"Credentials path {credentials_path} cannot be set to a directory."
        return os.path.exists(credentials_path)
