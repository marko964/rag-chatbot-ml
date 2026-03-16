from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_store"
    chroma_collection_name: str = "knowledge_base"

    # Database
    database_url: str = "sqlite:///./data/app.db"

    # Cal.com
    calcom_api_key: str = ""
    calcom_event_type_id: str = ""
    calcom_base_url: str = "https://api.cal.com/v2"
    calcom_booking_url: str = ""

    # Company
    company_name: str = "ML - Solutions"

    # Admin
    admin_api_key: str = "change-me"

    # CORS
    allowed_origins: str = "*"

    # Debug — set to False in production (hides /docs and /redoc)
    debug: bool = False

    def get_allowed_origins(self) -> List[str]:
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",")]

    model_config = {"env_file": ".env"}


settings = Settings()
