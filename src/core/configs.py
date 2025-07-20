import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    load_dotenv()
    API_V1_STR: str = '/api/v1'
    CACHE_TIMEOUT: int = int(os.getenv('CACHE_TIMEOUT', 300))

settings: Settings = Settings()
