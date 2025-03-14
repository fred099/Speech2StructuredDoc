import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Azure Storage Configuration
STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "speech2structureddoc-outputs")

# Azure Speech Service Configuration
SPEECH_REGION = os.getenv("SPEECH_REGION", "swedencentral")
SPEECH_ENDPOINT = os.getenv("SPEECH_ENDPOINT")
SPEECH_RESOURCE_NAME = os.getenv("SPEECH_RESOURCE_NAME")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Key Vault Configuration
KEY_VAULT_URL = os.getenv("KEY_VAULT_URL")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def get(key, default=None):
    """
    Get a configuration value from environment variables.
    In production, this would retrieve from Key Vault.
    
    Args:
        key: The configuration key to retrieve
        default: Default value if the key is not found
        
    Returns:
        The configuration value
    """
    return os.getenv(key, default)
