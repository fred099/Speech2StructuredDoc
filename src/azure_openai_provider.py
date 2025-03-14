"""
Azure OpenAI Provider module for handling API requests to Azure OpenAI.
This module follows the Azure AI foundry approach for accessing Azure OpenAI services.
"""

import os
import logging
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from src.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT
)
from src.auth import get_credential
from src.openai_client import create_azure_openai_client, get_completion, AzureOpenAIClientConfig, ChatMessage

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class CompletionRequest(BaseModel):
    """
    Pydantic model for representing a completion request to Azure OpenAI.
    """
    prompt: str = Field(..., description="The user prompt")
    system_message: Optional[str] = Field(None, description="The system message")
    temperature: float = Field(0.7, description="The temperature for generation")
    max_tokens: int = Field(1000, description="The maximum number of tokens to generate")


class AzureOpenAIProvider:
    """
    Provider for Azure OpenAI API requests using the Azure AI foundry approach.
    """
    def __init__(self, use_token: bool = True, token: str = None):
        """
        Initialize the Azure OpenAI provider.
        
        Args:
            use_token (bool): Whether to use token-based authentication
            token (str): The Azure AD token to use for authentication (deprecated, kept for backward compatibility)
        """
        self.azure_openai_endpoint = AZURE_OPENAI_ENDPOINT
        self.azure_openai_api_version = AZURE_OPENAI_API_VERSION
        self.azure_openai_deployment = AZURE_OPENAI_DEPLOYMENT
        
        # Initialize the client based on authentication method
        if use_token:
            # Use token-based authentication with DefaultAzureCredential
            logger.info("Initializing Azure OpenAI client with token-based authentication")
            
            # Get credential using our robust authentication approach
            # Following organization's security standards: All Azure resource access MUST use DefaultAzureCredential
            # For local development, this will use Azure CLI login credentials
            # For production, this will use Managed Identity
            credential = get_credential()
            
            # Create client using our custom implementation to avoid proxies parameter issue
            self.client = create_azure_openai_client(
                endpoint=self.azure_openai_endpoint,
                api_version=self.azure_openai_api_version,
                deployment_name=self.azure_openai_deployment,
                credential=credential
            )
        else:
            # Use API key from environment variables
            logger.info("Initializing Azure OpenAI client with API key authentication")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                logger.warning("AZURE_OPENAI_API_KEY not found in environment variables")
            
            # Create client using our custom implementation to avoid proxies parameter issue
            self.client = create_azure_openai_client(
                endpoint=self.azure_openai_endpoint,
                api_version=self.azure_openai_api_version,
                deployment_name=self.azure_openai_deployment,
                api_key=api_key
            )
    
    def get_completion(self, request: CompletionRequest):
        """
        Get a completion from Azure OpenAI.
        
        Args:
            request (CompletionRequest): The completion request
            
        Returns:
            str: The completion text
        """
        try:
            # Prepare messages for the API call
            messages = []
            
            # Add system message if provided
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message})
            
            # Add user message
            messages.append({"role": "user", "content": request.prompt})
            
            logger.info(f"Sending completion request to Azure OpenAI deployment: {self.azure_openai_deployment}")
            
            # Use our custom get_completion function to avoid proxies parameter issue
            return get_completion(
                client=self.client,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
        except Exception as e:
            logger.error(f"Error getting completion from Azure OpenAI: {str(e)}")
            raise
