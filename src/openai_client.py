"""
Custom OpenAI client module to ensure proper token-based authentication with Azure OpenAI.
This module provides a clean implementation that avoids proxy-related issues.
"""

import logging
import os
import requests
import json
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)

class AzureOpenAIClientConfig(BaseModel):
    """Pydantic model for Azure OpenAI client configuration."""
    endpoint: str
    api_version: str
    deployment_name: str
    auth_type: str
    token: Optional[str] = None
    api_key: Optional[str] = None

class ChatMessage(BaseModel):
    """Pydantic model for chat message."""
    role: str
    content: str
    
    def dict(self):
        """Convert to dictionary for JSON serialization."""
        return {"role": self.role, "content": self.content}

class ChatCompletionRequest(BaseModel):
    """Pydantic model for chat completion request."""
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1000
    
def create_azure_openai_client(endpoint: str, api_version: str, deployment_name: str, 
                              credential: Optional[Any] = None, api_key: Optional[str] = None) -> AzureOpenAIClientConfig:
    """
    Create a minimal Azure OpenAI client with proper authentication, avoiding proxy-related issues.
    
    This function returns a Pydantic model that can be used to make requests to Azure OpenAI,
    completely bypassing the OpenAI SDK to avoid any proxy-related issues.
    
    Args:
        endpoint: The Azure OpenAI endpoint
        api_version: The API version to use
        deployment_name: The deployment name to use
        credential: The Azure credential to use for token-based authentication (optional)
        api_key: The API key to use for key-based authentication (optional)
        
    Returns:
        A Pydantic model for use with the custom get_completion function
    """
    if not endpoint or not api_version or not deployment_name:
        raise ValueError("endpoint, api_version, and deployment_name must be provided")
    
    # Ensure endpoint doesn't have trailing slash
    endpoint = endpoint.rstrip('/')
    
    # Determine which authentication method to use and create the client config
    if credential is not None:
        logger.info("Creating minimal Azure OpenAI client with token-based authentication")
        
        # Get token directly - following organization's security standards
        # Always use DefaultAzureCredential for Azure resource access
        # For Azure OpenAI, we need to use the correct scope
        token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        # Create client config with token-based authentication
        client_config = AzureOpenAIClientConfig(
            endpoint=endpoint,
            api_version=api_version,
            deployment_name=deployment_name,
            auth_type="token",
            token=token
        )
        
    elif api_key is not None:
        logger.info("Creating minimal Azure OpenAI client with API key authentication")
        
        # Create client config with API key authentication
        client_config = AzureOpenAIClientConfig(
            endpoint=endpoint,
            api_version=api_version,
            deployment_name=deployment_name,
            auth_type="api_key",
            api_key=api_key
        )
        
    else:
        raise ValueError("Either credential or api_key must be provided")
    
    return client_config

def get_completion(client: AzureOpenAIClientConfig, 
                  messages: List[Dict[str, str]], temperature: float = 0.7, 
                  max_tokens: int = 1000) -> str:
    """
    Get a completion from Azure OpenAI using direct HTTP requests.
    
    This function completely bypasses the OpenAI SDK to avoid any proxy-related issues.
    
    Args:
        client: The client config from create_azure_openai_client
        messages: The messages for the chat completion
        temperature: The temperature for generation
        max_tokens: The maximum number of tokens to generate
        
    Returns:
        The completion text
    """
    try:
        # Construct the API URL - ensure no double slashes in the URL
        base_endpoint = client.endpoint.rstrip('/')
        url = f"{base_endpoint}/openai/deployments/{client.deployment_name}/chat/completions?api-version={client.api_version}"
        
        # Prepare the request headers based on auth type
        headers = {
            "Content-Type": "application/json"
        }
        
        if client.auth_type == "token":
            headers["Authorization"] = f"Bearer {client.token}"
        else:  # api_key
            headers["api-key"] = client.api_key
        
        # Prepare the request payload
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make the API request
        logger.info(f"Sending completion request to {url}")
        response = requests.post(url, headers=headers, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            error_msg = f"{response.status_code} {response.reason} for url: {url}"
            logger.error(f"Error getting completion from Azure OpenAI: {error_msg}")
            logger.error(f"Response content: {response.text}")
            response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Extract and return the completion text as a string
        # This is what the SpeakerAnalyzer expects - a raw text response that it can parse as JSON
        completion_text = result["choices"][0]["message"]["content"]
        return completion_text
        
    except Exception as e:
        logger.error(f"Error getting completion from Azure OpenAI: {str(e)}")
        raise
