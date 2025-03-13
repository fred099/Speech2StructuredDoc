#!/usr/bin/env python
"""
Comprehensive test script for Azure OpenAI service using hybrid authentication.
Follows the WebScraper-RAG authentication pattern with proper fallback mechanisms.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential
import requests

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

class AzureOpenAIAuthenticator:
    """Class to handle Azure OpenAI authentication with proper fallback mechanisms."""
    
    def __init__(self):
        self.credential = None
        self.api_key = AZURE_OPENAI_API_KEY
        self.token = None
        self.auth_method = None
    
    def authenticate(self):
        """Try different authentication methods in order of preference."""
        auth_methods = [
            self._try_default_azure_credential,
            self._try_azure_cli_credential,
            self._try_api_key
        ]
        
        for method in auth_methods:
            success = method()
            if success:
                return True
        
        print("All authentication methods failed.")
        return False
    
    def _try_default_azure_credential(self):
        """Try authenticating with DefaultAzureCredential."""
        try:
            print("Trying DefaultAzureCredential...")
            credential = DefaultAzureCredential(additionally_allowed_tenants=["*"])
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            self.credential = credential
            self.token = token
            self.auth_method = "DefaultAzureCredential"
            print("Successfully authenticated with DefaultAzureCredential")
            return True
        except Exception as e:
            print(f"DefaultAzureCredential failed: {str(e)}")
            return False
    
    def _try_azure_cli_credential(self):
        """Try authenticating with AzureCliCredential."""
        try:
            print("Trying AzureCliCredential...")
            credential = AzureCliCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            self.credential = credential
            self.token = token
            self.auth_method = "AzureCliCredential"
            print("Successfully authenticated with AzureCliCredential")
            return True
        except Exception as e:
            print(f"AzureCliCredential failed: {str(e)}")
            return False
    
    def _try_api_key(self):
        """Try authenticating with API key."""
        if self.api_key:
            print("Using API key authentication...")
            self.auth_method = "ApiKey"
            return True
        else:
            print("No API key available.")
            return False
    
    def get_headers(self):
        """Get the appropriate headers based on the authentication method."""
        headers = {"Content-Type": "application/json"}
        
        if self.auth_method in ["DefaultAzureCredential", "AzureCliCredential"]:
            # Refresh token if needed
            if self.token.expires_on < time.time() + 300:  # Refresh if expires in less than 5 minutes
                self.token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
            
            headers["Authorization"] = f"Bearer {self.token.token}"
        elif self.auth_method == "ApiKey":
            headers["api-key"] = self.api_key
        
        return headers

def test_azure_openai():
    """Test connection to Azure OpenAI using the hybrid authentication approach."""
    try:
        print(f"Testing Azure OpenAI with endpoint: {AZURE_OPENAI_ENDPOINT}")
        print(f"Deployment: {AZURE_OPENAI_DEPLOYMENT}")
        print(f"API version: {AZURE_OPENAI_API_VERSION}")
        
        # Try token-based authentication first
        authenticator = AzureOpenAIAuthenticator()
        if not authenticator.authenticate():
            print("Failed to authenticate with any method.")
            return False
        
        # Set up API call
        headers = authenticator.get_headers()
        
        # Simple completion request
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you hear me?"}
            ],
            "max_tokens": 100
        }
        
        # Make API call
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        print(f"Making request to: {url}")
        print(f"Using authentication method: {authenticator.auth_method}")
        
        response = requests.post(url, headers=headers, json=payload)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("Successfully connected to Azure OpenAI!")
            print("Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
            # If token-based auth failed, try API key as fallback
            if authenticator.auth_method in ["DefaultAzureCredential", "AzureCliCredential"] and "PermissionDenied" in response.text:
                print("\nToken-based authentication failed with permission denied. Trying API key as fallback...")
                authenticator = AzureOpenAIAuthenticator()
                # Skip to API key directly
                authenticator._try_api_key()
                headers = authenticator.get_headers()
                
                # Try again with API key
                print(f"Retrying with authentication method: {authenticator.auth_method}")
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print("Successfully connected to Azure OpenAI with API key fallback!")
                    print("Response:")
                    print(json.dumps(result, indent=2))
                    return True
                else:
                    print(f"API key fallback also failed: {response.status_code}")
                    print(response.text)
                    return False
            
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Azure OpenAI with hybrid authentication...")
    success = test_azure_openai()
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed.")
        sys.exit(1)
