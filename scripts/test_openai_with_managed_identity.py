#!/usr/bin/env python
"""
Test script for Azure OpenAI service using DefaultAzureCredential authentication.
Following the WebScraper-RAG authentication pattern.
"""

import os
import sys
import json
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, AzureCliCredential
import requests

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION

# Load environment variables
load_dotenv()

def get_credential():
    """
    Get Azure credential using DefaultAzureCredential with fallback to AzureCliCredential.
    Following the WebScraper-RAG authentication pattern.
    """
    try:
        # Try DefaultAzureCredential first (primary method)
        credential = DefaultAzureCredential(additionally_allowed_tenants=["*"])
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        print("Successfully authenticated with DefaultAzureCredential")
        return credential
    except Exception as e:
        print(f"DefaultAzureCredential failed: {str(e)}")
        try:
            # Fall back to AzureCliCredential
            credential = AzureCliCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            print("Successfully authenticated with AzureCliCredential")
            return credential
        except Exception as cli_error:
            print(f"AzureCliCredential also failed: {str(cli_error)}")
            raise Exception("All authentication methods failed")

def test_azure_openai():
    """Test connection to Azure OpenAI using managed identity."""
    try:
        # Get credential
        credential = get_credential()
        
        # Get access token for Azure OpenAI
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        
        # Set up API call
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        headers = {
            "Authorization": f"Bearer {token.token}",
            "Content-Type": "application/json"
        }
        
        # Simple completion request
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you hear me?"}
            ],
            "max_tokens": 100
        }
        
        # Make API call
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{deployment_name}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        print(f"Making request to: {url}")
        
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
            
            # If we get a permission error, try to assign the role
            if response.status_code == 401 and "PermissionDenied" in response.text:
                print("\nPermission denied. Let's try to assign the necessary role to the managed identity.")
                print("Please run the following command in Azure CLI:")
                print(f"az role assignment create --assignee \"6ef2e00e-ec5d-42e2-8dfb-3769f478f898\" --role \"Cognitive Services OpenAI User\" --scope \"/subscriptions/244a8131-c92e-46d3-a772-f312af33bc21/resourceGroups/GraphRag/providers/Microsoft.CognitiveServices/accounts/ai-fredrikwingren-2029\"")
            
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Azure OpenAI with DefaultAzureCredential...")
    success = test_azure_openai()
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed.")
        sys.exit(1)
