"""
Test script to check Azure OpenAI authentication using direct HTTP requests.
"""

import os
import json
import logging
import requests
from azure.identity import DefaultAzureCredential, AzureCliCredential
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_direct_requests():
    """Test authentication with Azure OpenAI using direct HTTP requests."""
    
    # Get configuration from environment variables
    endpoint = "https://ai-fredrikwingren-2029.openai.azure.com/"
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    # Ensure endpoint doesn't have trailing slash for URL construction
    endpoint = endpoint.rstrip('/')
    
    logger.info(f"Testing authentication with endpoint: {endpoint}")
    logger.info(f"Deployment name: {deployment_name}")
    logger.info(f"API version: {api_version}")
    
    # Test URL for chat completions
    url = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    
    # Request payload
    payload = {
        "messages": [{"role": "user", "content": "Hello, this is a test."}],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    # Test 1: Using API Key
    logger.info("\n=== Test 1: Using API Key with Direct Requests ===")
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        logger.info(f"Status code: {response.status_code}")
        if response.status_code == 200:
            logger.info(f"Success with API key! Response: {response.json()}")
        else:
            logger.error(f"Error with API key. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logger.error(f"Exception with API key: {str(e)}")
    
    # Test 2: Using DefaultAzureCredential
    logger.info("\n=== Test 2: Using DefaultAzureCredential with Direct Requests ===")
    try:
        credential = DefaultAzureCredential()
        
        # Try different scopes
        scopes_to_try = [
            "https://cognitiveservices.azure.com/.default",
            "https://management.azure.com/.default",
            "https://openai.azure.com/.default"
        ]
        
        for scope in scopes_to_try:
            logger.info(f"Getting token with scope: {scope}")
            try:
                token = credential.get_token(scope).token
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
                
                response = requests.post(url, headers=headers, json=payload)
                
                logger.info(f"Status code with scope {scope}: {response.status_code}")
                if response.status_code == 200:
                    logger.info(f"Success with DefaultAzureCredential using scope {scope}! Response: {response.json()}")
                else:
                    logger.error(f"Error with DefaultAzureCredential using scope {scope}. Status code: {response.status_code}, Response: {response.text}")
            except Exception as e:
                logger.error(f"Exception with DefaultAzureCredential using scope {scope}: {str(e)}")
    except Exception as e:
        logger.error(f"Exception with DefaultAzureCredential: {str(e)}")
    
    # Test 3: Using AzureCliCredential
    logger.info("\n=== Test 3: Using AzureCliCredential with Direct Requests ===")
    try:
        cli_credential = AzureCliCredential()
        
        # Try different scopes
        scopes_to_try = [
            "https://cognitiveservices.azure.com/.default",
            "https://management.azure.com/.default",
            "https://openai.azure.com/.default"
        ]
        
        for scope in scopes_to_try:
            logger.info(f"Getting token with scope: {scope}")
            try:
                token = cli_credential.get_token(scope).token
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
                
                response = requests.post(url, headers=headers, json=payload)
                
                logger.info(f"Status code with scope {scope}: {response.status_code}")
                if response.status_code == 200:
                    logger.info(f"Success with AzureCliCredential using scope {scope}! Response: {response.json()}")
                else:
                    logger.error(f"Error with AzureCliCredential using scope {scope}. Status code: {response.status_code}, Response: {response.text}")
            except Exception as e:
                logger.error(f"Exception with AzureCliCredential using scope {scope}: {str(e)}")
    except Exception as e:
        logger.error(f"Exception with AzureCliCredential: {str(e)}")

if __name__ == "__main__":
    test_direct_requests()
