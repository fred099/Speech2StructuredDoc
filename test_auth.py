"""
Simple test script to check Azure OpenAI authentication.
"""

import os
import logging
import requests
from azure.identity import DefaultAzureCredential, AzureCliCredential
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_azure_openai_auth():
    """Test authentication with Azure OpenAI using different methods."""
    
    # Get configuration from environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    logger.info(f"Testing authentication with endpoint: {endpoint}")
    logger.info(f"Deployment name: {deployment_name}")
    logger.info(f"API version: {api_version}")
    
    # Test 1: Using DefaultAzureCredential
    logger.info("\n=== Test 1: Using DefaultAzureCredential ===")
    try:
        credential = DefaultAzureCredential()
        # Try different scopes
        scopes = [
            "https://cognitiveservices.azure.com/.default",
            "https://openai.azure.com/.default",
            "https://management.azure.com/.default"
        ]
        
        for scope in scopes:
            try:
                logger.info(f"Getting token with scope: {scope}")
                token = credential.get_token(scope).token
                logger.info(f"Successfully got token with scope: {scope}")
                
                # Try to use the token
                url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
                payload = {
                    "messages": [{"role": "user", "content": "Hello, this is a test."}],
                    "temperature": 0.7,
                    "max_tokens": 50
                }
                
                logger.info(f"Making request to {url} with token from scope: {scope}")
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    logger.info(f"Success with scope {scope}! Status code: {response.status_code}")
                    logger.info(f"Response: {response.json()}")
                    return  # Exit if successful
                else:
                    logger.error(f"Failed with scope {scope}. Status code: {response.status_code}")
                    logger.error(f"Response: {response.text}")
            
            except Exception as e:
                logger.error(f"Error with scope {scope}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error with DefaultAzureCredential: {str(e)}")
    
    # Test 2: Using AzureCliCredential
    logger.info("\n=== Test 2: Using AzureCliCredential ===")
    try:
        cli_credential = AzureCliCredential()
        
        for scope in scopes:
            try:
                logger.info(f"Getting token with AzureCliCredential and scope: {scope}")
                token = cli_credential.get_token(scope).token
                logger.info(f"Successfully got token with AzureCliCredential and scope: {scope}")
                
                # Try to use the token
                url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
                payload = {
                    "messages": [{"role": "user", "content": "Hello, this is a test."}],
                    "temperature": 0.7,
                    "max_tokens": 50
                }
                
                logger.info(f"Making request to {url} with AzureCliCredential token from scope: {scope}")
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    logger.info(f"Success with AzureCliCredential and scope {scope}! Status code: {response.status_code}")
                    logger.info(f"Response: {response.json()}")
                    return  # Exit if successful
                else:
                    logger.error(f"Failed with AzureCliCredential and scope {scope}. Status code: {response.status_code}")
                    logger.error(f"Response: {response.text}")
            
            except Exception as e:
                logger.error(f"Error with AzureCliCredential and scope {scope}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error with AzureCliCredential: {str(e)}")
    
    # Test 3: Using API Key (if available)
    if api_key:
        logger.info("\n=== Test 3: Using API Key ===")
        try:
            url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key
            }
            payload = {
                "messages": [{"role": "user", "content": "Hello, this is a test."}],
                "temperature": 0.7,
                "max_tokens": 50
            }
            
            logger.info(f"Making request to {url} with API key")
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Success with API key! Status code: {response.status_code}")
                logger.info(f"Response: {response.json()}")
            else:
                logger.error(f"Failed with API key. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
        
        except Exception as e:
            logger.error(f"Error with API key: {str(e)}")
    else:
        logger.warning("API key not available, skipping API key test")

if __name__ == "__main__":
    test_azure_openai_auth()
