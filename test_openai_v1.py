"""
Test script to check Azure OpenAI authentication using the official SDK v1.50.1.
"""

import os
import logging
from azure.identity import DefaultAzureCredential, AzureCliCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_azure_openai_sdk():
    """Test authentication with Azure OpenAI using the official SDK v1.50.1."""
    
    # Get configuration from environment variables
    endpoint = "https://ai-fredrikwingren-2029.openai.azure.com/"
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    logger.info(f"Testing authentication with endpoint: {endpoint}")
    logger.info(f"Deployment name: {deployment_name}")
    logger.info(f"API version: {api_version}")
    
    # Test 1: Using API Key
    logger.info("\n=== Test 1: Using API Key with Official SDK ===")
    try:
        # For OpenAI v1.x, the correct parameters are:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            temperature=0.7,
            max_tokens=50
        )
        
        logger.info(f"Success with API key! Response: {response}")
    except Exception as e:
        logger.error(f"Error with API key: {str(e)}")
    
    # Test 2: Using DefaultAzureCredential
    logger.info("\n=== Test 2: Using DefaultAzureCredential with Official SDK ===")
    try:
        # For OpenAI v1.x with Azure AD authentication
        credential = DefaultAzureCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        client = AzureOpenAI(
            azure_ad_token=token,  # Use azure_ad_token instead of azure_ad_token_provider
            api_version=api_version,
            azure_endpoint=endpoint
        )
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            temperature=0.7,
            max_tokens=50
        )
        
        logger.info(f"Success with DefaultAzureCredential! Response: {response}")
    except Exception as e:
        logger.error(f"Error with DefaultAzureCredential: {str(e)}")
    
    # Test 3: Using AzureCliCredential
    logger.info("\n=== Test 3: Using AzureCliCredential with Official SDK ===")
    try:
        # For OpenAI v1.x with Azure AD authentication via CLI
        cli_credential = AzureCliCredential()
        token = cli_credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        client = AzureOpenAI(
            azure_ad_token=token,  # Use azure_ad_token instead of azure_ad_token_provider
            api_version=api_version,
            azure_endpoint=endpoint
        )
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            temperature=0.7,
            max_tokens=50
        )
        
        logger.info(f"Success with AzureCliCredential! Response: {response}")
    except Exception as e:
        logger.error(f"Error with AzureCliCredential: {str(e)}")

if __name__ == "__main__":
    test_azure_openai_sdk()
