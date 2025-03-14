"""
Test script to check Azure OpenAI authentication using the official SDK.
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
    """Test authentication with Azure OpenAI using the official SDK."""
    
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
        
        # Try with different parameters for older SDK versions
        try:
            logger.info("Trying with parameters for older SDK version...")
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_deployment=deployment_name
            )
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                temperature=0.7,
                max_tokens=50
            )
            
            logger.info(f"Success with API key (older SDK)! Response: {response}")
        except Exception as e2:
            logger.error(f"Error with API key (older SDK): {str(e2)}")
    
    # Test 2: Using DefaultAzureCredential
    logger.info("\n=== Test 2: Using DefaultAzureCredential with Official SDK ===")
    try:
        credential = DefaultAzureCredential()
        
        client = AzureOpenAI(
            azure_ad_token_provider=credential,
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
        
        # Try with different parameters for older SDK versions
        try:
            logger.info("Trying with DefaultAzureCredential for older SDK version...")
            client = AzureOpenAI(
                azure_ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token,
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_deployment=deployment_name
            )
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                temperature=0.7,
                max_tokens=50
            )
            
            logger.info(f"Success with DefaultAzureCredential (older SDK)! Response: {response}")
        except Exception as e2:
            logger.error(f"Error with DefaultAzureCredential (older SDK): {str(e2)}")
    
    # Test 3: Using AzureCliCredential
    logger.info("\n=== Test 3: Using AzureCliCredential with Official SDK ===")
    try:
        cli_credential = AzureCliCredential()
        
        client = AzureOpenAI(
            azure_ad_token_provider=cli_credential,
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
        
        # Try with different parameters for older SDK versions
        try:
            logger.info("Trying with AzureCliCredential for older SDK version...")
            client = AzureOpenAI(
                azure_ad_token=cli_credential.get_token("https://cognitiveservices.azure.com/.default").token,
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_deployment=deployment_name
            )
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                temperature=0.7,
                max_tokens=50
            )
            
            logger.info(f"Success with AzureCliCredential (older SDK)! Response: {response}")
        except Exception as e2:
            logger.error(f"Error with AzureCliCredential (older SDK): {str(e2)}")
    
if __name__ == "__main__":
    test_azure_openai_sdk()
