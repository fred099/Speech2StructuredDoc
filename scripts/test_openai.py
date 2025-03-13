import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from src.auth import get_credential
from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CAPABLE_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_openai_service():
    """
    Test connectivity to Azure OpenAI service and check for available models.
    """
    try:
        # Get Azure credentials using our robust authentication approach
        logger.info("Getting Azure credentials using DefaultAzureCredential with fallback...")
        credential = get_credential()
        
        # Get access token for Azure OpenAI
        logger.info("Getting access token for Azure OpenAI...")
        token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        # Initialize Azure OpenAI client with token-based authentication
        logger.info(f"Initializing Azure OpenAI client with endpoint: {AZURE_OPENAI_ENDPOINT}")
        
        # Create the client without the proxies parameter
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_ad_token=token
        )
        
        # List available models/deployments
        logger.info("Listing available models/deployments...")
        models = client.models.list()
        
        # Check if our configured model is available
        model_found = False
        logger.info(f"Looking for model: {AZURE_OPENAI_CAPABLE_MODEL}")
        
        logger.info("Available models:")
        for model in models:
            logger.info(f"- {model.id}")
            if model.id == AZURE_OPENAI_CAPABLE_MODEL:
                model_found = True
        
        if model_found:
            logger.info(f" Model '{AZURE_OPENAI_CAPABLE_MODEL}' is available")
        else:
            logger.warning(f" Model '{AZURE_OPENAI_CAPABLE_MODEL}' was not found in the available models")
        
        # Test a simple completion to verify the service is working
        logger.info("Testing a simple completion...")
        
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_CAPABLE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, are you working properly?"}
                ],
                max_tokens=50
            )
            
            logger.info("OpenAI API response received:")
            logger.info(f"Response: {response.choices[0].message.content}")
            logger.info(" Azure OpenAI service is working correctly")
            
        except Exception as api_error:
            logger.error(f"Error making API call: {str(api_error)}")
            logger.warning(" Azure OpenAI service API call failed")
            raise
            
    except Exception as e:
        logger.error(f"Error testing Azure OpenAI service: {str(e)}")
        raise

if __name__ == "__main__":
    test_openai_service()
