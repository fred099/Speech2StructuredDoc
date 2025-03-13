import os
import sys
import logging
import requests
from pathlib import Path
from getpass import getpass

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the deployment name from environment variable
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

def test_openai_service_with_key():
    """
    Test connectivity to Azure OpenAI service using API key authentication.
    """
    try:
        # Get API key from user input (this won't be stored in logs or history)
        api_key = getpass("Enter your Azure OpenAI API key: ")
        
        if not api_key:
            logger.error("API key is required to test the Azure OpenAI service")
            return
        
        # Test using direct REST API calls
        logger.info(f"Testing Azure OpenAI service at endpoint: {AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Using deployment: {AZURE_OPENAI_DEPLOYMENT}")
        
        # Set up headers with the API key
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        
        # Test a simple completion to verify the service is working
        logger.info("Testing a simple completion...")
        
        completion_url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        
        completion_payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, are you working properly?"}
            ],
            "max_tokens": 50
        }
        
        logger.info(f"Sending request to: {completion_url}")
        completion_response = requests.post(completion_url, headers=headers, json=completion_payload)
        
        if completion_response.status_code == 200:
            completion_data = completion_response.json()
            response_content = completion_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            logger.info("OpenAI API response received:")
            logger.info(f"Response: {response_content}")
            logger.info("✅ Azure OpenAI service is working correctly")
            
            # Suggest updating the .env file with the API key
            logger.info("\nTo use the API key in your application, add the following to your .env file:")
            logger.info("AZURE_OPENAI_API_KEY=<your-api-key>")
        else:
            logger.error(f"Failed to get completion. Status code: {completion_response.status_code}")
            logger.error(f"Response: {completion_response.text}")
            raise Exception(f"Failed to get completion: {completion_response.text}")
            
    except Exception as e:
        logger.error(f"Error testing Azure OpenAI service: {str(e)}")
        raise

if __name__ == "__main__":
    test_openai_service_with_key()
