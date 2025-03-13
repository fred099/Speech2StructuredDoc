import os
import sys
import logging
import requests
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.auth import get_credential
from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the deployment name from environment variable
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

def test_openai_service():
    """
    Test connectivity to Azure OpenAI service using direct REST API calls.
    """
    try:
        # Get Azure credentials using our robust authentication approach
        logger.info("Getting Azure credentials using DefaultAzureCredential with fallback...")
        credential = get_credential()
        
        # Get access token for Azure OpenAI
        logger.info("Getting access token for Azure OpenAI...")
        token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        # Test using direct REST API calls instead of the SDK to avoid version issues
        logger.info(f"Testing Azure OpenAI service at endpoint: {AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Using deployment: {AZURE_OPENAI_DEPLOYMENT}")
        
        # Set up headers with the token
        headers = {
            "Authorization": f"Bearer {token}",
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
        else:
            logger.error(f"Failed to get completion. Status code: {completion_response.status_code}")
            logger.error(f"Response: {completion_response.text}")
            raise Exception(f"Failed to get completion: {completion_response.text}")
            
    except Exception as e:
        logger.error(f"Error testing Azure OpenAI service: {str(e)}")
        raise

if __name__ == "__main__":
    test_openai_service()
