"""
Test script to verify Azure OpenAI authentication is working correctly with API key.
"""

import os
import logging
from dotenv import load_dotenv
from src.azure_openai_provider import AzureOpenAIProvider, CompletionRequest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_azure_openai():
    """Test Azure OpenAI authentication with API key."""
    
    logger.info("Testing Azure OpenAI authentication with API key")
    
    # Initialize the provider with API key authentication
    provider = AzureOpenAIProvider(use_token=False)
    
    # Create a simple completion request
    request = CompletionRequest(
        prompt="Hello, this is a test message. Please respond with a short greeting.",
        system_message="You are a helpful assistant. Keep your responses short and to the point.",
        temperature=0.7,
        max_tokens=50
    )
    
    try:
        # Get completion
        logger.info("Sending completion request to Azure OpenAI")
        response = provider.get_completion(request)
        
        logger.info(f"Successfully received response from Azure OpenAI: {response}")
        return True
    except Exception as e:
        logger.error(f"Error getting completion from Azure OpenAI: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_azure_openai()
    if success:
        print("\n✅ Azure OpenAI authentication with API key is working correctly!")
    else:
        print("\n❌ Azure OpenAI authentication with API key failed!")
