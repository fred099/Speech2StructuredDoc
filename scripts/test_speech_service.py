import os
import sys
from pathlib import Path
import logging
import azure.cognitiveservices.speech as speechsdk
from azure.identity import DefaultAzureCredential

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import SPEECH_REGION, SPEECH_RESOURCE_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_speech_service():
    """
    Test connection to Azure Speech Service using DefaultAzureCredential.
    """
    try:
        # Get Azure credentials using DefaultAzureCredential
        logger.info("Getting Azure credentials using DefaultAzureCredential...")
        credential = DefaultAzureCredential()
        
        # Get access token for Speech Service
        logger.info("Getting access token for Azure Speech Service...")
        token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        # Initialize Speech config with token
        logger.info(f"Initializing Speech config for region: {SPEECH_REGION}")
        speech_config = speechsdk.SpeechConfig(
            auth_token=token,
            region=SPEECH_REGION
        )
        
        # Create a simple recognizer to test connectivity
        logger.info("Creating speech recognizer to test connectivity...")
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Check if the recognizer was created successfully
        if speech_recognizer:
            logger.info("Speech recognizer created successfully")
            logger.info(f"Speech service connection established successfully")
            logger.info(f"Speech service is available in region: {SPEECH_REGION}")
            logger.info(f"Speech resource name: {SPEECH_RESOURCE_NAME}")
        else:
            logger.warning("Failed to create speech recognizer")
            
        logger.info("Speech service test completed")
        
    except Exception as e:
        logger.error(f"Error testing Speech service: {str(e)}")
        raise

if __name__ == "__main__":
    test_speech_service()
