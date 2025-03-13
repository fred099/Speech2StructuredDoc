import os
import sys
from pathlib import Path
import logging
import argparse

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from azure.identity import DefaultAzureCredential
from src.speech import transcribe_audio
from src.config import SPEECH_REGION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_transcription(audio_file_path):
    """
    Test audio transcription using Azure Speech Service.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
    """
    try:
        # Get Azure credentials using DefaultAzureCredential
        logger.info("Getting Azure credentials using DefaultAzureCredential...")
        credential = DefaultAzureCredential()
        
        # Get access token for Speech Service
        logger.info("Getting access token for Azure Speech Service...")
        token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        # Verify the audio file exists
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return
        
        logger.info(f"Transcribing audio file: {audio_file_path}")
        
        # Call the transcribe_audio function
        transcription = transcribe_audio(audio_file_path, token)
        
        # Display the transcription result
        logger.info("Transcription result:")
        logger.info("-" * 50)
        logger.info(transcription)
        logger.info("-" * 50)
        logger.info("Transcription test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing transcription: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test audio transcription using Azure Speech Service")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    args = parser.parse_args()
    
    test_transcription(args.audio_file)
