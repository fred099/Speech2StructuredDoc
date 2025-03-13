import azure.cognitiveservices.speech as speechsdk
import logging
from src.config import SPEECH_REGION

logger = logging.getLogger(__name__)

def transcribe_audio(audio_file_path, token):
    """
    Transcribe an audio file using Azure Speech Service.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        token: Authentication token for Azure Speech Service
        
    Returns:
        Transcribed text from the audio file
    """
    try:
        # Initialize speech config with token-based authentication
        logger.info(f"Initializing speech config for region: {SPEECH_REGION}")
        speech_config = speechsdk.SpeechConfig(
            auth_token=token,
            region=SPEECH_REGION
        )
        
        # Configure audio input
        logger.info(f"Configuring audio input from file: {audio_file_path}")
        audio_input = speechsdk.AudioConfig(filename=audio_file_path)
        
        # Create speech recognizer
        logger.info("Creating speech recognizer")
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_input
        )
        
        # Start speech recognition
        logger.info("Starting speech recognition")
        
        # Variable to store the complete transcription
        transcription = ""
        
        # Define callback for recognized speech
        def recognized_cb(evt):
            nonlocal transcription
            logger.info(f"RECOGNIZED: {evt.result.text}")
            transcription += evt.result.text + " "
        
        # Connect callbacks
        speech_recognizer.recognized.connect(recognized_cb)
        
        # Start continuous recognition
        logger.info("Starting continuous recognition")
        speech_recognizer.start_continuous_recognition()
        
        # Wait for recognition to complete
        import time
        time.sleep(5)  # Wait for 5 seconds to start
        
        # Check if audio file is still being processed
        while len(transcription) == 0:
            logger.info("Waiting for transcription to begin...")
            time.sleep(2)
        
        # Wait for 10 more seconds after transcription begins to ensure completion
        # This is a simple approach - in production, you'd implement a more robust solution
        time.sleep(10)
        
        # Stop recognition
        logger.info("Stopping continuous recognition")
        speech_recognizer.stop_continuous_recognition()
        
        logger.info("Transcription completed")
        return transcription.strip()
        
    except Exception as e:
        logger.error(f"Error in speech transcription: {str(e)}")
        raise
