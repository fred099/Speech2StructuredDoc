import azure.cognitiveservices.speech as speechsdk
import logging
import json
import time
import os
import threading
from src.config import SPEECH_REGION, SPEECH_API_KEY
from src.speaker_identification import configure_diarization

logger = logging.getLogger(__name__)

def transcribe_audio(audio_file_path, api_key, enable_diarization=False):
    """
    Transcribe an audio file using Azure Speech Service.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        api_key: API key for Azure Speech Service
        enable_diarization: Whether to enable speaker diarization (default: False)
        
    Returns:
        If enable_diarization is False:
            Transcribed text from the audio file
        If enable_diarization is True:
            Tuple of (transcribed_text, list of (speaker_id, utterance) tuples)
    """
    try:
        # Initialize speech config with appropriate authentication
        logger.info(f"Initializing speech config for region: {SPEECH_REGION}")
        if SPEECH_API_KEY:
            logger.info("Using API key authentication for Speech Service")
            speech_config = speechsdk.SpeechConfig(
                subscription=SPEECH_API_KEY,
                region=SPEECH_REGION
            )
        else:
            logger.info("Using provided API key for Speech Service")
            speech_config = speechsdk.SpeechConfig(
                subscription=api_key,
                region=SPEECH_REGION
            )
        
        # Configure diarization if enabled
        if enable_diarization:
            logger.info("Enabling speaker diarization")
            speech_config = configure_diarization(speech_config)
        
        # Configure audio input
        logger.info(f"Configuring audio input from file: {audio_file_path}")
        audio_input = speechsdk.AudioConfig(filename=audio_file_path)
        
        # Create appropriate recognizer based on diarization setting
        if enable_diarization:
            logger.info("Creating conversation transcriber for speaker diarization")
            recognizer = speechsdk.transcription.ConversationTranscriber(
                speech_config=speech_config,
                audio_config=audio_input
            )
        else:
            logger.info("Creating standard speech recognizer")
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_input
            )
        
        # Variables to store the complete transcription and speaker utterances
        transcription = ""
        speaker_utterances = []
        session_stopped = False
        transcription_done = threading.Event()
        file_processed = threading.Event()
        
        # Define callback for recognized speech (standard recognizer)
        def recognized_cb(evt):
            nonlocal transcription
            logger.info(f"RECOGNIZED: {evt.result.text}")
            transcription += evt.result.text + " "
        
        # Define callback for transcribed speech (conversation transcriber)
        def transcribed_cb(evt):
            nonlocal transcription, speaker_utterances
            
            # Get the text and speaker ID
            text = evt.result.text
            speaker_id = evt.result.speaker_id if evt.result.speaker_id else "Unknown"
            
            logger.info(f"TRANSCRIBED: {text} (Speaker: {speaker_id})")
            
            # Add to the transcription with speaker information
            line = f"[Speaker {speaker_id}]: {text}"
            transcription += line + "\n"
            
            # Add to speaker utterances list
            speaker_utterances.append((speaker_id, text))
        
        # Connect appropriate callbacks based on recognizer type
        if enable_diarization:
            recognizer.transcribed.connect(transcribed_cb)
            
            # Also connect session callbacks for better logging
            def session_started_cb(evt):
                logger.info(f"SESSION STARTED: {evt}")
            
            def session_stopped_cb(evt):
                nonlocal session_stopped
                logger.info(f"SESSION STOPPED: {evt}")
                session_stopped = True
                # Signal that transcription is done when session stops
                transcription_done.set()
                file_processed.set()
            
            # Add a callback for when the file is fully processed
            def canceled_cb(evt):
                logger.info(f"CANCELED: {evt}")
                # If the reason is EndOfStream, it means the file has been fully processed
                if evt.reason == speechsdk.CancellationReason.EndOfStream:
                    logger.info("Audio file fully processed")
                    file_processed.set()
            
            recognizer.session_started.connect(session_started_cb)
            recognizer.session_stopped.connect(session_stopped_cb)
            recognizer.canceled.connect(canceled_cb)
            
            # Start transcribing
            recognizer.start_transcribing_async()
        else:
            recognizer.recognized.connect(recognized_cb)
            
            # Add callbacks for recognition completed and canceled
            def recognition_completed_cb(evt):
                logger.info("RECOGNITION COMPLETED")
                file_processed.set()
            
            def canceled_cb(evt):
                logger.info(f"CANCELED: {evt}")
                # If the reason is EndOfStream, it means the file has been fully processed
                if evt.reason == speechsdk.CancellationReason.EndOfStream:
                    logger.info("Audio file fully processed")
                    file_processed.set()
            
            recognizer.recognized_completed.connect(recognition_completed_cb)
            recognizer.canceled.connect(canceled_cb)
            
            # Start continuous recognition
            recognizer.start_continuous_recognition()
        
        # Wait for recognition to complete or session to stop
        logger.info("Waiting for audio file to be fully processed...")
        
        # Wait for file to be fully processed or a safety timeout
        max_wait = 300  # Maximum total wait time in seconds (safety timeout)
        
        # Wait for the file_processed event with a timeout
        file_processed.wait(timeout=max_wait)
        
        # Check if file was fully processed
        if file_processed.is_set():
            logger.info("Audio file fully processed")
        else:
            logger.info("Maximum wait time reached, stopping transcription")
        
        # Stop recognition to clean up resources
        if enable_diarization:
            recognizer.stop_transcribing_async()
        else:
            recognizer.stop_continuous_recognition()
        
        logger.info("Transcription completed")
        
        # Return appropriate result based on diarization setting
        if enable_diarization:
            return transcription.strip(), speaker_utterances
        else:
            return transcription.strip()
            
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise
