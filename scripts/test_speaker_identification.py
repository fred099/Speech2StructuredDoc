#!/usr/bin/env python
"""
Test script to check speaker identification capabilities with the generated test audio file.
This script processes the test_advisory_meeting.wav file and attempts to identify different speakers.
"""

import os
import sys
import time
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Configuration
SPEECH_REGION = os.getenv("SPEECH_REGION", "swedencentral")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")

def process_audio_file_with_speaker_id(audio_file_path):
    """
    Process an audio file and attempt to identify different speakers.
    
    Args:
        audio_file_path: Path to the audio file to process
        
    Returns:
        A list of tuples containing (speaker_id, text) for each recognized segment
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return []
    
    print(f"Processing audio file: {audio_file_path}")
    
    # Initialize speech config with API key authentication
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
    
    # Set Swedish language explicitly
    speech_config.speech_recognition_language = "sv-SE"
    
    # Enable speaker recognition
    speech_config.enable_audio_logging = True
    
    # Configure audio input from the file
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    
    # Create speech recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )
    
    # Store results
    results = []
    speakers = {}
    done = False
    
    # Define callbacks
    def recognized_cb(evt):
        if evt.result.text:
            recognized_text = evt.result.text
            
            # Get speaker ID if available
            speaker_id = None
            if hasattr(evt.result, 'speaker_id'):
                speaker_id = evt.result.speaker_id
            elif hasattr(evt.result, 'properties') and evt.result.properties.get('SpeakerId'):
                speaker_id = evt.result.properties.get('SpeakerId')
            
            # Check for voice profile ID in properties
            if hasattr(evt.result, 'properties') and evt.result.properties.get('VoiceProfileId'):
                speaker_id = evt.result.properties.get('VoiceProfileId')
            
            # If we have a speaker ID, track it
            if speaker_id:
                if speaker_id not in speakers:
                    speaker_num = len(speakers) + 1
                    speakers[speaker_id] = f"Speaker {speaker_num}"
                speaker_name = speakers[speaker_id]
            else:
                speaker_name = "Unknown Speaker"
                
            print(f"{speaker_name}: {recognized_text}")
            results.append((speaker_id, recognized_text))
    
    def session_stopped_cb(evt):
        nonlocal done
        print("Session stopped")
        done = True
    
    def canceled_cb(evt):
        nonlocal done
        print(f"Recognition canceled: {evt.reason}")
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {evt.error_details}")
        done = True
    
    # Connect callbacks
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_stopped.connect(session_stopped_cb)
    speech_recognizer.canceled.connect(canceled_cb)
    
    # Start continuous recognition
    print("Starting recognition...")
    speech_recognizer.start_continuous_recognition()
    
    # Wait until recognition is complete
    while not done:
        time.sleep(0.5)
    
    # Stop recognition
    speech_recognizer.stop_continuous_recognition()
    
    # Print summary of speakers
    print("\nSpeaker Summary:")
    for speaker_id, speaker_name in speakers.items():
        speaker_segments = [text for sid, text in results if sid == speaker_id]
        print(f"{speaker_name} (ID: {speaker_id}): {len(speaker_segments)} segments")
    
    return results

def test_file_with_diarization():
    """
    Test the speaker identification capabilities with the generated test audio file.
    """
    # Path to the test audio file
    audio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_advisory_meeting.wav")
    
    print("Testing speaker identification with the generated test advisory meeting audio file")
    results = process_audio_file_with_speaker_id(audio_file)
    
    # Count unique speakers
    unique_speakers = set([speaker_id for speaker_id, _ in results if speaker_id])
    print(f"\nDetected {len(unique_speakers)} unique speakers in the audio file")
    
    return len(unique_speakers) > 0

if __name__ == "__main__":
    success = test_file_with_diarization()
    if success:
        print("\nSpeaker identification test completed successfully!")
    else:
        print("\nNo speakers were identified in the audio file.")
        print("Note: Basic Azure Speech Service may not support speaker diarization.")
        print("For advanced speaker identification, you may need to use Azure Speech Service with diarization capabilities.")
