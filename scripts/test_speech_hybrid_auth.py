#!/usr/bin/env python
"""
Test script for Azure Speech service using API key authentication.
"""

import os
import sys
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Configuration
SPEECH_REGION = os.getenv("SPEECH_REGION", "swedencentral")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")

def test_azure_speech():
    """Test connection to Azure Speech using API key authentication."""
    try:
        print(f"Testing Azure Speech with region: {SPEECH_REGION}")
        print(f"Using API key authentication")
        
        # Create speech configuration with API key
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        
        # Use file output for testing
        output_filename = "test_output.wav"
        file_config = speechsdk.audio.AudioOutputConfig(filename=output_filename)
        
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
        
        print("Attempting to synthesize speech...")
        result = synthesizer.speak_text_async("This is a test of the Azure Speech service.").get()
        
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesis succeeded!")
            print(f"Audio saved to {output_filename}")
            return True
        else:
            print(f"Speech synthesis failed: {result.reason}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Azure Speech with API key authentication...")
    success = test_azure_speech()
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed.")
        sys.exit(1)
