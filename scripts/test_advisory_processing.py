#!/usr/bin/env python
"""
Test script to process the generated advisory meeting audio file through the full pipeline.
This verifies that the audio can be transcribed and structured according to the Pydantic model.
"""

import os
import sys
import json
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from pydantic import ValidationError

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models import AudioFormData, ProcessingResult
from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION

# Load environment variables
load_dotenv()

# Configuration
SPEECH_REGION = os.getenv("SPEECH_REGION", "swedencentral")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

def transcribe_audio(audio_file_path):
    """
    Transcribe an audio file using Azure Speech Service with API key authentication.
    """
    try:
        print(f"Transcribing audio file: {audio_file_path}")
        
        # Initialize speech config with API key authentication
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        
        # Configure audio input
        audio_input = speechsdk.AudioConfig(filename=audio_file_path)
        
        # Create speech recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_input,
            language="sv-SE"  # Swedish language for better recognition
        )
        
        # Variable to store the complete transcription
        transcription = ""
        done = False
        
        # Define callbacks
        def recognized_cb(evt):
            nonlocal transcription
            print(f"RECOGNIZED: {evt.result.text}")
            transcription += evt.result.text + " "
        
        def session_stopped_cb(evt):
            nonlocal done
            print("Session stopped")
            done = True
        
        # Connect callbacks
        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.session_stopped.connect(session_stopped_cb)
        
        # Start continuous recognition
        print("Starting continuous recognition")
        speech_recognizer.start_continuous_recognition()
        
        # Wait for recognition to complete
        import time
        while not done:
            time.sleep(0.5)
        
        # Stop recognition
        speech_recognizer.stop_continuous_recognition()
        
        print("Transcription completed")
        return transcription.strip()
        
    except Exception as e:
        print(f"Error in speech transcription: {str(e)}")
        raise

def extract_structured_data(transcription):
    """
    Extract structured data from transcription text using Azure OpenAI with API key authentication.
    """
    try:
        print("Extracting structured data from transcription")
        
        # Initialize Azure OpenAI client with API key authentication
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY
        )
        
        # Define system prompt for structured data extraction
        system_message = """
        Extract these fields from the transcription as JSON:
        - client_name: The name of the client or organization mentioned
        - meeting_date: The date of the meeting in YYYY-MM-DD format
        - key_points: A summary of the main points discussed
        - action_items: A list of action items or next steps mentioned
        - participants: Names of participants mentioned in the meeting
        
        Set missing fields to null. Format the output as valid JSON.
        """
        
        # Make the API call
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcription}
            ],
            temperature=0.3,  # Lower temperature for more deterministic output
            response_format={"type": "json_object"}
        )
        
        # Extract and validate the response
        result = response.choices[0].message.content
        
        # Validate that the result is valid JSON
        try:
            json_data = json.loads(result)
            print("Successfully extracted structured data")
            return result
        except json.JSONDecodeError:
            print("Azure OpenAI returned invalid JSON")
            raise ValueError("Failed to parse structured data as JSON")
            
    except Exception as e:
        print(f"Error in structured data extraction: {str(e)}")
        raise

def process_audio_file(audio_file_path):
    """
    Process an audio file through the full pipeline:
    1. Transcribe the audio
    2. Extract structured data from the transcription
    3. Validate against the Pydantic model
    """
    try:
        # Step 1: Transcribe the audio
        transcription = transcribe_audio(audio_file_path)
        print(f"\nTranscription:\n{transcription}\n")
        
        # Step 2: Extract structured data
        structured_data_json = extract_structured_data(transcription)
        print(f"\nStructured Data (JSON):\n{structured_data_json}\n")
        
        # Step 3: Parse and validate with Pydantic
        structured_data = AudioFormData.model_validate_json(structured_data_json)
        print("\nStructured Data (Pydantic validated):")
        print(f"- Client Name: {structured_data.client_name}")
        print(f"- Meeting Date: {structured_data.meeting_date}")
        print(f"- Key Points: {structured_data.key_points}")
        print(f"- Action Items: {structured_data.action_items}")
        print(f"- Participants: {structured_data.participants}")
        
        # Step 4: Create the full processing result
        result = ProcessingResult(
            transcription=transcription,
            structured_data=structured_data,
            transcription_url=None,  # Would be set in production
            json_url=None  # Would be set in production
        )
        
        print("\nFull Processing Result:")
        print(result.model_dump_json(indent=2))
        
        return result
        
    except ValidationError as e:
        print(f"Pydantic validation error: {str(e)}")
        raise
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        raise

if __name__ == "__main__":
    print("Testing advisory meeting processing pipeline...")
    
    # Path to the generated test audio file
    audio_file_path = "test_advisory_meeting.wav"
    
    # Process the audio file
    try:
        result = process_audio_file(audio_file_path)
        print("\nTest completed successfully!")
        
        # Save the result to a file for reference
        with open("test_processing_result.json", "w") as f:
            f.write(result.model_dump_json(indent=2))
        print("Results saved to test_processing_result.json")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        sys.exit(1)
