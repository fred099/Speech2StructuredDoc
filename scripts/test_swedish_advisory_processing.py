#!/usr/bin/env python
"""
Test script to process the generated Swedish advisory meeting audio file through the full pipeline.
This verifies that the audio can be transcribed and structured according to the Pydantic model.
"""

import os
import sys
import json
import requests
import threading
import time
import subprocess
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
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

def play_audio(audio_file_path):
    """
    Play audio file using Windows Media Player in a separate thread.
    """
    try:
        print(f"Spelar upp ljudfil: {audio_file_path}")
        # Use PowerShell to play the audio file
        full_path = os.path.abspath(audio_file_path)
        subprocess.Popen(['powershell', '-c', f'(New-Object Media.SoundPlayer "{full_path}").PlaySync()'])
        print("Ljuduppspelning startad")
    except Exception as e:
        print(f"Fel vid uppspelning av ljud: {str(e)}")

def transcribe_audio(audio_file_path):
    """
    Transcribe a Swedish audio file using Azure Speech Service with API key authentication.
    """
    try:
        print(f"Transkriberar ljudfil: {audio_file_path}")
        
        # Start playing the audio file in a separate thread
        audio_thread = threading.Thread(target=play_audio, args=(audio_file_path,))
        audio_thread.daemon = True
        audio_thread.start()
        
        # Initialize speech config with API key authentication
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        
        # Set Swedish language explicitly
        speech_config.speech_recognition_language = "sv-SE"
        
        # Configure audio input
        audio_input = speechsdk.AudioConfig(filename=audio_file_path)
        
        # Create speech recognizer with Swedish language
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_input
        )
        
        # Variable to store the complete transcription
        transcription = ""
        done = False
        
        # Define callbacks
        def recognized_cb(evt):
            nonlocal transcription
            print(f"IGENKÄNT: {evt.result.text}")
            transcription += evt.result.text + " "
        
        def session_stopped_cb(evt):
            nonlocal done
            print("Session avslutad")
            done = True
        
        def canceled_cb(evt):
            nonlocal done
            print(f"Igenkänning avbruten: {evt.reason}")
            if evt.reason == speechsdk.CancellationReason.Error:
                print(f"Feldetaljer: {evt.error_details}")
            done = True
        
        # Connect callbacks
        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.session_stopped.connect(session_stopped_cb)
        speech_recognizer.canceled.connect(canceled_cb)
        
        # Start continuous recognition
        print("Startar kontinuerlig igenkänning...")
        speech_recognizer.start_continuous_recognition()
        
        # Wait for recognition to complete
        max_wait_time = 120  # Maximum wait time in seconds
        start_time = time.time()
        
        while not done and (time.time() - start_time) < max_wait_time:
            time.sleep(0.5)
        
        # Stop recognition
        speech_recognizer.stop_continuous_recognition()
        
        if not done:
            print("Timeout - avbryter igenkänning")
        
        print("Transkription slutförd")
        return transcription.strip()
        
    except Exception as e:
        print(f"Fel vid taltranskription: {str(e)}")
        raise

def extract_structured_data(transcription):
    """
    Extract structured data from Swedish transcription text using Azure OpenAI with API key authentication.
    """
    try:
        print("Extraherar strukturerad data från transkription")
        
        # Set up API call with API key authentication
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }
        
        # Define system prompt for structured data extraction in Swedish
        system_message = """
        Du är en AI-assistent som hjälper till att extrahera strukturerad information från transkriptioner av rådgivningsmöten.
        Extrahera följande fält från transkriptionen som JSON:
        - client_name: Namnet på kunden eller organisationen som nämns (string)
        - meeting_date: Datumet för mötet i formatet ÅÅÅÅ-MM-DD (string)
        - key_points: En sammanfattning av huvudpunkterna som diskuterades (string, inte en array)
        - action_items: En lista över åtgärdspunkter eller nästa steg som nämndes (array av strings)
        - participants: Namn på deltagare som nämndes i mötet (array av strings)
        
        VIKTIGT: key_points måste vara en sträng, inte en array.
        Sätt saknade fält till null. Formatera utdata som giltig JSON.
        """
        
        # Simple completion request
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcription}
            ],
            "temperature": 0.3,
            "max_tokens": 800,
            "response_format": {"type": "json_object"}
        }
        
        # Make API call
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        print(f"Gör förfrågan till: {url}")
        
        response = requests.post(url, headers=headers, json=payload)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            structured_data = result["choices"][0]["message"]["content"]
            print("Strukturerad data extraherades framgångsrikt")
            
            # Validate that the result is valid JSON
            try:
                json_data = json.loads(structured_data)
                
                # Fix key_points if it's an array
                if isinstance(json_data.get('key_points'), list):
                    json_data['key_points'] = ' '.join(json_data['key_points'])
                    # Update the structured_data with the fixed version
                    structured_data = json.dumps(json_data)
                    print("Fixade key_points format (konverterade från array till sträng)")
                
                return structured_data
            except json.JSONDecodeError:
                print("Azure OpenAI returnerade ogiltig JSON")
                raise ValueError("Kunde inte tolka strukturerad data som JSON")
        else:
            print(f"Fel: {response.status_code}")
            print(response.text)
            raise ValueError(f"API-anrop misslyckades med statuskod {response.status_code}")
            
    except Exception as e:
        print(f"Fel vid extraktion av strukturerad data: {str(e)}")
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
        print(f"\nTranskription:\n{transcription}\n")
        
        # Step 2: Extract structured data
        structured_data_json = extract_structured_data(transcription)
        print(f"\nStrukturerad Data (JSON):\n{structured_data_json}\n")
        
        # Step 3: Parse and validate with Pydantic
        structured_data = AudioFormData.model_validate_json(structured_data_json)
        print("\nStrukturerad Data (Pydantic validerad):")
        print(f"- Kundnamn: {structured_data.client_name}")
        print(f"- Mötesdatum: {structured_data.meeting_date}")
        print(f"- Huvudpunkter: {structured_data.key_points}")
        print(f"- Åtgärdspunkter: {structured_data.action_items}")
        print(f"- Deltagare: {structured_data.participants}")
        
        # Step 4: Create the full processing result
        result = ProcessingResult(
            transcription=transcription,
            structured_data=structured_data,
            transcription_url=None,  # Would be set in production
            json_url=None  # Would be set in production
        )
        
        print("\nFullständigt bearbetningsresultat:")
        print(result.model_dump_json(indent=2))
        
        return result
        
    except ValidationError as e:
        print(f"Pydantic valideringsfel: {str(e)}")
        raise
    except Exception as e:
        print(f"Fel vid bearbetning av ljudfil: {str(e)}")
        raise

if __name__ == "__main__":
    print("Testar bearbetningspipeline för rådgivningsmöte...")
    
    # Path to the generated test audio file
    audio_file_path = "test_advisory_meeting.wav"
    
    # Process the audio file
    try:
        result = process_audio_file(audio_file_path)
        print("\nTest slutfördes framgångsrikt!")
        
        # Save the result to a file for reference
        with open("test_processing_result.json", "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        print("Resultat sparade till test_processing_result.json")
        
    except Exception as e:
        print(f"\nTest misslyckades: {str(e)}")
        sys.exit(1)
