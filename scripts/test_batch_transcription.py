#!/usr/bin/env python
"""
Test script for batch transcription of a WAV file with speaker identification.
This script uses Azure Speech SDK's batch transcription capabilities to process
a pre-recorded WAV file and identify speakers.
"""

import os
import sys
import json
import time
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from realtime_meeting_processor import extract_structured_data
from src.models import ParticipantRole

# Load environment variables
load_dotenv()

# Configuration
SPEECH_REGION = os.getenv("SPEECH_REGION", "swedencentral")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")
WAV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_advisory_meeting.wav")
SPEAKERS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_advisory_meeting_speakers.json")

def batch_transcribe_wav_file(wav_file):
    """
    Perform batch transcription of a WAV file with speaker identification.
    
    Args:
        wav_file: Path to the WAV file to transcribe
        
    Returns:
        dict: Dictionary containing transcription results and speaker information
    """
    print(f"Starting batch transcription of WAV file: {wav_file}")
    
    if not os.path.exists(wav_file):
        print(f"Error: WAV file not found: {wav_file}")
        return None
    
    try:
        # Initialize speech config with API key authentication
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        
        # Set Swedish language explicitly
        speech_config.speech_recognition_language = "sv-SE"
        
        # Enable audio logging for better diagnostics
        speech_config.enable_audio_logging = True
        
        # Enable speaker diarization
        speech_config.set_service_property(
            name="speechcontext-PhraseOutput.Format",
            value="Detailed",
            channel=speechsdk.ServicePropertyChannel.UriQueryParameter
        )
        
        # Generate a unique session ID for this transcription
        import uuid
        session_id = str(uuid.uuid4())
        
        # Set the session ID for speaker diarization
        speech_config.set_service_property(
            name="speechcontext-dialog.sessionId",
            value=session_id,
            channel=speechsdk.ServicePropertyChannel.UriQueryParameter
        )
        
        speech_config.set_service_property(
            name="speechcontext-phraseDetection.speakerDiarization.enabled",
            value="true",
            channel=speechsdk.ServicePropertyChannel.UriQueryParameter
        )
        
        # Set maximum number of speakers
        speech_config.set_service_property(
            name="speechcontext-phraseDetection.speakerDiarization.maxSpeakerCount",
            value="10",
            channel=speechsdk.ServicePropertyChannel.UriQueryParameter
        )
        
        # Configure audio input from the WAV file
        audio_config = speechsdk.audio.AudioConfig(filename=wav_file)
        
        # Create speech recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Load speaker information if available
        speaker_info = {}
        if os.path.exists(SPEAKERS_FILE):
            try:
                with open(SPEAKERS_FILE, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                    print(f"Loaded speaker information: {len(speaker_data['speakers'])} speakers")
                    
                    # Map voice names to roles and names
                    for speaker in speaker_data['speakers']:
                        voice_name = speaker.get('voice')
                        role = speaker.get('role', 'unknown')
                        name = speaker.get('name', 'Unknown')
                        
                        speaker_info[voice_name] = {
                            'name': name,
                            'role': role
                        }
                        
                        print(f"Speaker: {name} (Voice: {voice_name}, Role: {role})")
            except Exception as e:
                print(f"Error loading speaker information: {str(e)}")
        
        # Store all transcription results
        all_results = []
        
        # Callback to handle recognized speech
        def recognized_cb(evt):
            recognized_text = evt.result.text
            if not recognized_text:
                return
            
            # Get speaker ID if available
            speaker_id = None
            speaker_name = "Unknown Speaker"
            speaker_role = "unknown"
            
            # Try to extract speaker ID from the result
            if hasattr(evt.result, 'speaker_id'):
                speaker_id = evt.result.speaker_id
            
            # Try to extract from properties
            if not speaker_id and hasattr(evt.result, 'properties'):
                try:
                    # Print all properties for debugging
                    print("\nAvailable properties:")
                    for prop_name in evt.result.properties:
                        print(f"  - {prop_name}: {evt.result.properties[prop_name]}")
                    
                    # Try to extract from JSON result
                    if "SpeechServiceResponse_JsonResult" in evt.result.properties:
                        json_str = evt.result.properties["SpeechServiceResponse_JsonResult"]
                        print(f"JSON Result: {json_str}")
                        json_result = json.loads(json_str)
                        
                        if 'SpeakerId' in json_result:
                            speaker_id = json_result['SpeakerId']
                        elif 'NBest' in json_result and len(json_result['NBest']) > 0 and 'SpeakerId' in json_result['NBest'][0]:
                            speaker_id = json_result['NBest'][0]['SpeakerId']
                except Exception as e:
                    print(f"Error extracting speaker ID from properties: {str(e)}")
            
            # If we still don't have a speaker ID, try to infer from the text
            if not speaker_id:
                # Try to match the text with known speakers
                for voice_name, info in speaker_info.items():
                    if info['name'] in recognized_text:
                        speaker_id = voice_name
                        print(f"Inferred speaker ID {speaker_id} from text content")
                        break
            
            # Map speaker ID to name and role if possible
            if speaker_id and speaker_id in speaker_info:
                speaker_name = speaker_info[speaker_id]['name']
                speaker_role = speaker_info[speaker_id]['role']
            
            # Print the recognized text with speaker information
            print(f"\nRECOGNIZED: {recognized_text}")
            print(f"SPEAKER: {speaker_name} (ID: {speaker_id}, Role: {speaker_role.upper()})")
            
            # Add to results
            all_results.append({
                'text': recognized_text,
                'speaker_id': speaker_id,
                'speaker_name': speaker_name,
                'speaker_role': speaker_role
            })
        
        # Connect the callback to the recognizer
        speech_recognizer.recognized.connect(recognized_cb)
        
        # Start recognition and wait for completion
        print("Starting batch transcription...")
        result_future = speech_recognizer.recognize_once_async()
        result = result_future.get()
        
        # Process the result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Speech recognized successfully")
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print(f"No speech could be recognized: {result.no_match_details}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print(f"Speech recognition canceled: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation.error_details}")
        
        # Combine all results into a single transcription
        transcription_text = "\n".join([f"{r['speaker_name']} ({r['speaker_role'].upper()}): {r['text']}" for r in all_results])
        
        # Create a result object
        result = {
            'transcription_text': transcription_text,
            'transcription_lines': all_results,
            'speakers': {r['speaker_id']: r['speaker_name'] for r in all_results if r['speaker_id']},
            'voice_roles': {r['speaker_id']: r['speaker_role'] for r in all_results if r['speaker_id']}
        }
        
        return result
    
    except Exception as e:
        print(f"Error in batch transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_batch_transcription():
    """
    Test the batch transcription functionality with speaker identification.
    """
    # Perform batch transcription
    result = batch_transcribe_wav_file(WAV_FILE)
    
    if not result:
        print("Batch transcription failed")
        return False
    
    # Print transcription results
    print("\nTranscription Results:")
    print("-" * 50)
    print(result['transcription_text'])
    
    # Print speaker information
    print("\nSpeaker Information:")
    print("-" * 50)
    for speaker_id, name in result['speakers'].items():
        role = result['voice_roles'].get(speaker_id, "unknown")
        print(f"- {name} (ID: {speaker_id}): {role.upper()}")
    
    # Extract structured data
    try:
        print("\nExtracting structured data...")
        structured_data = extract_structured_data(result['transcription_text'])
        
        if structured_data:
            print("\nStructured Data:")
            print("-" * 50)
            print(f"Client: {structured_data.get('client_name', 'Unknown')}")
            
            participants = structured_data.get('participants', [])
            if participants:
                print("Participants:")
                for p in participants:
                    if isinstance(p, dict):
                        print(f"- {p.get('name', 'Unknown')} ({p.get('role', 'unknown').upper()})")
            
            # Save the structured data to a file
            output_file = os.path.join(os.path.dirname(WAV_FILE), "batch_structured_data.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            print(f"\nStructured data saved to: {output_file}")
            
            # Save the transcription to a file
            transcription_file = os.path.join(os.path.dirname(WAV_FILE), "batch_transcription.txt")
            with open(transcription_file, 'w', encoding='utf-8') as f:
                f.write(result['transcription_text'])
            print(f"Transcription saved to: {transcription_file}")
            
            print("\nTest completed successfully!")
            return True
        else:
            print("Failed to extract structured data")
            return False
    
    except Exception as e:
        print(f"Error extracting structured data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_batch_transcription()
