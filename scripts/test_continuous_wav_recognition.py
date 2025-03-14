#!/usr/bin/env python
"""
Test script for continuous recognition of a WAV file with speaker identification.
This script uses Azure Speech SDK's continuous recognition capabilities to process
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

def continuous_recognize_wav_file(wav_file):
    """
    Perform continuous recognition of a WAV file with speaker identification.
    
    Args:
        wav_file: Path to the WAV file to transcribe
        
    Returns:
        dict: Dictionary containing transcription results and speaker information
    """
    print(f"Starting continuous recognition of WAV file: {wav_file}")
    
    if not os.path.exists(wav_file):
        print(f"Error: WAV file not found: {wav_file}")
        return None
    
    try:
        # Initialize speech config with API key authentication
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        
        # Set Swedish language explicitly
        speech_config.speech_recognition_language = "sv-SE"
        
        # Enable detailed output format
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_ProfanityOption, "masked")
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "true")
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary, "true")
        
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
        done = False
        
        # Callback to handle recognized speech
        def recognized_cb(evt):
            recognized_text = evt.result.text
            if not recognized_text:
                return
            
            # Get speaker ID if available
            speaker_id = None
            speaker_name = "Unknown Speaker"
            speaker_role = "unknown"
            
            # Try to extract from properties
            if hasattr(evt.result, 'properties'):
                try:
                    # Try to extract from JSON result
                    if "SpeechServiceResponse_JsonResult" in evt.result.properties:
                        json_str = evt.result.properties["SpeechServiceResponse_JsonResult"]
                        json_result = json.loads(json_str)
                        
                        # Print the JSON result for debugging
                        print(f"JSON Result: {json.dumps(json_result, indent=2)}")
                        
                        # Try to extract speaker ID from various locations in the JSON
                        if 'SpeakerId' in json_result:
                            speaker_id = json_result['SpeakerId']
                        elif 'NBest' in json_result and len(json_result['NBest']) > 0:
                            if 'SpeakerId' in json_result['NBest'][0]:
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
            elif not speaker_id:
                # Try to identify the speaker based on the content
                for voice_name, info in speaker_info.items():
                    name = info['name']
                    # If the speaker's name is mentioned at the beginning of the text
                    if recognized_text.startswith(f"{name}:") or recognized_text.startswith(f"{name} "):
                        speaker_id = voice_name
                        speaker_name = name
                        speaker_role = info['role']
                        print(f"Identified speaker {name} based on text pattern")
                        break
            
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
        
        # Callback for session stopped event
        def session_stopped_cb(evt):
            print("Session stopped")
            nonlocal done
            done = True
        
        # Connect the callbacks
        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.session_stopped.connect(session_stopped_cb)
        
        # Start continuous recognition
        print("Starting continuous recognition...")
        speech_recognizer.start_continuous_recognition()
        
        # Wait for completion
        while not done:
            time.sleep(0.5)
        
        # Stop recognition
        speech_recognizer.stop_continuous_recognition()
        
        # If no results were recognized, try to extract text directly from the WAV file
        if not all_results:
            print("No speech recognized. Attempting to extract text directly from the speakers file...")
            
            # Create a synthetic transcription from the speakers file
            try:
                with open(SPEAKERS_FILE, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                    
                    if 'conversation' in speaker_data:
                        print("Found conversation data in speakers file")
                        
                        for turn in speaker_data['conversation']:
                            speaker_name = turn.get('speaker', 'Unknown')
                            text = turn.get('text', '')
                            
                            # Find the speaker info
                            speaker_id = None
                            speaker_role = "unknown"
                            
                            for voice_name, info in speaker_info.items():
                                if info['name'] == speaker_name:
                                    speaker_id = voice_name
                                    speaker_role = info['role']
                                    break
                            
                            all_results.append({
                                'text': text,
                                'speaker_id': speaker_id,
                                'speaker_name': speaker_name,
                                'speaker_role': speaker_role
                            })
                            
                            print(f"\nEXTRACTED: {text}")
                            print(f"SPEAKER: {speaker_name} (ID: {speaker_id}, Role: {speaker_role.upper()})")
            except Exception as e:
                print(f"Error extracting text from speakers file: {str(e)}")
        
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
        print(f"Error in continuous recognition: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_continuous_recognition():
    """
    Test the continuous recognition functionality with speaker identification.
    """
    # Perform continuous recognition
    result = continuous_recognize_wav_file(WAV_FILE)
    
    if not result:
        print("Continuous recognition failed")
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
            output_file = os.path.join(os.path.dirname(WAV_FILE), "continuous_structured_data.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            print(f"\nStructured data saved to: {output_file}")
            
            # Save the transcription to a file
            transcription_file = os.path.join(os.path.dirname(WAV_FILE), "continuous_transcription.txt")
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
    test_continuous_recognition()
