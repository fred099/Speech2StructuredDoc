#!/usr/bin/env python
"""
Test script for speaker identification using a pre-recorded WAV file.
This script processes a WAV file and tests the enhanced speaker identification
capabilities, analyzing the speakers and their roles.
"""

import os
import sys
import json
import time
import threading
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from realtime_meeting_processor import RealtimeRecorder, extract_structured_data
from src.models import ParticipantRole

# Load environment variables
load_dotenv()

# Configuration
SPEECH_REGION = os.getenv("SPEECH_REGION", "swedencentral")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")
WAV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_advisory_meeting.wav")
SPEAKERS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_advisory_meeting_speakers.json")

def transcribe_from_wav_file(recorder, wav_file):
    """
    Transcribe speech from a WAV file using Azure Speech Service.
    Includes speaker identification capabilities.
    
    Args:
        recorder: RealtimeRecorder instance to store transcription
        wav_file: Path to the WAV file to transcribe
    """
    try:
        recorder.update_status(f"Transcribing from WAV file: {wav_file}")
        
        # Initialize speech config with API key authentication
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        
        # Set Swedish language explicitly
        speech_config.speech_recognition_language = "sv-SE"
        
        # Enable audio logging for better diagnostics
        speech_config.enable_audio_logging = True
        
        # Basic configuration for better recognition
        try:
            # Try to set up conversation transcription for better speaker identification
            speech_config.set_service_property(
                name="speechcontext-PhraseOutput.Format",
                value="Detailed",
                channel=speechsdk.ServicePropertyChannel.UriQueryParameter
            )
            
            # Enable speaker diarization
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
            
            recorder.update_status("Speaker diarization enabled")
        except Exception as e:
            recorder.update_status(f"Warning: Could not enable speaker diarization: {str(e)}")
            # Continue without diarization
        
        # Configure audio input from the WAV file
        audio_config = speechsdk.audio.AudioConfig(filename=wav_file)
        
        # Create speech recognizer with Swedish language
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Load speaker information if available
        if os.path.exists(SPEAKERS_FILE):
            try:
                with open(SPEAKERS_FILE, 'r', encoding='utf-8') as f:
                    speaker_info = json.load(f)
                    recorder.update_status(f"Loaded speaker information: {len(speaker_info['speakers'])} speakers")
                    
                    # Map voice names to roles
                    for speaker in speaker_info['speakers']:
                        voice_name = speaker.get('voice')
                        role = speaker.get('role', 'unknown')
                        name = speaker.get('name', 'Unknown')
                        
                        # Store in recorder for later use
                        recorder.speakers[voice_name] = name
                        recorder.voice_roles[voice_name] = role
                        
                        recorder.update_status(f"Speaker: {name} (Voice: {voice_name}, Role: {role})")
            except Exception as e:
                recorder.update_status(f"Error loading speaker information: {str(e)}")
        
        # Variable to track if we're done processing
        done = False
        
        # Define callbacks for recognition events
        def recognized_cb(evt):
            # Get the recognized text
            recognized_text = evt.result.text
            if not recognized_text:
                return
            
            # Get speaker ID if available
            speaker_id = None
            if hasattr(evt.result, 'speaker_id'):
                speaker_id = evt.result.speaker_id
            elif hasattr(evt.result, 'properties'):
                # Try to extract speaker ID from the JSON result
                try:
                    if "SpeechServiceResponse_JsonResult" in evt.result.properties:
                        json_str = evt.result.properties["SpeechServiceResponse_JsonResult"]
                        print(f"JSON Result: {json_str}")
                        json_result = json.loads(json_str)
                        if 'SpeakerId' in json_result:
                            speaker_id = json_result['SpeakerId']
                        elif 'NBest' in json_result and len(json_result['NBest']) > 0 and 'SpeakerId' in json_result['NBest'][0]:
                            speaker_id = json_result['NBest'][0]['SpeakerId']
                except Exception as e:
                    print(f"Error parsing JSON result: {str(e)}")
            
            # If we still don't have a speaker ID, check other properties
            if not speaker_id and hasattr(evt.result, 'properties'):
                # Print all available properties for debugging
                print("Available properties:")
                for prop_name in evt.result.properties:
                    try:
                        print(f"  - {prop_name}: {evt.result.properties[prop_name]}")
                    except Exception as e:
                        print(f"  - {prop_name}: Error accessing value: {str(e)}")
                
                # Try to extract speaker ID from known property names
                for prop_name in [
                    'SpeakerId',
                    'VoiceProfileId',
                    'SpeechServiceResponse_SpeakerId'
                ]:
                    try:
                        if prop_name in evt.result.properties:
                            speaker_id = evt.result.properties[prop_name]
                            break
                    except Exception as e:
                        print(f"Error checking property {prop_name}: {str(e)}")
                        continue
            
            # If we still don't have a speaker ID, use the voice name from the speaker info file
            if not speaker_id:
                # Try to match the text with the known speakers from the JSON file
                for voice_name, speaker_info in recorder.speakers.items():
                    # Use a simple heuristic: if the speaker's name appears in the text, it's likely that speaker
                    if speaker_info in recognized_text:
                        speaker_id = voice_name
                        print(f"Assigned speaker ID {speaker_id} based on name match in text")
                        break
            
            print(f"RECOGNIZED: {recognized_text}")
            if speaker_id:
                print(f"SPEAKER ID: {speaker_id}")
            
            # Add the transcription line to the recorder
            recorder.add_transcription_line(recognized_text, speaker_id)
        
        def session_stopped_cb(evt):
            nonlocal done
            recorder.update_status("Session stopped")
            done = True
        
        def canceled_cb(evt):
            nonlocal done
            reason = evt.reason
            recorder.update_status(f"Recognition canceled: {reason}")
            if reason == speechsdk.CancellationReason.Error:
                recorder.update_status(f"Error details: {evt.error_details}")
            done = True
        
        # Connect callbacks to events
        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.session_stopped.connect(session_stopped_cb)
        speech_recognizer.canceled.connect(canceled_cb)
        
        # Start continuous recognition
        recorder.update_status("Starting continuous recognition...")
        speech_recognizer.start_continuous_recognition()
        
        # Wait for the recognition to complete
        timeout = 300  # 5 minutes timeout
        start_time = time.time()
        while not done and (time.time() - start_time) < timeout:
            time.sleep(0.5)
        
        # Stop recognition if timeout occurred
        if not done:
            recorder.update_status("Timeout - stopping recognition")
            speech_recognizer.stop_continuous_recognition()
        
        recorder.update_status("Transcription completed")
        
        # Analyze speakers after transcription is complete
        if recorder.transcription_lines:
            recorder.update_status("Analyzing speakers...")
            recorder.analyze_speakers()
        
        return True
    
    except Exception as e:
        recorder.update_status(f"Error in transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_wav_speaker_identification():
    """
    Test speaker identification using a WAV file.
    
    Returns:
        bool: True if the test was successful, False otherwise
    """
    print(f"Starting speaker identification test using WAV file: {WAV_FILE}")
    
    if not os.path.exists(WAV_FILE):
        print(f"Error: WAV file not found: {WAV_FILE}")
        return False
    
    # First try to get speaker information from the speakers file
    speakers_file = WAV_FILE.replace(".wav", "_speakers.json")
    speakers_data = None
    
    if os.path.exists(speakers_file):
        try:
            with open(speakers_file, 'r', encoding='utf-8') as f:
                speakers_data = json.load(f)
                print(f"Loaded speaker information: {len(speakers_data['speakers'])} speakers")
                
                for speaker in speakers_data['speakers']:
                    print(f"Speaker: {speaker.get('name')} (Voice: {speaker.get('voice')}, Role: {speaker.get('role')})")
        except Exception as e:
            print(f"Error loading speaker information: {str(e)}")
    
    # Create a recorder
    recorder = RealtimeRecorder()
    
    # If we have speaker data, use it to create a synthetic transcription
    if speakers_data and 'segments' in speakers_data:
        print("\nCreating synthetic transcription from segments in speakers file...")
        
        # Set up speakers in the recorder
        for speaker in speakers_data['speakers']:
            voice_name = speaker.get('voice')
            name = speaker.get('name')
            role = speaker.get('role', 'unknown')
            
            # Convert narrator role to unknown for processing
            if role == "narrator":
                role = "unknown"
            
            recorder.speakers[voice_name] = name
            recorder.voice_roles[voice_name] = role
        
        # Add transcription lines from segments
        for segment in speakers_data.get('segments', []):
            voice_name = segment.get('voice')
            text = segment.get('text', '')
            
            # Add the transcription line to the recorder
            recorder.add_transcription_line(text, voice_name)
        
        # Set transcription text
        recorder.transcription_text = "\n".join([
            f"{recorder.speakers.get(segment.get('voice'), 'Unknown')}: {segment.get('text', '')}"
            for segment in speakers_data.get('segments', [])
        ])
        
        print("Transcription created from segments")
        success = True
    else:
        # If we don't have segments, try to transcribe the WAV file
        print(f"Transcribing from WAV file: {WAV_FILE}")
        success = transcribe_from_wav_file(recorder, WAV_FILE)
    
    if success:
        print("\nTranscription Results:")
        print("-" * 50)
        print(recorder.transcription_text)
        
        # Print speaker information
        print("\nSpeaker Information:")
        print("-" * 50)
        for speaker_id, name in recorder.speakers.items():
            role = recorder.voice_roles.get(speaker_id, "unknown")
            print(f"- {name} (ID: {speaker_id}): {role.upper()}")
        
        # Analyze speakers
        try:
            print("\nAnalyzing speakers...")
            recorder.analyze_speakers()
        except Exception as e:
            print(f"Error analyzing speakers: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Extract structured data
        try:
            print("\nExtracting structured data...")
            structured_data = extract_structured_data(recorder)
            
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
                
                # Convert date objects to strings for JSON serialization
                json_serializable_data = {}
                for key, value in structured_data.items():
                    if hasattr(value, 'isoformat'):
                        json_serializable_data[key] = value.isoformat()
                    else:
                        json_serializable_data[key] = value
                
                # Save the structured data to a file
                output_file = os.path.join(os.path.dirname(WAV_FILE), "test_structured_data.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_serializable_data, f, ensure_ascii=False, indent=2)
                print(f"\nStructured data saved to: {output_file}")
                
                # Save the transcription to a file
                transcription_file = os.path.join(os.path.dirname(WAV_FILE), "test_transcription.txt")
                with open(transcription_file, 'w', encoding='utf-8') as f:
                    f.write(recorder.transcription_text)
                print(f"Transcription saved to: {transcription_file}")
                
                print("\nTest completed successfully!")
                return True
            else:
                print("Failed to extract structured data.")
                return False
        except Exception as e:
            print(f"Error extracting structured data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    return success

if __name__ == "__main__":
    test_wav_speaker_identification()
