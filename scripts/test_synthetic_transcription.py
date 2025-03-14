#!/usr/bin/env python
"""
Test script for speaker identification using a synthetic transcription created from
the speakers file. This allows testing the speaker identification functionality
without relying on speech recognition.
"""

import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from realtime_meeting_processor import extract_structured_data, RealtimeRecorder
from src.models import ParticipantRole

# Load environment variables
load_dotenv()

# Configuration
SPEAKERS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_advisory_meeting_speakers.json")
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_synthetic_transcription():
    """
    Create a synthetic transcription from the speakers file.
    
    Returns:
        dict: Dictionary containing transcription results and speaker information
    """
    print(f"Creating synthetic transcription from speakers file: {SPEAKERS_FILE}")
    
    if not os.path.exists(SPEAKERS_FILE):
        print(f"Error: Speakers file not found: {SPEAKERS_FILE}")
        return None
    
    try:
        # Load the speakers file
        with open(SPEAKERS_FILE, 'r', encoding='utf-8') as f:
            speaker_data = json.load(f)
        
        # Extract speaker information
        speakers = {}
        voice_roles = {}
        
        for speaker in speaker_data.get('speakers', []):
            voice_name = speaker.get('voice')
            name = speaker.get('name')
            role = speaker.get('role', 'unknown')
            
            # Convert narrator role to unknown for processing
            if role == "narrator":
                role = "unknown"
                
            speakers[voice_name] = name
            voice_roles[voice_name] = role
            
            print(f"Speaker: {name} (Voice: {voice_name}, Role: {role})")
        
        # Create transcription lines from segments
        transcription_lines = []
        
        for segment in speaker_data.get('segments', []):
            voice_name = segment.get('voice')
            text = segment.get('text', '')
            role = segment.get('role', 'unknown')
            
            # Get speaker name
            speaker_name = speakers.get(voice_name, "Unknown Speaker")
            
            transcription_lines.append({
                'text': text,
                'speaker_id': voice_name,
                'speaker_name': speaker_name,
                'speaker_role': role
            })
            
            print(f"\nSEGMENT: {text}")
            print(f"SPEAKER: {speaker_name} (ID: {voice_name}, Role: {role.upper()})")
        
        # Combine all results into a single transcription
        transcription_text = "\n".join([f"{r['speaker_name']} ({r['speaker_role'].upper()}): {r['text']}" for r in transcription_lines])
        
        # Create a result object
        result = {
            'transcription_text': transcription_text,
            'transcription_lines': transcription_lines,
            'speakers': speakers,
            'voice_roles': voice_roles
        }
        
        return result
    
    except Exception as e:
        print(f"Error creating synthetic transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_recorder_from_synthetic_data(synthetic_data):
    """
    Create a RealtimeRecorder object from synthetic transcription data.
    
    Args:
        synthetic_data: Dictionary containing synthetic transcription data
        
    Returns:
        RealtimeRecorder: A recorder object with the synthetic data
    """
    # Create a new recorder
    recorder = RealtimeRecorder()
    
    # Set the transcription text
    recorder.transcription_text = synthetic_data['transcription_text']
    
    # Add the transcription lines
    for line in synthetic_data['transcription_lines']:
        recorder.add_transcription_line(line['text'], line['speaker_id'])
    
    # Set the speakers
    recorder.speakers = synthetic_data['speakers']
    
    # Set the voice roles
    recorder.voice_roles = synthetic_data['voice_roles']
    
    return recorder

def test_synthetic_transcription():
    """
    Test the speaker identification functionality using a synthetic transcription.
    """
    # Create synthetic transcription
    synthetic_data = create_synthetic_transcription()
    
    if not synthetic_data:
        print("Failed to create synthetic transcription")
        return False
    
    # Print transcription results
    print("\nTranscription Results:")
    print("-" * 50)
    print(synthetic_data['transcription_text'])
    
    # Print speaker information
    print("\nSpeaker Information:")
    print("-" * 50)
    for speaker_id, name in synthetic_data['speakers'].items():
        role = synthetic_data['voice_roles'].get(speaker_id, "unknown")
        print(f"- {name} (ID: {speaker_id}): {role.upper()}")
    
    # Create a recorder from the synthetic data
    recorder = create_recorder_from_synthetic_data(synthetic_data)
    
    # Analyze speakers
    print("\nAnalyzing speakers...")
    try:
        speaker_analysis = recorder.analyze_speakers()
        
        if speaker_analysis:
            print("\nSpeaker Analysis:")
            print("-" * 50)
            for speaker_id, role in speaker_analysis.items():
                speaker_name = synthetic_data['speakers'].get(speaker_id, "Unknown")
                print(f"- {speaker_name} (ID: {speaker_id}): {role.upper()}")
            
            # Save the speaker analysis to a file
            analysis_file = os.path.join(OUTPUT_DIR, "synthetic_speaker_analysis.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(speaker_analysis, f, ensure_ascii=False, indent=2)
            print(f"\nSpeaker analysis saved to: {analysis_file}")
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
            output_file = os.path.join(OUTPUT_DIR, "synthetic_structured_data.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_serializable_data, f, ensure_ascii=False, indent=2)
            print(f"\nStructured data saved to: {output_file}")
            
            # Save the transcription to a file
            transcription_file = os.path.join(OUTPUT_DIR, "synthetic_transcription.txt")
            with open(transcription_file, 'w', encoding='utf-8') as f:
                f.write(synthetic_data['transcription_text'])
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
    test_synthetic_transcription()
