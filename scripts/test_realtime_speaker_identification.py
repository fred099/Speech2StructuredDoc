#!/usr/bin/env python
"""
Test script for real-time speaker identification in meetings.
This script runs the real-time meeting processor and tests the enhanced
speaker identification capabilities during a live meeting.
"""

import os
import sys
import time
import threading
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from realtime_meeting_processor import RealtimeRecorder, transcribe_from_microphone, run_realtime_processor

# Load environment variables
load_dotenv()

def test_speaker_identification():
    """
    Test the speaker identification capabilities of the real-time meeting processor.
    """
    print("Starting real-time speaker identification test...")
    print("This test will record audio from your microphone and analyze speakers.")
    print("Speak clearly and have multiple people speak to test speaker identification.")
    print("Press Ctrl+C to stop the test.")
    
    try:
        # Run the real-time meeting processor
        run_realtime_processor()
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed. Check the meeting_recordings directory for results.")

def manual_speaker_test():
    """
    Run a manual test of speaker identification without the GUI.
    This is useful for debugging speaker identification issues.
    """
    print("Starting manual speaker identification test...")
    print("This test will record audio from your microphone and analyze speakers.")
    print("Speak clearly and have multiple people speak to test speaker identification.")
    print("Press Ctrl+C to stop the test.")
    
    try:
        # Create a recorder instance
        recorder = RealtimeRecorder()
        recorder.recording = True
        
        # Start transcription in a separate thread
        transcription_thread = threading.Thread(target=transcribe_from_microphone, args=(recorder,))
        transcription_thread.daemon = True
        transcription_thread.start()
        
        # Keep the main thread running to display status
        while True:
            time.sleep(1)
            print(f"Status: {recorder.current_status if hasattr(recorder, 'current_status') else 'Initializing...'}")
            
            # Display the most recent transcription line if available
            if hasattr(recorder, 'transcription_lines') and recorder.transcription_lines:
                print(f"Latest: {recorder.transcription_lines[-1]}")
            
            # Display speaker information if available
            if hasattr(recorder, 'speakers') and recorder.speakers:
                print("\nSpeaker Information:")
                for speaker_id, name in recorder.speakers.items():
                    role = recorder.voice_roles.get(speaker_id, "unknown") if hasattr(recorder, 'voice_roles') else "unknown"
                    print(f"- {name} (ID: {speaker_id}): {role.upper()}")
            
            # Display structured data if available
            if hasattr(recorder, 'structured_data') and recorder.structured_data:
                print("\nStructured Data:")
                print(f"Client: {recorder.structured_data.get('client_name', 'Unknown')}")
                print(f"Key Points: {recorder.structured_data.get('key_points', 'None')}")
                
                participants = recorder.structured_data.get('participants', [])
                if participants:
                    print("Participants:")
                    for p in participants:
                        if isinstance(p, dict):
                            print(f"- {p.get('name', 'Unknown')} ({p.get('role', 'unknown').upper()})")
                
                action_items = recorder.structured_data.get('action_items', [])
                if action_items:
                    print("Action Items:")
                    for item in action_items:
                        print(f"- {item}")
            
            print("\n" + "-" * 50)
    
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
        recorder.recording = False
        time.sleep(2)  # Give time for threads to clean up
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed. Check the meeting_recordings directory for results.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test real-time speaker identification")
    parser.add_argument("--manual", action="store_true", help="Run manual test without GUI")
    args = parser.parse_args()
    
    if args.manual:
        manual_speaker_test()
    else:
        test_speaker_identification()
