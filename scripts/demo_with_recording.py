#!/usr/bin/env python
"""
Demo script that records video and audio while processing a Swedish advisory meeting.
This script runs the full pipeline including audio playback, transcription, and structured data extraction,
while recording the screen and audio for demonstration purposes.
"""

import os
import sys
import json
import time
import requests
import threading
import subprocess
import cv2
import numpy as np
from datetime import datetime
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

# Recording settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 30.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)  # White
FONT_THICKNESS = 2
BG_COLOR = (0, 0, 0)  # Black background
HIGHLIGHT_COLOR = (0, 255, 0)  # Green for highlights

class DemoRecorder:
    """Class to handle recording of the demo process."""
    
    def __init__(self):
        self.recording = False
        self.frame_buffer = []
        self.current_status = "Initializing..."
        self.transcription_buffer = []
        self.structured_data = None
        self.start_time = None
        
        # Create output directory if it doesn't exist
        os.makedirs("demo_recordings", exist_ok=True)
        
        # Generate timestamp for the recording filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = os.path.join("demo_recordings", f"demo_recording_{timestamp}.mp4")
        
    def start_recording(self):
        """Start the screen recording process."""
        self.recording = True
        self.start_time = time.time()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.video_filename, 
            fourcc, 
            FPS, 
            (SCREEN_WIDTH, SCREEN_HEIGHT)
        )
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_frames)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        print(f"Started recording to {self.video_filename}")
        
    def _record_frames(self):
        """Record frames continuously while recording is active."""
        while self.recording:
            # Create a blank frame
            frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            
            # Add elapsed time
            elapsed_time = time.time() - self.start_time
            time_text = f"Elapsed Time: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
            cv2.putText(frame, time_text, (10, 30), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            
            # Add current status
            cv2.putText(frame, f"Status: {self.current_status}", (10, 70), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            
            # Add transcription (last few lines)
            y_pos = 120
            cv2.putText(frame, "Transcription:", (10, y_pos), FONT, FONT_SCALE, HIGHLIGHT_COLOR, FONT_THICKNESS)
            y_pos += 40
            
            # Show the last 10 lines of transcription
            visible_lines = self.transcription_buffer[-10:] if self.transcription_buffer else []
            for line in visible_lines:
                # Wrap text if too long
                if len(line) > 70:
                    parts = [line[i:i+70] for i in range(0, len(line), 70)]
                    for part in parts:
                        cv2.putText(frame, part, (20, y_pos), FONT, FONT_SCALE * 0.8, FONT_COLOR, FONT_THICKNESS - 1)
                        y_pos += 30
                else:
                    cv2.putText(frame, line, (20, y_pos), FONT, FONT_SCALE * 0.8, FONT_COLOR, FONT_THICKNESS - 1)
                    y_pos += 30
            
            # Add structured data if available
            if self.structured_data:
                y_pos = max(y_pos + 20, 400)  # Ensure some spacing
                cv2.putText(frame, "Structured Data:", (10, y_pos), FONT, FONT_SCALE, HIGHLIGHT_COLOR, FONT_THICKNESS)
                y_pos += 40
                
                # Client name and meeting date
                cv2.putText(frame, f"Client: {self.structured_data.get('client_name', 'N/A')}", (20, y_pos), FONT, FONT_SCALE * 0.8, FONT_COLOR, FONT_THICKNESS - 1)
                y_pos += 30
                cv2.putText(frame, f"Date: {self.structured_data.get('meeting_date', 'N/A')}", (20, y_pos), FONT, FONT_SCALE * 0.8, FONT_COLOR, FONT_THICKNESS - 1)
                y_pos += 30
                
                # Participants
                participants = self.structured_data.get('participants', [])
                if participants:
                    cv2.putText(frame, f"Participants: {', '.join(participants[:3])}", (20, y_pos), FONT, FONT_SCALE * 0.8, FONT_COLOR, FONT_THICKNESS - 1)
                    y_pos += 30
                
                # Action items (first 3)
                action_items = self.structured_data.get('action_items', [])
                if action_items:
                    cv2.putText(frame, "Action Items:", (20, y_pos), FONT, FONT_SCALE * 0.8, FONT_COLOR, FONT_THICKNESS - 1)
                    y_pos += 30
                    for i, item in enumerate(action_items[:3]):
                        if len(item) > 70:
                            item = item[:67] + "..."
                        cv2.putText(frame, f"• {item}", (30, y_pos), FONT, FONT_SCALE * 0.7, FONT_COLOR, FONT_THICKNESS - 1)
                        y_pos += 25
            
            # Write the frame
            self.video_writer.write(frame)
            
            # Sleep to maintain FPS
            time.sleep(1/FPS)
    
    def update_status(self, status):
        """Update the current status displayed in the recording."""
        self.current_status = status
        print(status)
    
    def add_transcription_line(self, line):
        """Add a line to the transcription buffer."""
        self.transcription_buffer.append(line)
    
    def set_structured_data(self, data):
        """Set the structured data to display."""
        self.structured_data = data
    
    def stop_recording(self):
        """Stop the recording process."""
        if not self.recording:
            return
            
        self.recording = False
        time.sleep(0.5)  # Give time for the recording thread to finish
        
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
            
        print(f"Recording saved to {self.video_filename}")
        
        # Open the video file with the default player
        try:
            os.startfile(self.video_filename)
        except:
            print(f"Video saved to {self.video_filename}. Please open it manually.")

def play_audio(audio_file_path):
    """
    Play audio file using Windows Media Player.
    """
    try:
        print(f"Playing audio file: {audio_file_path}")
        # Use PowerShell to play the audio file
        full_path = os.path.abspath(audio_file_path)
        subprocess.Popen(['powershell', '-c', f'(New-Object Media.SoundPlayer "{full_path}").PlaySync()'])
        print("Audio playback started")
    except Exception as e:
        print(f"Error playing audio: {str(e)}")

def transcribe_audio(audio_file_path, recorder):
    """
    Transcribe a Swedish audio file using Azure Speech Service with API key authentication.
    """
    try:
        recorder.update_status("Transcribing audio file...")
        
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
            recognized_text = evt.result.text
            print(f"RECOGNIZED: {recognized_text}")
            recorder.add_transcription_line(recognized_text)
            transcription += recognized_text + " "
        
        def session_stopped_cb(evt):
            nonlocal done
            print("Session stopped")
            recorder.update_status("Transcription completed")
            done = True
        
        def canceled_cb(evt):
            nonlocal done
            print(f"Recognition canceled: {evt.reason}")
            if evt.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {evt.error_details}")
                recorder.update_status(f"Error: {evt.error_details}")
            done = True
        
        # Connect callbacks
        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.session_stopped.connect(session_stopped_cb)
        speech_recognizer.canceled.connect(canceled_cb)
        
        # Start continuous recognition
        recorder.update_status("Starting continuous recognition...")
        speech_recognizer.start_continuous_recognition()
        
        # Wait for recognition to complete
        max_wait_time = 120  # Maximum wait time in seconds
        start_time = time.time()
        
        while not done and (time.time() - start_time) < max_wait_time:
            time.sleep(0.5)
        
        # Stop recognition
        speech_recognizer.stop_continuous_recognition()
        
        if not done:
            recorder.update_status("Timeout - stopping recognition")
        
        recorder.update_status("Transcription completed")
        return transcription.strip()
        
    except Exception as e:
        error_msg = f"Error during speech transcription: {str(e)}"
        recorder.update_status(error_msg)
        print(error_msg)
        raise

def extract_structured_data(transcription, recorder):
    """
    Extract structured data from Swedish transcription text using Azure OpenAI with API key authentication.
    """
    try:
        recorder.update_status("Extracting structured data from transcription")
        
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
        recorder.update_status(f"Making request to OpenAI API...")
        
        response = requests.post(url, headers=headers, json=payload)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            structured_data = result["choices"][0]["message"]["content"]
            recorder.update_status("Structured data extracted successfully")
            
            # Validate that the result is valid JSON
            try:
                json_data = json.loads(structured_data)
                
                # Fix key_points if it's an array
                if isinstance(json_data.get('key_points'), list):
                    json_data['key_points'] = ' '.join(json_data['key_points'])
                    # Update the structured_data with the fixed version
                    structured_data = json.dumps(json_data)
                    recorder.update_status("Fixed key_points format (converted from array to string)")
                
                recorder.set_structured_data(json_data)
                return structured_data
            except json.JSONDecodeError:
                error_msg = "Azure OpenAI returned invalid JSON"
                recorder.update_status(error_msg)
                raise ValueError("Could not parse structured data as JSON")
        else:
            error_msg = f"Error: {response.status_code}"
            recorder.update_status(error_msg)
            print(response.text)
            raise ValueError(f"API call failed with status code {response.status_code}")
            
    except Exception as e:
        error_msg = f"Error extracting structured data: {str(e)}"
        recorder.update_status(error_msg)
        print(error_msg)
        raise

def process_audio_file(audio_file_path, recorder):
    """
    Process an audio file through the full pipeline:
    1. Transcribe the audio
    2. Extract structured data from the transcription
    3. Validate against the Pydantic model
    """
    try:
        # Step 1: Transcribe the audio
        transcription = transcribe_audio(audio_file_path, recorder)
        recorder.update_status("Transcription completed")
        
        # Step 2: Extract structured data
        structured_data_json = extract_structured_data(transcription, recorder)
        recorder.update_status("Structured data extraction completed")
        
        # Step 3: Parse and validate with Pydantic
        structured_data = AudioFormData.model_validate_json(structured_data_json)
        recorder.update_status("Pydantic validation successful")
        
        # Display structured data details
        recorder.update_status(f"Client: {structured_data.client_name}")
        recorder.update_status(f"Meeting date: {structured_data.meeting_date}")
        
        # Step 4: Create the full processing result
        result = ProcessingResult(
            transcription=transcription,
            structured_data=structured_data,
            transcription_url=None,  # Would be set in production
            json_url=None  # Would be set in production
        )
        
        recorder.update_status("Processing completed successfully")
        
        return result
        
    except ValidationError as e:
        error_msg = f"Pydantic validation error: {str(e)}"
        recorder.update_status(error_msg)
        raise
    except Exception as e:
        error_msg = f"Error processing audio file: {str(e)}"
        recorder.update_status(error_msg)
        raise

def run_demo():
    """Run the demo with recording."""
    print("Starting Swedish advisory meeting processing demo with recording...")
    
    # Initialize the recorder
    recorder = DemoRecorder()
    
    # Path to the generated test audio file
    audio_file_path = "test_advisory_meeting.wav"
    
    try:
        # Start recording
        recorder.start_recording()
        recorder.update_status("Demo started")
        
        # Process the audio file
        result = process_audio_file(audio_file_path, recorder)
        
        # Save the result to a file for reference
        result_filename = "demo_processing_result.json"
        with open(result_filename, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        recorder.update_status(f"Results saved to {result_filename}")
        
        # Display success message
        recorder.update_status("Demo completed successfully!")
        
        # Give some time to show the final status
        time.sleep(3)
        
    except Exception as e:
        recorder.update_status(f"Demo failed: {str(e)}")
    finally:
        # Stop recording
        recorder.stop_recording()

if __name__ == "__main__":
    # Check if OpenCV is installed
    try:
        import cv2
    except ImportError:
        print("OpenCV is required for this demo. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        print("OpenCV installed. Restarting script...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    # Run the demo
    run_demo()
