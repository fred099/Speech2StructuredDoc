#!/usr/bin/env python
"""
Real-time meeting processor that records audio from the microphone,
transcribes it using Azure Speech Service, and extracts structured data.
This script can be used to demonstrate the system during a live meeting.
"""

import os
import sys
import json
import time
import requests
import threading
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

class RealtimeRecorder:
    """Class to handle recording and visualization of the real-time meeting processing."""
    
    def __init__(self):
        self.recording = False
        self.frame_buffer = []
        self.current_status = "Initializing..."
        self.transcription_buffer = []
        self.full_transcription = ""
        self.structured_data = None
        self.start_time = None
        self.processing_active = False
        
        # Create output directory if it doesn't exist
        os.makedirs("meeting_recordings", exist_ok=True)
        
        # Generate timestamp for the recording filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = os.path.join("meeting_recordings", f"meeting_recording_{timestamp}.mp4")
        self.transcription_filename = os.path.join("meeting_recordings", f"transcription_{timestamp}.txt")
        self.json_filename = os.path.join("meeting_recordings", f"structured_data_{timestamp}.json")
        
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
            
            # Add recording indicator
            if self.processing_active:
                indicator_color = (0, 255, 0)  # Green when active
                indicator_text = "● RECORDING"
            else:
                indicator_color = (0, 0, 255)  # Red when inactive
                indicator_text = "○ PAUSED"
            cv2.putText(frame, indicator_text, (SCREEN_WIDTH - 200, 30), FONT, FONT_SCALE, indicator_color, FONT_THICKNESS)
            
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
            
            # Add instructions at the bottom
            instructions = [
                "Press 'P' to pause/resume recording",
                "Press 'E' to extract structured data",
                "Press 'S' to save current transcription",
                "Press 'Q' to quit"
            ]
            y_pos = SCREEN_HEIGHT - 120
            cv2.putText(frame, "Controls:", (10, y_pos), FONT, FONT_SCALE, HIGHLIGHT_COLOR, FONT_THICKNESS)
            y_pos += 30
            for instruction in instructions:
                cv2.putText(frame, instruction, (20, y_pos), FONT, FONT_SCALE * 0.7, FONT_COLOR, FONT_THICKNESS - 1)
                y_pos += 25
            
            # Write the frame
            self.video_writer.write(frame)
            
            # Display the frame
            cv2.imshow("Real-time Meeting Processor", frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_recording()
                break
            elif key == ord('p'):
                self.toggle_processing()
            elif key == ord('e'):
                self.extract_data_now()
            elif key == ord('s'):
                self.save_transcription()
            
            # Sleep to maintain FPS
            time.sleep(1/FPS)
    
    def update_status(self, status):
        """Update the current status displayed in the recording."""
        self.current_status = status
        print(status)
    
    def add_transcription_line(self, line):
        """Add a line to the transcription buffer."""
        self.transcription_buffer.append(line)
        self.full_transcription += line + " "
        
        # Also save to the transcription file
        with open(self.transcription_filename, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    
    def set_structured_data(self, data):
        """Set the structured data to display."""
        self.structured_data = data
        
        # Save to the JSON file
        with open(self.json_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def toggle_processing(self):
        """Toggle the processing state (pause/resume)."""
        self.processing_active = not self.processing_active
        status = "resumed" if self.processing_active else "paused"
        self.update_status(f"Recording {status}")
    
    def extract_data_now(self):
        """Trigger immediate extraction of structured data from current transcription."""
        if not self.full_transcription.strip():
            self.update_status("Cannot extract data: No transcription available")
            return
            
        self.update_status("Extracting structured data from current transcription...")
        threading.Thread(target=self._extract_data_thread).start()
    
    def _extract_data_thread(self):
        """Thread function to extract structured data without blocking the UI."""
        try:
            structured_data_json = extract_structured_data(self.full_transcription)
            json_data = json.loads(structured_data_json)
            self.set_structured_data(json_data)
            self.update_status("Structured data extracted successfully")
        except Exception as e:
            self.update_status(f"Error extracting data: {str(e)}")
    
    def save_transcription(self):
        """Save the current transcription to a file."""
        if not self.full_transcription.strip():
            self.update_status("Cannot save: No transcription available")
            return
            
        try:
            # Already saving continuously, but we'll update the status
            self.update_status(f"Transcription saved to {self.transcription_filename}")
        except Exception as e:
            self.update_status(f"Error saving transcription: {str(e)}")
    
    def stop_recording(self):
        """Stop the recording process."""
        if not self.recording:
            return
            
        self.recording = False
        time.sleep(0.5)  # Give time for the recording thread to finish
        
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        print(f"Recording saved to {self.video_filename}")
        print(f"Transcription saved to {self.transcription_filename}")
        if self.structured_data:
            print(f"Structured data saved to {self.json_filename}")

def transcribe_from_microphone(recorder):
    """
    Continuously transcribe speech from the microphone using Azure Speech Service.
    """
    try:
        recorder.update_status("Initializing microphone...")
        
        # Initialize speech config with API key authentication
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        
        # Set Swedish language explicitly
        speech_config.speech_recognition_language = "sv-SE"
        
        # Configure audio input from the default microphone
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        
        # Create speech recognizer with Swedish language
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Variable to track if processing is active
        recorder.processing_active = True
        
        # Define callbacks
        def recognized_cb(evt):
            if recorder.processing_active and evt.result.text:
                recognized_text = evt.result.text
                print(f"RECOGNIZED: {recognized_text}")
                recorder.add_transcription_line(recognized_text)
        
        def session_stopped_cb(evt):
            recorder.update_status("Session stopped")
            recorder.processing_active = False
        
        def canceled_cb(evt):
            recorder.update_status(f"Recognition canceled: {evt.reason}")
            if evt.reason == speechsdk.CancellationReason.Error:
                recorder.update_status(f"Error details: {evt.error_details}")
            recorder.processing_active = False
        
        # Connect callbacks
        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.session_stopped.connect(session_stopped_cb)
        speech_recognizer.canceled.connect(canceled_cb)
        
        # Start continuous recognition
        recorder.update_status("Listening to microphone... (Press 'P' to pause/resume)")
        speech_recognizer.start_continuous_recognition()
        
        # Keep the program running until the recorder is stopped
        while recorder.recording:
            time.sleep(0.1)
        
        # Stop recognition when recorder is stopped
        speech_recognizer.stop_continuous_recognition()
        
    except Exception as e:
        error_msg = f"Error during speech transcription: {str(e)}"
        recorder.update_status(error_msg)
        print(error_msg)

def extract_structured_data(transcription):
    """
    Extract structured data from Swedish transcription text using Azure OpenAI with API key authentication.
    """
    try:
        print("Extracting structured data from transcription")
        
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
        Om mötet verkar pågå och inte är avslutat, extrahera ändå den information som finns tillgänglig hittills.
        Om datumet inte nämns explicit, använd dagens datum.
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
        print(f"Making request to: {url}")
        
        response = requests.post(url, headers=headers, json=payload)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            structured_data = result["choices"][0]["message"]["content"]
            print("Structured data extracted successfully")
            
            # Validate that the result is valid JSON
            try:
                json_data = json.loads(structured_data)
                
                # Fix key_points if it's an array
                if isinstance(json_data.get('key_points'), list):
                    json_data['key_points'] = ' '.join(json_data['key_points'])
                    # Update the structured_data with the fixed version
                    structured_data = json.dumps(json_data)
                    print("Fixed key_points format (converted from array to string)")
                
                # If meeting_date is null, set to today's date
                if not json_data.get('meeting_date'):
                    json_data['meeting_date'] = datetime.now().strftime("%Y-%m-%d")
                    structured_data = json.dumps(json_data)
                    print("Set missing meeting date to today's date")
                
                return structured_data
            except json.JSONDecodeError:
                print("Azure OpenAI returned invalid JSON")
                raise ValueError("Could not parse structured data as JSON")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            raise ValueError(f"API call failed with status code {response.status_code}")
            
    except Exception as e:
        print(f"Error extracting structured data: {str(e)}")
        raise

def run_realtime_processor():
    """Run the real-time meeting processor."""
    print("Starting real-time meeting processor...")
    
    # Check if OpenCV is installed
    try:
        import cv2
    except ImportError:
        print("OpenCV is required for this demo. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        print("OpenCV installed. Restarting script...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    # Initialize the recorder
    recorder = RealtimeRecorder()
    
    try:
        # Start recording
        recorder.start_recording()
        recorder.update_status("Real-time meeting processor started")
        
        # Start transcription in a separate thread
        transcription_thread = threading.Thread(target=transcribe_from_microphone, args=(recorder,))
        transcription_thread.daemon = True
        transcription_thread.start()
        
        # Wait for the recording thread to finish (when user quits)
        while recorder.recording:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Stop recording
        recorder.stop_recording()

if __name__ == "__main__":
    run_realtime_processor()
