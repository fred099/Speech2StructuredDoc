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
import re

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models import AudioFormData, ProcessingResult, CompletionRequest, SpeakerInfo, SpeakerAnalysisResult
from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
from src.speaker_identification import SpeakerAnalyzer, configure_diarization, get_completion_with_api_key

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
    
    def __init__(self, output_dir=None, azure_openai_provider=None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.recording = True
        self.processing_active = True  # Start with processing active by default
        self.transcription_lines = []
        self.transcription_text = ""
        self.structured_data = {}
        self.status_message = "Initializing..."
        self.speakers = {}  # Dictionary of speaker_id -> display_name
        self.voice_roles = {}  # Dictionary of speaker_id -> role
        self.speaker_analysis_complete = False
        self.azure_openai_provider = azure_openai_provider
        
        # Initialize our speaker analyzer with appropriate configuration
        self.speaker_analyzer = SpeakerAnalyzer()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(self.output_dir, "meeting_recordings"), exist_ok=True)
        
        # Set up filenames for output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transcription_filename = os.path.join(self.output_dir, "meeting_recordings", f"transcription_{timestamp}.txt")
        self.json_filename = os.path.join(self.output_dir, "meeting_recordings", f"structured_data_{timestamp}.json")
        self.video_filename = os.path.join(self.output_dir, "meeting_recordings", f"recording_{timestamp}.mp4")
        
        # Initialize OpenCV video writer
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.video_filename, self.fourcc, FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Initialize the window
        cv2.namedWindow('Real-time Meeting Processor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Meeting Processor', SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Start the speaker analysis thread if we have an OpenAI provider
        if self.azure_openai_provider:
            self._start_speaker_analysis_thread()
    
    def initialize_azure_openai_provider(self):
        """Initialize the Azure OpenAI provider for completions."""
        try:
            # Import the provider here to avoid circular imports
            from src.azure_openai_provider import AzureOpenAIProvider
            
            # Create the provider instance
            provider = AzureOpenAIProvider()
            self.update_status("Azure OpenAI provider initialized")
            return provider
        except Exception as e:
            print(f"Error initializing Azure OpenAI provider: {str(e)}")
            self.update_status(f"Error initializing Azure OpenAI provider: {str(e)}")
            return None
    
    def start_recording(self):
        """Start the screen recording process."""
        self.recording = True
        self.start_time = time.time()
        
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
            cv2.putText(frame, f"Status: {self.status_message}", (10, 70), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            
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
            visible_lines = self.transcription_lines[-10:] if self.transcription_lines else []
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
                    cv2.putText(frame, f"Participants: {', '.join([p['name'] for p in participants[:3]])}", (20, y_pos), FONT, FONT_SCALE * 0.8, FONT_COLOR, FONT_THICKNESS - 1)
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
            self.out.write(frame)
            
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
        self.status_message = status
        print(status)
    
    def add_transcription_line(self, text, speaker_id=None):
        """
        Add a line to the transcription with speaker identification if available.
        
        Args:
            text (str): The transcribed text
            speaker_id (str, optional): The ID of the speaker
        """
        # Get the current timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # If no speaker ID is provided, use "Unknown"
        if not speaker_id:
            speaker_id = "Unknown"
        
        # If this is a new speaker, add it to our speakers dictionary
        if speaker_id not in self.speakers:
            new_speaker_num = len(self.speakers) + 1
            self.speakers[speaker_id] = f"Speaker {new_speaker_num}"
            print(f"New speaker detected: {self.speakers[speaker_id]} (ID: {speaker_id})")
        
        # Format the line with timestamp and speaker
        speaker_name = self.speakers[speaker_id]
        line = f"[{timestamp}] {speaker_name}: {text}"
        
        # Add the line to our transcription
        self.transcription_lines.append(line)
        self.transcription_text = "\n".join(self.transcription_lines)
        
        # Save the transcription to a file
        with open(self.transcription_filename, "w", encoding="utf-8") as f:
            f.write(self.transcription_text)
        
        # Add the utterance to the speaker analyzer for role analysis
        if text.strip():  # Only add non-empty utterances
            is_new_speaker = self.speaker_analyzer.add_utterance(speaker_id, text)
            
            # If this is a new speaker with enough utterances, trigger analysis
            utterance_count = self.speaker_analyzer.get_utterance_count(speaker_id)
            if is_new_speaker and utterance_count >= 2:
                print(f"New speaker {speaker_id} has {utterance_count} utterances. Triggering analysis...")
                # We'll let the parallel analysis thread handle this
        
        # Update the UI
        self.update_status(f"Transcribing: {text[:30]}..." if len(text) > 30 else f"Transcribing: {text}")
    
    def analyze_speakers(self):
        """
        Analyze the speakers to determine their roles (advisor or client).
        Uses the SpeakerAnalyzer class for more robust analysis.
        """
        if self.speaker_analysis_complete:
            return
            
        self.update_status("Analyzing speakers to determine roles...")
        
        try:
            # Define a function to get completions that works with our SpeakerAnalyzer
            def get_completion_func(request):
                """Get completion function that works with our SpeakerAnalyzer."""
                if self.azure_openai_provider:
                    # Use the provider if available
                    return self.azure_openai_provider.get_completion(request)
                else:
                    # Fall back to direct API calls
                    return get_completion_with_api_key(request)
            
            # Force an analysis now
            success = self.speaker_analyzer.analyze_with_llm(get_completion_func)
            
            if success:
                # Get the analysis results
                results = self.speaker_analyzer.get_analysis_results()
                
                # Update our voice_roles dictionary
                self.voice_roles.update(results["roles"])
                
                # Mark analysis as complete
                self.speaker_analysis_complete = True
                
                # Print the results
                print("Speaker role analysis complete:")
                for speaker_id, role in self.voice_roles.items():
                    speaker_name = self.speakers.get(speaker_id, f"Speaker {speaker_id}")
                    print(f"- {speaker_name} (ID: {speaker_id}): {role.upper()}")
                
                # Create speaker info objects for the result
                speaker_info_list = []
                for speaker_id, speaker_name in self.speakers.items():
                    role = self.voice_roles.get(speaker_id, "unknown")
                    confidence = results["confidence"].get(speaker_id, 0.7)
                    reasoning = results["reasoning"].get(speaker_id, "")
                    
                    speaker_info = SpeakerInfo(
                        id=speaker_id,
                        name=speaker_name,
                        role=role,
                        confidence=confidence,
                        reasoning=reasoning,
                        utterance_count=len(self.speaker_analyzer.utterances.get(speaker_id, []))
                    )
                    speaker_info_list.append(speaker_info)
                
                # Save speaker analysis to a file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                analysis_file = os.path.join(self.output_dir, "meeting_recordings", f"speaker_analysis_{timestamp}.json")
                analysis_result = SpeakerAnalysisResult(
                    speakers=speaker_info_list,
                    transcription=self.transcription_lines
                )
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result.dict(), f, ensure_ascii=False, indent=2)
                print(f"Speaker analysis saved to: {analysis_file}")
                
                # Update the status
                self.update_status("Speaker roles identified successfully")
            else:
                self.update_status("Speaker role analysis failed")
                
        except Exception as e:
            self.update_status(f"Error analyzing speakers: {str(e)}")
            print(f"Error analyzing speakers: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _start_speaker_analysis_thread(self):
        """Start the speaker analysis thread for continuous speaker role analysis."""
        self.update_status("Starting speaker analysis thread...")
        
        # Define a function to get completions that works with our SpeakerAnalyzer
        def get_completion_func(request):
            """Get completion function that works with our SpeakerAnalyzer."""
            if self.azure_openai_provider:
                # Use the provider if available
                return self.azure_openai_provider.get_completion(request)
            else:
                # Fall back to direct API calls
                return get_completion_with_api_key(request)
        
        # Start the parallel analysis with a 2-utterance minimum
        self.speaker_analyzer.start_parallel_analysis(get_completion_func, min_utterances_per_speaker=2)
        self.update_status("Speaker analysis thread started")
    
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
        self.update_status(f"Processing {status}")
        print(f"Processing {status}")
    
    def extract_data_now(self):
        """Force extraction of structured data now."""
        self.update_status("Extracting structured data...")
        try:
            structured_data = extract_structured_data(self)
            if structured_data:
                self.set_structured_data(structured_data)
                self.update_status("Structured data extracted successfully")
            else:
                self.update_status("No structured data could be extracted")
        except Exception as e:
            self.update_status(f"Error extracting data: {str(e)}")
            print(f"Error extracting data: {str(e)}")
    
    def save_all_data(self):
        """Save all data (transcription, structured data, and speaker analysis)."""
        self.update_status("Saving all data...")
        
        # Save transcription
        self.save_transcription()
        
        # Save structured data
        if self.structured_data:
            try:
                # If we have a JSON string, parse it first
                if isinstance(self.structured_data, str):
                    try:
                        data_dict = json.loads(self.structured_data)
                    except json.JSONDecodeError:
                        data_dict = {"raw_text": self.structured_data}
                else:
                    data_dict = self.structured_data
                
                # Create a ProcessingResult object
                speaker_info_list = []
                for speaker_id, speaker_name in self.speakers.items():
                    role = self.voice_roles.get(speaker_id, "unknown")
                    
                    # Get confidence and reasoning if available
                    confidence = 0.7  # Default confidence
                    reasoning = ""
                    if hasattr(self.speaker_analyzer, 'get_analysis_results'):
                        results = self.speaker_analyzer.get_analysis_results()
                        if results:
                            confidence = results["confidence"].get(speaker_id, confidence)
                            reasoning = results["reasoning"].get(speaker_id, reasoning)
                    
                    speaker_info = SpeakerInfo(
                        id=speaker_id,
                        name=speaker_name,
                        role=role,
                        confidence=confidence,
                        reasoning=reasoning,
                        utterance_count=self.speaker_analyzer.get_utterance_count(speaker_id)
                    )
                    speaker_info_list.append(speaker_info)
                
                # Create the result object
                result = ProcessingResult(
                    client_name=data_dict.get("client_name", "Unknown"),
                    meeting_date=data_dict.get("meeting_date", datetime.now().strftime("%Y-%m-%d")),
                    key_points=data_dict.get("key_points", ""),
                    action_items=data_dict.get("action_items", []),
                    participants=data_dict.get("participants", []),
                    transcription=self.transcription_lines,
                    speaker_analysis=speaker_info_list
                )
                
                # Save to file
                with open(self.json_filename, 'w', encoding='utf-8') as f:
                    json.dump(result.dict(), f, ensure_ascii=False, indent=2)
                
                self.update_status(f"All data saved to {self.json_filename}")
                print(f"All data saved to {self.json_filename}")
            except Exception as e:
                self.update_status(f"Error saving structured data: {str(e)}")
                print(f"Error saving structured data: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def save_transcription(self):
        """Save the current transcription to a file."""
        if self.transcription_lines:
            try:
                with open(self.transcription_filename, 'w', encoding='utf-8') as f:
                    f.write(self.transcription_text)
                print(f"Transcription saved to {self.transcription_filename}")
            except Exception as e:
                print(f"Error saving transcription: {str(e)}")
    
    def cleanup(self):
        """Clean up resources."""
        # Release the video writer
        if hasattr(self, 'out') and self.out:
            self.out.release()
        
        # Close the window
        cv2.destroyAllWindows()
        
        # Stop the speaker analysis thread if running
        if hasattr(self.speaker_analyzer, 'stop_parallel_analysis'):
            self.speaker_analyzer.stop_parallel_analysis()
        
        print("Resources cleaned up")
    
    def stop_recording(self):
        """Stop the recording process."""
        if not self.recording:
            return
            
        self.recording = False
        time.sleep(0.5)  # Give time for the recording thread to finish
        
        if hasattr(self, 'out'):
            self.out.release()
        
        cv2.destroyAllWindows()
        print(f"Recording saved to {self.video_filename}")
        print(f"Transcription saved to {self.transcription_filename}")
        if self.structured_data:
            print(f"Structured data saved to {self.json_filename}")

def transcribe_from_microphone(recorder):
    """
    Continuously transcribe speech from the microphone using Azure Speech Service.
    Includes speaker diarization for identifying different speakers.
    
    Args:
        recorder (RealtimeRecorder): The recorder object to update with transcription
    """
    try:
        recorder.update_status("Initializing speech recognition...")
        
        # Create a speech config
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        speech_config.speech_recognition_language = "sv-SE"
        
        # Configure diarization
        configure_diarization(speech_config)
        
        # Create an audio config for microphone
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        
        # Create a conversation transcriber
        transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Set up event handlers
        def handle_session_started(evt):
            recorder.update_status(f"Session started: {evt}")
            print(f"Session started: {evt}")
        
        def handle_session_stopped(evt):
            recorder.update_status(f"Session stopped: {evt}")
            print(f"Session stopped: {evt}")
            transcriber.stop_transcribing_async()
        
        def handle_transcribed(evt):
            if not recorder.processing_active:
                return
                
            # Get the transcription text and speaker ID
            text = evt.result.text
            speaker_id = evt.result.speaker_id if evt.result.speaker_id else "Unknown"
            
            # Add to the recorder's transcription
            recorder.add_transcription_line(text, speaker_id)
            
            # Periodically extract structured data
            if len(recorder.transcription_lines) % 10 == 0 and len(recorder.transcription_lines) >= 20:
                try:
                    structured_data = extract_structured_data(recorder)
                    if structured_data:
                        recorder.set_structured_data(structured_data)
                except Exception as e:
                    print(f"Error extracting structured data: {str(e)}")
        
        # Connect the event handlers
        transcriber.session_started.connect(handle_session_started)
        transcriber.session_stopped.connect(handle_session_stopped)
        transcriber.transcribed.connect(handle_transcribed)
        
        # Start transcribing
        recorder.update_status("Starting transcription...")
        transcriber.start_transcribing_async()
        
        # Keep the transcriber running until recording is stopped
        while recorder.recording:
            time.sleep(0.1)
        
        # Stop transcribing
        transcriber.stop_transcribing_async()
        recorder.update_status("Transcription stopped")
        
    except Exception as e:
        recorder.update_status(f"Error in transcription: {str(e)}")
        print(f"Error in transcription: {str(e)}")
        import traceback
        traceback.print_exc()

def extract_structured_data(transcription):
    """
    Extract structured data from Swedish transcription text using Azure OpenAI with API key authentication.
    """
    try:
        # Get the transcription text
        if hasattr(transcription, 'transcription_text'):
            # If we're passed a recorder object
            transcription_text = transcription.transcription_text
            speaker_analyzer = transcription.speaker_analyzer if hasattr(transcription, 'speaker_analyzer') else None
        else:
            # If we're passed a string
            transcription_text = transcription
            speaker_analyzer = None
        
        # Skip if transcription is too short
        if not transcription_text or len(transcription_text.split()) < 20:
            return json.dumps({
                "client_name": "Waiting for more data...",
                "meeting_date": datetime.now().strftime("%Y-%m-%d"),
                "key_points": "Waiting for more data...",
                "action_items": [],
                "participants": []
            })
        
        # Prepare the prompt for Azure OpenAI
        system_message = """
        Du är en AI-assistent som analyserar transkriptioner från finansiella rådgivningsmöten.
        Din uppgift är att extrahera strukturerad information från transkriptionen.
        Svara endast med JSON-data som innehåller följande fält:
        
        - client_name: Namnet på kunden eller organisationen (t.ex. "Volvo Group", "Ericsson", "Familjen Andersson")
        - meeting_date: Mötesdatum i format YYYY-MM-DD (om det nämns, annars dagens datum)
        - key_points: En sammanfattning av de viktigaste punkterna som diskuterades
        - action_items: En lista med åtgärdspunkter eller nästa steg
        - participants: En lista med deltagare i mötet, med deras roll (rådgivare eller kund)
        
        Exempel på JSON-svar:
        {
            "client_name": "Volvo Group",
            "meeting_date": "2025-03-13",
            "key_points": "Diskuterade investeringsstrategier och pensionsplanering för Volvo Groups anställda.",
            "action_items": ["Schemalägg uppföljningsmöte", "Dela investeringsförslag", "Uppdatera portföljallokering"],
            "participants": [
                {"name": "Maria Johansson", "role": "advisor"},
                {"name": "Erik Andersson", "role": "client"},
                {"name": "Lena Karlsson", "role": "client"}
            ]
        }
        
        Svara endast med JSON-data, inga andra förklaringar.
        """
        
        # If we have speaker analysis results, include them in the prompt
        if speaker_analyzer and hasattr(speaker_analyzer, 'roles') and speaker_analyzer.roles:
            system_message += """
            
            VIKTIGT: Jag har redan identifierat roller för talarna i transkriptionen.
            Använd denna information för att fylla i participants-listan:
            """
            
            for speaker_id, role in speaker_analyzer.roles.items():
                speaker_name = f"Speaker {speaker_id}" if isinstance(speaker_id, int) else speaker_id
                system_message += f"\n- {speaker_name}: {role}"
        
        user_message = f"Här är transkriptionen från ett finansiellt rådgivningsmöte. Extrahera strukturerad information enligt instruktionerna:\n\n{transcription_text}"
        
        # Create a CompletionRequest
        completion_request = CompletionRequest(
            prompt=user_message,
            system_message=system_message,
            model="smart",  # Use smart model for better extraction
            temperature=0.3,
            max_tokens=1000
        )
        
        # Try to use the azure_openai_provider if available
        if hasattr(transcription, 'azure_openai_provider') and transcription.azure_openai_provider:
            response = transcription.azure_openai_provider.get_completion(completion_request)
            return response
        
        # Otherwise, fall back to direct API calls
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }
        
        api_url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        structured_data = result["choices"][0]["message"]["content"]
        
        # Try to parse the response as JSON to validate it
        try:
            json.loads(structured_data)
        except json.JSONDecodeError:
            # If it's not valid JSON, try to extract JSON from the response
            match = re.search(r'({.*})', structured_data, re.DOTALL)
            if match:
                structured_data = match.group(1)
        
        return structured_data
        
    except Exception as e:
        print(f"Error extracting structured data: {str(e)}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "error": str(e),
            "client_name": "Error occurred",
            "meeting_date": datetime.now().strftime("%Y-%m-%d"),
            "key_points": f"Error extracting data: {str(e)}",
            "action_items": [],
            "participants": []
        })

def run_realtime_processor():
    """Run the real-time meeting processor."""
    try:
        print("Starting real-time meeting processor...")
        print(f"Using Azure OpenAI deployment: {AZURE_OPENAI_DEPLOYMENT}")
        
        # Create output directory if it doesn't exist
        os.makedirs("meeting_recordings", exist_ok=True)
        
        # Initialize the recorder
        recorder = RealtimeRecorder()
        
        # Initialize Azure OpenAI provider
        azure_openai_provider = recorder.initialize_azure_openai_provider()
        recorder.azure_openai_provider = azure_openai_provider
        
        # Start recording
        recorder.start_recording()
        
        # Start transcription in a separate thread
        transcription_thread = threading.Thread(target=transcribe_from_microphone, args=(recorder,))
        transcription_thread.daemon = True
        transcription_thread.start()
        
        # Main loop to handle UI events
        while recorder.recording:
            key = cv2.waitKey(100)
            
            # Handle key presses
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                print("Quitting...")
                recorder.recording = False
            elif key == ord('p'):  # 'p' to pause/resume
                recorder.toggle_processing()
            elif key == ord('e'):  # 'e' to extract data now
                recorder.extract_data_now()
            elif key == ord('s'):  # 's' to analyze speakers
                recorder.analyze_speakers()
            elif key == ord('a'):  # 'a' to save all data
                recorder.save_all_data()
            elif key == ord('t'):  # 't' to save transcription
                recorder.save_transcription()
        
        # Wait for threads to finish
        transcription_thread.join(timeout=1)
        
        # Save the final results
        recorder.save_all_data()
        
        # Clean up
        recorder.cleanup()
        
        print("Real-time meeting processor finished.")
        
    except Exception as e:
        print(f"Error in real-time processor: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_realtime_processor()
