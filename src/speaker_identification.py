"""
Speaker identification module for analyzing speakers in audio conversations.
This module provides functionality to identify speakers and determine their roles
(advisor or client) in financial advisory meetings.
"""

import os
import json
import re
import time
import threading
import requests
from datetime import datetime as dt
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT
from src.openai_client import ChatMessage

# Load environment variables
load_dotenv()

# Import models if available, otherwise define CompletionRequest here
try:
    from src.models import CompletionRequest, SpeakerInfo, SpeakerAnalysisResult
    from src.azure_openai_provider import CompletionRequest
except ImportError:
    # Define a minimal version for standalone use
    class CompletionRequest(BaseModel):
        """Simple Pydantic model for completion requests when models module is not available."""
        prompt: str
        system_message: Optional[str] = None
        temperature: float = 0.7
        max_tokens: int = 1000


class SpeakerAnalyzer:
    """
    Class for analyzing speakers in a conversation and determining their roles.
    """
    
    def __init__(self):
        """Initialize the SpeakerAnalyzer."""
        self.utterances = {}  # Dictionary of speaker_id -> list of utterances
        self.roles = {}  # Dictionary of speaker_id -> role
        self.confidence = {}  # Dictionary of speaker_id -> confidence score
        self.reasoning = {}  # Dictionary of speaker_id -> reasoning
        self.known_speakers = set()  # Set of known speaker IDs
        self.analysis_thread = None
        self.stop_analysis = False
        self.analysis_lock = threading.Lock()
        self.min_utterances_per_speaker = 2  # Minimum utterances needed for analysis
        self.last_utterance_count = {}  # Track utterance counts for parallel analysis
    
    def add_utterance(self, speaker_id, text):
        """
        Add an utterance for a speaker and check if it's a new speaker.
        
        Args:
            speaker_id (str): The ID of the speaker
            text (str): The text of the utterance
            
        Returns:
            bool: True if this is a new speaker, False otherwise
        """
        with self.analysis_lock:
            # Check if this is a new speaker
            is_new_speaker = speaker_id not in self.known_speakers
            
            if is_new_speaker:
                print(f"New speaker detected: {speaker_id}")
            
            self.known_speakers.add(speaker_id)
            
            # Add utterance to the speaker's list
            if speaker_id not in self.utterances:
                self.utterances[speaker_id] = []
            
            self.utterances[speaker_id].append(text)
            
            return is_new_speaker
    
    def get_utterance_count(self, speaker_id):
        """
        Get the number of utterances for a speaker.
        
        Args:
            speaker_id (str): The ID of the speaker
            
        Returns:
            int: The number of utterances for the speaker
        """
        return len(self.utterances.get(speaker_id, []))
    
    def analyze_with_llm(self, get_completion_func):
        """
        Analyze the speakers using an LLM to determine their roles.
        
        Args:
            get_completion_func (callable): A function that takes a CompletionRequest and returns a completion
            
        Returns:
            bool: True if the analysis was successful, False otherwise
        """
        with self.analysis_lock:
            # Check if we have enough utterances to analyze
            if not self.utterances:
                print("No utterances to analyze")
                return False
            
            # Prepare the prompt
            system_prompt = """
            You are an AI assistant that analyzes financial advisory meeting transcripts.
            Your task is to determine which speakers are financial advisors and which are clients.
            
            Financial advisors typically:
            - Use professional financial terminology
            - Provide advice and recommendations
            - Explain investment concepts
            - Ask questions to understand client needs
            - Present options and strategies
            
            Clients typically:
            - Ask questions about investments
            - Express concerns or goals
            - Share personal financial information
            - Respond to advisor recommendations
            - Make decisions based on advice
            
            Analyze the provided utterances and determine the role of each speaker.
            Return your analysis as a JSON object with the following structure:
            {
                "roles": {
                    "speaker_id": "advisor" or "client"
                },
                "confidence": {
                    "speaker_id": confidence_score (0.0 to 1.0)
                },
                "reasoning": {
                    "speaker_id": "Brief explanation of why you classified this speaker as advisor or client"
                }
            }
            """
            
            # Build the prompt with utterances for each speaker
            prompt = "Here are the utterances from each speaker in a financial advisory meeting:\n\n"
            
            for speaker_id, texts in self.utterances.items():
                prompt += f"Speaker {speaker_id}:\n"
                for text in texts:
                    prompt += f"- \"{text}\"\n"
                prompt += "\n"
            
            prompt += "Analyze these utterances and determine which speakers are financial advisors and which are clients."
            
            # Create the completion request using Pydantic model
            request = CompletionRequest(
                prompt=prompt,
                system_message=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=1000
            )
            
            # Get the completion
            try:
                print("Analyzing speakers with LLM...")
                completion = get_completion_func(request)
                
                # Parse the completion as JSON
                try:
                    # Extract JSON from the response if it's wrapped in markdown code blocks
                    if "```json" in completion:
                        json_content = completion.split("```json")[1].split("```")[0].strip()
                    elif "```" in completion:
                        json_content = completion.split("```")[1].split("```")[0].strip()
                    else:
                        json_content = completion.strip()
                    
                    # Try to parse the response as JSON
                    result = json.loads(json_content)
                    
                    # Update the roles, confidence, and reasoning
                    if "roles" in result:
                        self.roles.update(result["roles"])
                    if "confidence" in result:
                        self.confidence.update(result["confidence"])
                    if "reasoning" in result:
                        self.reasoning.update(result["reasoning"])
                    
                    return True
                except json.JSONDecodeError as e:
                    print(f"Error parsing LLM response as JSON: {str(e)}")
                    print(f"Raw response: {completion}")
                    
                    # Try to extract JSON from the response if it's embedded in text
                    try:
                        # Look for JSON-like patterns in the response
                        match = re.search(r'({.*})', completion, re.DOTALL)
                        if match:
                            json_str = match.group(1)
                            result = json.loads(json_str)
                            
                            # Update the roles, confidence, and reasoning
                            if "roles" in result:
                                self.roles.update(result["roles"])
                            if "confidence" in result:
                                self.confidence.update(result["confidence"])
                            if "reasoning" in result:
                                self.reasoning.update(result["reasoning"])
                            
                            return True
                    except (json.JSONDecodeError, AttributeError) as e:
                        print(f"Error extracting JSON from response: {str(e)}")
                        return False
                
            except Exception as e:
                print(f"Error getting completion from LLM: {str(e)}")
                return False
    
    def start_parallel_analysis(self, get_completion_func, min_utterances_per_speaker=2):
        """
        Start a parallel thread for analyzing speakers.
        
        Args:
            get_completion_func (callable): A function that takes a CompletionRequest and returns a completion
            min_utterances_per_speaker (int, optional): Minimum utterances needed per speaker for analysis
        """
        self.min_utterances_per_speaker = min_utterances_per_speaker
        self.stop_analysis = False
        self.last_utterance_count = {}
        
        def analysis_thread_func():
            """Function to run in the parallel analysis thread."""
            print("Starting parallel speaker analysis thread...")
            
            while not self.stop_analysis:
                # Check if we have new utterances to analyze
                should_analyze = False
                
                with self.analysis_lock:
                    # Check if we have new utterances for any speaker
                    current_counts = {spk_id: len(utterances) for spk_id, utterances in self.utterances.items()}
                    
                    # Compare with last counts
                    for spk_id, count in current_counts.items():
                        last_count = self.last_utterance_count.get(spk_id, 0)
                        if count > last_count and count >= self.min_utterances_per_speaker:
                            should_analyze = True
                            break
                    
                    # Update last counts
                    if should_analyze:
                        self.last_utterance_count = current_counts.copy()
                
                if should_analyze:
                    print("Analyzing speakers in parallel thread...")
                    self.analyze_with_llm(get_completion_func)
                
                # Sleep for a short time before checking again
                time.sleep(1)
            
            print("Parallel speaker analysis thread stopped")
        
        # Start the analysis thread
        self.analysis_thread = threading.Thread(target=analysis_thread_func)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def stop_parallel_analysis(self):
        """Stop the parallel analysis thread."""
        if self.analysis_thread:
            self.stop_analysis = True
            self.analysis_thread.join(timeout=2)
    
    def get_analysis_results(self):
        """
        Get the current analysis results.
        
        Returns:
            dict: A dictionary containing the roles, confidence, and reasoning for each speaker
        """
        with self.analysis_lock:
            return {
                "roles": self.roles,
                "confidence": self.confidence,
                "reasoning": self.reasoning
            }
    
    def get_results(self):
        """
        Get the results of the speaker analysis.
        
        Returns:
            dict: Dictionary with roles, confidence, and reasoning
        """
        with self.analysis_lock:
            return {
                "roles": self.roles,
                "confidence": self.confidence,
                "reasoning": self.reasoning
            }
    
    def get_results_json(self):
        """
        Get the results of the speaker analysis as a JSON string.
        
        Returns:
            str: JSON string with roles, confidence, and reasoning
        """
        with self.analysis_lock:
            results = {
                "roles": self.roles,
                "confidence": self.confidence,
                "reasoning": self.reasoning
            }
            return json.dumps(results, ensure_ascii=False)


def get_completion_with_api_key(request):
    """
    Get a completion from Azure OpenAI API using the API key from environment variables.
    
    Args:
        request (CompletionRequest): The completion request object
        
    Returns:
        str: The completion text
    """
    # Get API key and endpoint from environment variables
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = AZURE_OPENAI_ENDPOINT
    api_version = AZURE_OPENAI_API_VERSION
    
    if not api_key or not endpoint:
        print("Error: Azure OpenAI API key or endpoint not found in environment variables")
        return "Error: Azure OpenAI API key or endpoint not found in environment variables"
    
    # Get the deployment name from environment variables
    deployment_name = AZURE_OPENAI_DEPLOYMENT
    
    print(f"Using deployment: {deployment_name}")
    
    # Prepare the request
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    # Prepare the messages
    messages = []
    
    # Add system message if provided
    if request.system_message:
        messages.append({
            "role": "system",
            "content": request.system_message
        })
    
    # Add user message
    messages.append({
        "role": "user",
        "content": request.prompt
    })
    
    # Prepare the request body
    body = {
        "messages": messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens
    }
    
    # Make the API request
    try:
        url = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
        print(f"Making request to: {url}")
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        response_json = response.json()
        
        # Extract the completion text
        completion = response_json["choices"][0]["message"]["content"]
        
        return completion
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}")
        return f"Error making API request: {str(e)}"
    except json.JSONDecodeError as e:
        print(f"Error parsing API response: {str(e)}")
        return f"Error parsing API response: {str(e)}"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}"


def configure_diarization(speech_config):
    """
    Configure speech recognition for speaker diarization.
    
    Args:
        speech_config (speechsdk.SpeechConfig): The speech config to configure
        
    Returns:
        speechsdk.SpeechConfig: The configured speech config
    """
    # Configure for speaker diarization
    speech_config.speech_recognition_language = "sv-SE"
    speech_config.enable_dictation()
    speech_config.request_word_level_timestamps()
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText")
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary, "true")
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_OutputFormatOption, "Detailed")
    
    # Set timeouts
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
        value="5000")
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
        value="5000")
    
    # Enable diarization
    speech_config.set_service_property(
        name="speechcontext-PhraseDetection.SpeakerDiarization.Enable", 
        value="true",
        channel=speechsdk.ServicePropertyChannel.UriQueryParameter)
    
    # Set the maximum number of speakers to detect
    speech_config.set_service_property(
        name="speechcontext-PhraseDetection.SpeakerDiarization.MaxSpeakerCount", 
        value="3",  # Expect up to 3 speakers
        channel=speechsdk.ServicePropertyChannel.UriQueryParameter)
    
    # Set the minimum speaker separation in milliseconds
    speech_config.set_service_property(
        name="speechcontext-PhraseDetection.SpeakerDiarization.MinimumSpeakerSeparationMs", 
        value="500",  # 500ms minimum separation between speakers
        channel=speechsdk.ServicePropertyChannel.UriQueryParameter)
    
    # Enable intermediate diarization results
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults, 
        "true")
    
    return speech_config


def identify_speakers_from_audio(audio_file_path, timeout=60, output_dir=None):
    """
    Process an audio file to identify speakers and their roles.
    
    Args:
        audio_file_path (str): Path to the audio file to process
        timeout (int, optional): Timeout in seconds for recognition
        output_dir (str, optional): Directory to save output files
        
    Returns:
        dict: A dictionary containing the results of the analysis
    """
    print(f"Processing audio file for speaker identification: {audio_file_path}")
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up output paths
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    transcription_path = os.path.join(output_dir, f"{base_name}_transcription.txt") if output_dir else None
    speakers_info_path = os.path.join(output_dir, f"{base_name}_speakers.json") if output_dir else None
    
    # Set up speech config
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get("SPEECH_API_KEY"),
        region=os.environ.get("SPEECH_REGION", "swedencentral"))
    
    # Configure for speaker diarization
    speech_config = configure_diarization(speech_config)
    
    print("Speaker diarization enabled via service properties")
    
    # Initialize variables for tracking speakers and transcription
    speakers = {}
    transcription_lines = []
    speaker_analyzer = SpeakerAnalyzer()
    
    # Set up a completion event to track when recognition is done
    done = threading.Event()
    
    # Define get_completion function for LLM analysis
    def get_completion_func(request):
        """Get completion function that works with our system pattern"""
        return get_completion_with_api_key(request)
    
    # Start parallel analysis with a 2-utterance minimum
    print("Starting parallel speaker analysis...")
    speaker_analyzer.start_parallel_analysis(get_completion_func, min_utterances_per_speaker=2)
    
    # Set up a lock for thread-safe printing
    print_lock = threading.Lock()
    
    # Record the start time for timestamps
    start_time = time.time()
    
    # Define callbacks for recognition events
    def handle_final_result(evt):
        """Handle final recognition result."""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            # Get the result details
            result_json = json.loads(evt.result.json)
            
            # Extract speaker ID - look for the speaker ID in different possible locations in the result
            speaker_id = None
            
            # Try to get speaker ID from NBest results
            if "NBest" in result_json and result_json["NBest"]:
                for item in result_json["NBest"]:
                    if "Speaker" in item and item["Speaker"]:
                        speaker_id = str(item["Speaker"])
                        break
            
            # If not found in NBest, try the top level
            if not speaker_id and "Speaker" in result_json:
                speaker_id = str(result_json["Speaker"])
            
            # If still not found, check for SpeakerId
            if not speaker_id and "SpeakerId" in result_json:
                speaker_id = str(result_json["SpeakerId"])
            
            # If still no speaker ID, use a default
            if not speaker_id:
                speaker_id = "Unknown"
            
            # Extract the recognized text
            text = result_json.get("DisplayText", "")
            
            # Get the timestamp
            offset = result_json.get("Offset", 0) / 10000000  # Convert from 100ns to seconds
            timestamp = dt.fromtimestamp(start_time + offset).strftime("%H:%M:%S")
            
            # Use print_lock to ensure clean output
            with print_lock:
                # Check if this is a new speaker
                if speaker_id not in speakers:
                    speakers[speaker_id] = f"Speaker {len(speakers) + 1}"
                    print(f"Detected new speaker: {speakers[speaker_id]} (ID: {speaker_id})")
                
                # Add the utterance to the transcription
                line = f"[{timestamp}] {speakers[speaker_id]}: {text}"
                transcription_lines.append(line)
                print(line)
            
            # Add the utterance to the speaker analyzer for role analysis
            new_speaker_detected = speaker_analyzer.add_utterance(speaker_id, text)
            if new_speaker_detected:
                with print_lock:
                    print(f"New speaker detected: {speaker_id}. Triggering speaker role analysis...")
                speaker_analyzer.analyze_with_llm(get_completion_func)
        else:
            with print_lock:
                print(f"Recognition result was not successful: {evt.result.reason}")
            
    def handle_canceled(evt):
        print(f"Speech recognition canceled: {evt.result.reason}")
        done.set()
    
    def handle_session_stopped(evt):
        print("Speech recognition stopped")
        done.set()
    
    # Set up the transcriber
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config,
        audio_config=audio_config)
    
    # Subscribe to events
    transcriber.transcribed.connect(handle_final_result)
    transcriber.canceled.connect(handle_canceled)
    transcriber.session_stopped.connect(handle_session_stopped)
    
    # Start recognition
    print("Starting recognition...")
    transcriber.start_transcribing_async()
    
    # Wait for recognition to complete or timeout
    print(f"Waiting for recognition to complete (timeout: {timeout} seconds)...")
    done.wait(timeout=timeout)
    
    # Stop recognition
    transcriber.stop_transcribing_async()
    
    # Stop parallel analysis
    speaker_analyzer.stop_parallel_analysis()
    
    # Perform final analysis to ensure all speakers are analyzed
    print("\n============================== FINAL ANALYSIS ==============================")
    print("Performing final analysis of speaker roles...")
    speaker_analyzer.analyze_with_llm(get_completion_func)
    print(f"Final analysis complete. Success: {bool(speaker_analyzer.roles)}")
    
    # Save the transcription to a file if output_dir is provided
    if transcription_path:
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcription_lines))
        print(f"Transcription saved to: {transcription_path}")
    
    # Save the speaker information to a file if output_dir is provided
    if speakers_info_path:
        speaker_info = []
        for speaker_id, display_name in speakers.items():
            role = speaker_analyzer.roles.get(speaker_id, "unknown")
            confidence = speaker_analyzer.confidence.get(speaker_id, 0.0)
            reasoning = speaker_analyzer.reasoning.get(speaker_id, "")
            
            speaker_info.append({
                "id": speaker_id,
                "display_name": display_name,
                "role": role,
                "confidence": confidence,
                "reasoning": reasoning,
                "utterance_count": len(speaker_analyzer.utterances.get(speaker_id, []))
            })
        
        with open(speakers_info_path, "w", encoding="utf-8") as f:
            json.dump({"speakers": speaker_info}, f, indent=2)
        print(f"Speaker information saved to: {speakers_info_path}")
    
    # Print the analysis results
    print("\n=== Speaker Analysis Results ===")
    print(f"Total speakers detected: {len(speakers)}")
    print()
    
    for speaker_id, display_name in speakers.items():
        role = speaker_analyzer.roles.get(speaker_id, "unknown")
        confidence = speaker_analyzer.confidence.get(speaker_id, 0.0)
        reasoning = speaker_analyzer.reasoning.get(speaker_id, "")
        utterance_count = len(speaker_analyzer.utterances.get(speaker_id, []))
        
        print(f"Speaker: {display_name} (ID: {speaker_id})")
        print(f"Role: {role}")
        print(f"Confidence: {confidence}")
        print(f"Reasoning: {reasoning}")
        print(f"Utterances: {utterance_count}")
        print()
    
    # Save the results to a JSON file
    results = {
        "speakers": [],
        "transcription": transcription_lines
    }
    
    for speaker_id, display_name in speakers.items():
        results["speakers"].append({
            "id": speaker_id,
            "name": display_name,
            "role": speaker_analyzer.roles.get(speaker_id, "unknown"),
            "confidence": speaker_analyzer.confidence.get(speaker_id, 0.0),
            "reasoning": speaker_analyzer.reasoning.get(speaker_id, "")
        })
    
    results_path = os.path.join(output_dir, "speaker_analysis_results.json") if output_dir else "speaker_analysis_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Identify speakers and their roles in an audio file")
    parser.add_argument("--audio-file", required=True, help="Path to the audio file to process")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for recognition")
    parser.add_argument("--output-dir", default=".", help="Directory to save output files")
    
    args = parser.parse_args()
    
    identify_speakers_from_audio(args.audio_file, args.timeout, args.output_dir)
