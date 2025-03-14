#!/usr/bin/env python
"""
Test script to process a dynamically generated advisory meeting audio file,
identify different speakers, and determine which are advisors and which are clients.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime as dt
import copy
import re

import azure.cognitiveservices.speech as speechsdk
import requests
from dotenv import load_dotenv

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try to import from the project
    from utils.llm import get_completion as system_get_completion
    from utils.models import CompletionRequest
except ImportError:
    # Try to import from webscraper-rag
    try:
        from webscraper_rag.llm.azure_openai_provider import AzureOpenAIProvider
        from webscraper_rag.llm.models import CompletionRequest
        from webscraper_rag.config import config
    except ImportError:
        # Define a fallback CompletionRequest class
        class CompletionRequest:
            def __init__(self, prompt, system_message, model="fast", temperature=0.7, max_tokens=800):
                self.prompt = prompt
                self.system_message = system_message
                self.model = model
                self.temperature = temperature
                self.max_tokens = max_tokens

# Load environment variables
load_dotenv()

# Import the generate_test_audio function from the generate script
try:
    from generate_test_advisory_meeting import generate_test_audio
    
    def generate_test_advisory_meeting_audio():
        """Wrapper for generate_test_audio function."""
        print("Generating test audio meeting using Azure OpenAI and Speech...")
        output_filename = "test_advisory_meeting.wav"
        success = generate_test_audio(output_filename=output_filename)
        if success:
            return output_filename
        else:
            raise Exception("Failed to generate test audio file")
except ImportError as e:
    print(f"Warning: Could not import generate_test_audio: {e}")
    def generate_test_advisory_meeting_audio():
        """Fallback function if we can't import the real one."""
        print("Warning: Using dummy implementation. No actual audio file will be created.")
        # Create an empty file as a placeholder
        with open("test_advisory_meeting.wav", "wb") as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        return "test_advisory_meeting.wav"

def get_completion(request: CompletionRequest):
    """
    Get a completion from Azure OpenAI with API key authentication.
    
    Args:
        request (CompletionRequest): The request object with prompt, system message, etc.
        
    Returns:
        str: The completion text
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv("AZURE_OPENAI_API_KEY")
    }
    
    # Map logical model name to actual deployment
    model_name = request.model
    if model_name == "fast":
        model_name = os.getenv("AZURE_OPENAI_FAST_MODEL", "gpt-4o-mini")  # Using gpt-4o-mini for fast model
    elif model_name == "smart":
        model_name = os.getenv("AZURE_OPENAI_CAPABLE_MODEL", "gpt-4o")  # Using gpt-4o for smart model
    
    # Prepare the API request
    api_url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{model_name}/chat/completions?api-version={os.getenv('AZURE_OPENAI_API_VERSION', '2023-12-01-preview')}"
    
    messages = []
    if request.system_message:
        messages.append({"role": "system", "content": request.system_message})
    messages.append({"role": "user", "content": request.prompt})
    
    data = {
        "messages": messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "n": 1
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        
        # Print raw response for debugging
        print(f"Raw API Response: {response.text}")
        
        response_json = response.json()
        completion = response_json["choices"][0]["message"]["content"]
        return completion
    except Exception as e:
        print(f"Error calling Azure OpenAI API: {str(e)}")
        if response:
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
        return f"Error: {str(e)}"

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
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
    
    if not api_key or not endpoint:
        print("Error: Azure OpenAI API key or endpoint not found in environment variables")
        return "Error: Azure OpenAI API key or endpoint not found in environment variables"
    
    # Get the deployment name from environment variables
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not deployment_name:
        print("Error: AZURE_OPENAI_DEPLOYMENT not found in environment variables")
        return "Error: AZURE_OPENAI_DEPLOYMENT not found in environment variables"
    
    print(f"Using deployment: {deployment_name}")
    
    # Prepare the request
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    # Construct the messages array
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
        result = response.json()
        
        # Extract the completion text
        if "choices" in result and len(result["choices"]) > 0:
            completion_text = result["choices"][0]["message"]["content"]
            return completion_text
        else:
            print("Error: No completion choices found in response")
            return "Error: No completion choices found in response"
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}")
        return f"Error making API request: {str(e)}"
    except json.JSONDecodeError as e:
        print(f"Error parsing API response: {str(e)}")
        return f"Error parsing API response: {str(e)}"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}"

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
            - Provide investment advice and recommendations
            - Explain financial concepts and products
            - Ask about financial goals and risk tolerance
            - Use professional financial terminology
            
            Clients typically:
            - Ask questions about investments and financial products
            - Share personal financial information and goals
            - Express concerns or preferences about investments
            - Seek clarification on financial concepts
            
            Analyze each speaker's utterances and determine their role.
            Return your analysis as a JSON object with the following structure:
            {
                "roles": {
                    "speaker_id_1": "advisor",
                    "speaker_id_2": "client"
                },
                "confidence": {
                    "speaker_id_1": 0.9,
                    "speaker_id_2": 0.8
                },
                "reasoning": {
                    "speaker_id_1": "This speaker uses financial terminology typical of an advisor.",
                    "speaker_id_2": "This speaker asks questions about investments and shares personal financial goals, which is typical of a client."
                }
            }
            """
            
            # Create a string representation of the utterances
            utterances_str = ""
            for speaker_id, texts in self.utterances.items():
                utterances_str += f"Speaker {speaker_id}:\n"
                for text in texts:
                    utterances_str += f"- {text}\n"
                utterances_str += "\n"
            
            user_prompt = f"""
            Analyze the following conversation between financial advisors and clients.
            Determine which speakers are financial advisors and which are clients.
            
            {utterances_str}
            
            Return your analysis as a JSON object as specified.
            """
            
            # Create the completion request
            completion_request = CompletionRequest(
                prompt=user_prompt,
                system_message=system_prompt,
                model="reasoning",  # Use reasoning model for this complex task
                temperature=0.3,
                max_tokens=1000
            )
            
            try:
                # Get the completion
                completion = get_completion_func(completion_request)
                
                # Parse the JSON response
                try:
                    result = json.loads(completion)
                    
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
                    except Exception:
                        pass
                    
                    # If we still couldn't parse the response, make a best guess based on the utterances
                    print("Making a best guess for speaker roles based on utterances")
                    for speaker_id, texts in self.utterances.items():
                        # Simple heuristic: if the speaker mentions portfolio, investments, or uses financial terms,
                        # they're likely an advisor
                        advisor_indicators = ["portfölj", "investering", "tillgång", "fond", "aktie", "obligation", 
                                             "avkastning", "risk", "marknad", "rekommenderar", "strategi"]
                        
                        advisor_score = 0
                        for text in texts:
                            for indicator in advisor_indicators:
                                if indicator.lower() in text.lower():
                                    advisor_score += 1
                        
                        # If they use multiple advisor indicators, classify as advisor
                        if advisor_score >= 2:
                            self.roles[speaker_id] = "advisor"
                            self.confidence[speaker_id] = 0.7
                            self.reasoning[speaker_id] = "This speaker uses financial terminology typical of an advisor."
                        else:
                            self.roles[speaker_id] = "client"
                            self.confidence[speaker_id] = 0.6
                            self.reasoning[speaker_id] = "This speaker does not use enough financial terminology to be classified as an advisor."
                    
                    return True
            except Exception as e:
                print(f"Error during speaker analysis: {str(e)}")
                
                # Make a best guess based on the utterances
                print("Making a best guess for speaker roles based on utterances")
                for speaker_id, texts in self.utterances.items():
                    # Simple heuristic: if the speaker mentions portfolio, investments, or uses financial terms,
                    # they're likely an advisor
                    advisor_indicators = ["portfölj", "investering", "tillgång", "fond", "aktie", "obligation", 
                                         "avkastning", "risk", "marknad", "rekommenderar", "strategi"]
                    
                    advisor_score = 0
                    for text in texts:
                        for indicator in advisor_indicators:
                            if indicator.lower() in text.lower():
                                advisor_score += 1
                    
                    # If they use multiple advisor indicators, classify as advisor
                    if advisor_score >= 2:
                        self.roles[speaker_id] = "advisor"
                        self.confidence[speaker_id] = 0.7
                        self.reasoning[speaker_id] = "This speaker uses financial terminology typical of an advisor."
                    else:
                        self.roles[speaker_id] = "client"
                        self.confidence[speaker_id] = 0.6
                        self.reasoning[speaker_id] = "This speaker does not use enough financial terminology to be classified as an advisor."
                
                return True
    
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
            self.analysis_thread = None
    
    def get_results(self):
        """
        Get the results of the speaker analysis.
        
        Returns:
            dict: A dictionary containing the roles, confidence, and reasoning for each speaker
        """
        return {
            "roles": self.roles,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }

def process_audio_file_with_speaker_analysis(audio_file_path, timeout=60, output_dir='.', ground_truth_path=None):
    """
    Process an audio file with speaker analysis to identify speakers and their roles.
    
    Args:
        audio_file_path (str): Path to the audio file to process
        timeout (int, optional): Timeout in seconds for recognition
        output_dir (str, optional): Directory to save output files
        ground_truth_path (str, optional): Path to ground truth file
        
    Returns:
        dict: A dictionary containing the results of the analysis
    """
    print(f"Processing audio file: {audio_file_path}")
    
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
    speech_config.speech_recognition_language = "sv-SE"
    speech_config.enable_dictation()
    speech_config.request_word_level_timestamps()
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText")
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary, "true")
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_OutputFormatOption, "Detailed")
    
    # Enable speaker recognition
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US", "sv-SE"])
    
    # Set up audio config
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    
    # Create the speech recognizer with speaker diarization
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
    
    # Set the maximum number of speakers to detect (adjust as needed)
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
    
    # Define callbacks for recognition events
    def handle_final_result(evt):
        """Handle final recognition result."""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            # Get the result details
            result_json = json.loads(evt.result.json)
            
            # Extract speaker ID - use the Speaker field from the result
            # Look for the speaker ID in different possible locations in the result
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
    
    # Subscribe to events
    transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config,
        audio_config=audio_config)
    
    transcriber.transcribed.connect(handle_final_result)
    transcriber.canceled.connect(handle_canceled)
    transcriber.session_stopped.connect(handle_session_stopped)
    
    # Start recognition
    print("Starting recognition...")
    start_time = time.time()
    transcriber.start_transcribing_async()

    # Wait for recognition to complete or timeout
    print(f"Waiting for recognition to complete (timeout: {timeout} seconds)...")
    done.wait(timeout=timeout)
    
    # Stop recognition
    transcriber.stop_transcribing_async()
    
    # Stop parallel analysis
    speaker_analyzer.stop_parallel_analysis()
    
    # Check if we have any transcription
    if not transcription_lines:
        print("No transcription generated")
        # Return empty results
        results = {
            "speakers": {},
            "roles": {},
            "transcription": [],
            "speaker_infos": []
        }
        
        # Save empty results to file
        if speakers_info_path:
            with open(speakers_info_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Speaker information saved to: {speakers_info_path}")
            
        return results
    
    # Perform one final analysis to ensure we have the most up-to-date results
    print("\n" + "="*30 + " FINAL ANALYSIS " + "="*30)
    print("Performing final analysis of speaker roles...")
    
    # Map display names to speaker IDs for consistent output
    speaker_display_names = {}
    for speaker_id, display_name in speakers.items():
        speaker_display_names[speaker_id] = display_name
    
    # Perform final analysis
    final_analysis_result = speaker_analyzer.analyze_with_llm(get_completion_func)
    print(f"Final analysis complete. Success: {final_analysis_result}")
    
    # Get speaker roles
    speaker_roles = speaker_analyzer.get_results()
    
    # Save speaker information to file
    if speakers_info_path:
        # Enhance the speaker roles with display names for better readability
        enhanced_roles = copy.deepcopy(speaker_roles)
        enhanced_roles["display_names"] = {speaker_id: name for speaker_id, name in speaker_display_names.items()}
        
        with open(speakers_info_path, "w") as f:
            json.dump(enhanced_roles, f, indent=2)
        print(f"Speaker information saved to: {speakers_info_path}")
    
    # Create results dictionary
    results = {
        "speakers": speakers,
        "roles": speaker_roles.get("roles", {}),
        "confidence": speaker_roles.get("confidence", {}),
        "reasoning": speaker_roles.get("reasoning", {}),
        "transcription": transcription_lines,
        "speaker_infos": []
    }
    
    # Create speaker info objects
    for speaker_id, speaker_name in speakers.items():
        role = speaker_roles.get("roles", {}).get(speaker_id, "unknown")
        confidence = speaker_roles.get("confidence", {}).get(speaker_id, 0.0)
        reasoning = speaker_roles.get("reasoning", {}).get(speaker_id, "")
        
        speaker_info = {
            "id": speaker_id,
            "display_name": speaker_name,
            "role": role,
            "confidence": confidence,
            "reasoning": reasoning,
            "utterance_count": len(speaker_analyzer.utterances.get(speaker_id, []))
        }
        results["speaker_infos"].append(speaker_info)
    
    # Save transcription to file
    if transcription_path:
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcription_lines))
        print(f"Transcription saved to: {transcription_path}")
    
    # Compare with ground truth if available
    if ground_truth_path and os.path.exists(ground_truth_path):
        print("\n=== Comparing with Ground Truth ===")
        try:
            with open(ground_truth_path, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
            
            # Extract ground truth speaker roles
            ground_truth_roles = {}
            if isinstance(ground_truth, dict) and "speakers" in ground_truth:
                for speaker in ground_truth["speakers"]:
                    if isinstance(speaker, dict) and "id" in speaker and "role" in speaker:
                        ground_truth_roles[speaker["id"]] = speaker["role"]
            
            # Compare detected roles with ground truth
            for speaker_id, role in results["roles"].items():
                ground_truth_role = ground_truth_roles.get(speaker_id, "unknown")
                print(f"Speaker {speakers[speaker_id]} (ID: {speaker_id}):")
                print(f"  Detected Role: {role}")
                print(f"  Ground Truth Role: {ground_truth_role}")
                print(f"  Match: {role.lower() == ground_truth_role.lower()}")
        except Exception as e:
            print(f"Error comparing with ground truth: {str(e)}")
    
    return results

def test_dynamic_speaker_identification():
    """
    Test the speaker identification and role determination with a dynamically generated meeting.
    """
    print("Testing dynamic speaker identification with the generated advisory meeting audio file")
    
    # Path to the generated audio file
    audio_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_advisory_meeting.wav")
    
    # In a real-world scenario, we won't have speaker information
    # Process the audio file without speaker information
    results = process_audio_file_with_speaker_analysis(audio_file)
    
    # Check if we got any results
    if results and "speakers" in results and len(results["speakers"]) > 0:
        print("\nSuccessfully identified speakers and their roles:")
        
        # Count speakers with assigned roles
        speakers_with_roles = 0
        for speaker_num, role_info in results.get("roles", {}).items():
            if role_info.get("role") in ["advisor", "client"]:
                speakers_with_roles += 1
                speaker_name = f"Speaker {speaker_num}"
                role = role_info.get("role", "unknown")
                confidence = role_info.get("confidence", 0.0)
                print(f"- {speaker_name}: {role.upper()} (Confidence: {confidence:.2f})")
        
        if speakers_with_roles > 0:
            print(f"\nSuccessfully identified roles for {speakers_with_roles} speakers")
            return True
        else:
            print("\nNo speaker roles were identified")
            return False
    else:
        print("\nFailed to identify any speakers or their roles")
        return False

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test dynamic speaker identification.')
    parser.add_argument('--generate', action='store_true', help='Generate a test audio file first')
    parser.add_argument('--audio-file', type=str, help='Path to existing audio file to process', 
                        default='test_advisory_meeting.wav')
    parser.add_argument('--speakers-info', type=str, help='Path to speakers info JSON file',
                        default='test_advisory_meeting_speakers.json')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds for recognition', 
                        default=60)
    parser.add_argument('--use-ground-truth', action='store_true', 
                        help='Use ground truth data from speakers info file instead of speech recognition')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files', default='.')
    parser.add_argument('--ground-truth', type=str, help='Path to ground truth file', default=None)
    args = parser.parse_args()
    
    audio_file_path = args.audio_file
    speakers_info_path = args.speakers_info
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate test file if requested
    if args.generate:
        try:
            print("Generating test audio file...")
            audio_file_path = generate_test_advisory_meeting_audio()
            print(f"Test audio file generated: {audio_file_path}")
        except Exception as e:
            print(f"Error generating test audio file: {str(e)}")
            return
    
    # Process the audio file
    print(f"Processing audio file: {audio_file_path}")
    
    if args.use_ground_truth and os.path.exists(speakers_info_path):
        # Use ground truth data from speakers info file
        print(f"Using ground truth data from {speakers_info_path}")
        
        # Load speaker information
        with open(speakers_info_path, 'r', encoding='utf-8') as f:
            speakers_data = json.load(f)
        
        # Initialize speaker analyzer
        speaker_analyzer = SpeakerAnalyzer()
        
        # Add utterances from ground truth data
        speakers = {}
        transcription_lines = []
        
        for i, segment in enumerate(speakers_data.get('segments', [])):
            role = segment.get('role')
            voice = segment.get('voice')
            text = segment.get('text')
            
            # Find the speaker name
            speaker_name = None
            for speaker in speakers_data.get('speakers', []):
                if speaker.get('voice') == voice:
                    speaker_name = speaker.get('name')
                    break
            
            if not speaker_name:
                speaker_name = f"Speaker {i+1}"
            
            # Create a speaker ID
            speaker_id = f"speaker_{i+1}"
            
            # Add to speakers dictionary
            speakers[speaker_id] = speaker_name
            
            # Add utterance to speaker analyzer
            speaker_analyzer.add_utterance(speaker_id, text)
            
            # Add to transcription
            timestamp = dt.now().strftime("%H:%M:%S")
            formatted_line = f"[{timestamp}] {speaker_name}: {text}"
            transcription_lines.append(formatted_line)
            print(formatted_line)
        
        # Initialize Azure OpenAI for LLM analysis
        try:
            # Helper function to get completion from Azure OpenAI
            def get_completion(request):
                """Get completion function that works with our system pattern."""
                return get_completion_with_api_key(request)
        except Exception as e:
            print(f"Warning: Cannot initialize Azure OpenAI: {str(e)}")
            def get_completion(request):
                return "LLM analysis not available."
        
        # Analyze speakers with LLM
        print("\nPerforming LLM analysis of speaker roles...")
        speaker_analyzer.start_parallel_analysis(get_completion)
        speaker_analyzer.stop_parallel_analysis()
        speaker_roles = speaker_analyzer.get_results()
        
        # Create results dictionary
        results = {
            "speakers": speakers,
            "roles": speaker_roles["roles"],
            "transcription": transcription_lines,
            "speaker_infos": []
        }
        
        # Create speaker info objects
        for speaker_id, speaker_name in speakers.items():
            role = speaker_roles["roles"].get(speaker_id, "unknown")
            confidence = speaker_roles["confidence"].get(speaker_id, 0.0)
            reasoning = speaker_roles["reasoning"].get(speaker_id, "")
            
            speaker_info = {
                "id": speaker_id,
                "display_name": speaker_name,
                "role": role,
                "confidence": confidence,
                "reasoning": reasoning,
                "utterance_count": len(speaker_analyzer.utterances.get(speaker_id, []))
            }
            results["speaker_infos"].append(speaker_info)
        
    else:
        # Use speech recognition
        results = process_audio_file_with_speaker_analysis(
            audio_file_path=audio_file_path, 
            timeout=args.timeout,
            output_dir=args.output_dir,
            ground_truth_path=args.ground_truth
        )
    
    # Print results
    print("\n=== Speaker Analysis Results ===")
    
    # Check if we have valid results
    if results is None or "speakers" not in results:
        print("No valid results were generated.")
        return
    
    print(f"Total speakers detected: {len(results['speakers'])}")
    
    if "speaker_infos" in results:
        for speaker_info in results["speaker_infos"]:
            print(f"\nSpeaker: {speaker_info['display_name']} (ID: {speaker_info['id']})")
            print(f"Role: {speaker_info['role']}")
            print(f"Confidence: {speaker_info['confidence']:.2f}")
            if "reasoning" in speaker_info:
                print(f"Reasoning: {speaker_info['reasoning']}")
            print(f"Utterances: {speaker_info['utterance_count']}")
    
    # If ground truth is available, compare with it
    if args.ground_truth and os.path.exists(args.ground_truth):
        print("\n=== Comparing with Ground Truth ===")
        try:
            with open(args.ground_truth, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            # Extract ground truth speaker roles
            ground_truth_roles = {}
            if isinstance(ground_truth, dict) and "speakers" in ground_truth:
                for speaker in ground_truth["speakers"]:
                    if isinstance(speaker, dict) and "id" in speaker and "role" in speaker:
                        ground_truth_roles[speaker["id"]] = speaker["role"]
            
            # Compare detected roles with ground truth
            for speaker_id, speaker_name in results["speakers"].items():
                detected_role = "unknown"
                for speaker_info in results["speaker_infos"]:
                    if speaker_info["id"] == speaker_id:
                        detected_role = speaker_info["role"]
                        break
                
                ground_truth_role = ground_truth_roles.get(speaker_id, "unknown")
                
                print(f"Speaker {speaker_name} (ID: {speaker_id}):")
                print(f"  Detected Role: {detected_role}")
                print(f"  Ground Truth Role: {ground_truth_role}")
                print(f"  Match: {detected_role.lower() == ground_truth_role.lower()}")
        except Exception as e:
            print(f"Error comparing with ground truth: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save the results to a file
    output_file = os.path.join(args.output_dir, "speaker_analysis_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

print_lock = threading.Lock()

if __name__ == "__main__":
    main()
