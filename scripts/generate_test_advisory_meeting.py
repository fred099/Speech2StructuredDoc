#!/usr/bin/env python
"""
Script to generate a test audio file simulating an advisory meeting in Swedish that can be
processed by the Speech2StructuredDoc application according to the Pydantic model.
Uses different voices for different speakers to test speaker identification.
The meeting content is dynamically generated using Azure OpenAI.
"""

import os
import sys
import json
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import datetime
import tempfile
import wave
import contextlib
import requests

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION

# Load environment variables
load_dotenv()

# Configuration
SPEECH_REGION = os.getenv("SPEECH_REGION", "swedencentral")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

# Define the available Swedish voices
ADVISOR_VOICE = "sv-SE-SofieNeural"     # Female advisor
CLIENT1_VOICE = "sv-SE-MattiasNeural"   # Male client
CLIENT2_VOICE = "sv-SE-HilleviNeural"   # Female client

def generate_meeting_script():
    """
    Use Azure OpenAI to generate a dynamic meeting script for a Swedish financial advisory meeting.
    
    Returns:
        dict: A dictionary containing the meeting script parts for each speaker
    """
    print("Generating dynamic meeting script using Azure OpenAI...")
    
    # Set up API call with API key authentication
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    
    # Get today's date in Swedish format
    today = datetime.date.today().strftime("%d %B, %Y").replace("March", "mars")
    
    # Define system prompt for generating a meeting script
    system_message = f"""
    Du är en AI-assistent som skapar realistiska mötesscript för finansiella rådgivningsmöten på svenska.
    Skapa ett detaljerat script för ett rådgivningsmöte mellan en finansiell rådgivare (Maria Johansson) från Söderberg & Partners 
    och två kunder (Erik Andersson och Lena Karlsson) från Volvo Group.
    
    Mötet äger rum den {today}.
    
    Scriptet ska innehålla:
    1. Diskussion om kundernas investeringsportfölj, inklusive specifika siffror och procentsatser
    2. Diskussion om ESG-investeringar och hållbarhet
    3. Pensionsplanering för båda kunderna
    4. Sammanfattning av huvudpunkter och nästa steg
    5. Avslutning av mötet
    
    Formatera scriptet så att det är tydligt vem som talar (Maria, Erik, Lena).
    Inkludera all nödvändig information som krävs för att extrahera strukturerad data enligt en Pydantic-modell:
    - Kundnamn (Volvo Group)
    - Mötesdatum ({today})
    - Deltagare (Maria Johansson, Erik Andersson, Lena Karlsson)
    - Huvudpunkter från mötet
    - Åtgärdspunkter/nästa steg
    
    VIKTIGT: Du måste returnera ett giltigt JSON-objekt utan några extra förklaringar eller text utanför JSON-objektet.
    Returnera scriptet i exakt följande JSON-format:
    {{
        "maria_parts": ["Del 1 för Maria", "Del 2 för Maria", ...],
        "erik_parts": ["Del 1 för Erik", "Del 2 för Erik", ...],
        "lena_parts": ["Del 1 för Lena", "Del 2 för Lena", ...]
    }}
    
    Svara ENDAST med JSON-objektet och inget annat.
    """
    
    # User prompt to request the meeting script
    user_message = "Skapa ett realistiskt mötesscript för ett finansiellt rådgivningsmöte på svenska."
    
    # Prepare the API request
    api_url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        # Make the API request
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Extract the generated script
        result = response.json()
        script_text = result["choices"][0]["message"]["content"]
        
        # Print the raw response for debugging
        print("Raw response from OpenAI:")
        print(script_text[:500] + "..." if len(script_text) > 500 else script_text)
        
        # Clean the response - remove any markdown code block indicators
        script_text = script_text.replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON response
        try:
            script_json = json.loads(script_text)
            print("Successfully generated dynamic meeting script")
            return script_json
        except json.JSONDecodeError:
            print("Error parsing JSON from OpenAI response. Using fallback approach.")
            # Try to extract JSON from the text if it's not properly formatted
            import re
            json_match = re.search(r'({.*})', script_text.replace('\n', ''), re.DOTALL)
            if json_match:
                try:
                    script_json = json.loads(json_match.group(1))
                    print("Successfully extracted JSON from response text")
                    
                    # Ensure the script has the required fields
                    if not all(key in script_json for key in ["maria_parts", "erik_parts", "lena_parts"]):
                        print("JSON is missing required fields, creating basic structure")
                        script_json = create_basic_script()
                    
                    return script_json
                except json.JSONDecodeError as e:
                    print(f"Failed to parse extracted JSON: {str(e)}")
            
            # If all parsing attempts fail, create a basic structure
            print("Creating basic script structure as fallback")
            return create_basic_script()
    
    except Exception as e:
        print(f"Error generating meeting script: {str(e)}")
        # Return a basic script structure as fallback
        return create_basic_script()

def create_basic_script():
    """Create a basic script structure as fallback."""
    return {
        "maria_parts": [
            "God morgon allihopa. Tack för att ni deltar i vårt kvartalsvisa finansiella rådgivningsmöte. Idag ska vi gå igenom er nuvarande investeringsportfölj och planera för framtiden.",
            "Er nuvarande portfölj består av 60% aktier, 30% obligationer och 10% alternativa investeringar. Avkastningen hittills i år är 7.5%.",
            "Baserat på vår analys rekommenderar jag att vi ökar andelen hållbara investeringar i er portfölj, särskilt inom förnybar energi och grön teknologi.",
            "Jag kommer att skicka över mer information om de hållbara fonderna och en detaljerad pensionsplan nästa vecka. Har ni några frågor innan vi avslutar?"
        ],
        "erik_parts": [
            "Tack Maria. Vi ser fram emot att höra dina rekommendationer.",
            "Hur ser riskprofilen ut för de hållbara investeringarna jämfört med vår nuvarande portfölj?",
            "Det låter bra. Vi är intresserade av att öka vår exponering mot förnybar energi."
        ],
        "lena_parts": [
            "Ja, särskilt angående våra hållbara investeringar.",
            "Dessutom skulle jag vilja diskutera våra pensionsplaner. Jag är 45 år och Erik är 47, så vi behöver en strategi som fungerar för oss båda.",
            "Tack för all information, Maria. Vi ser fram emot att få detaljerna nästa vecka."
        ]
    }

def synthesize_with_voice(text, voice_name, output_file):
    """Synthesize speech with a specific voice and save to a file."""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = voice_name
    file_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
    
    print(f"Syntetiserar tal med röst: {voice_name}")
    result = synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Tal syntetiserat framgångsrikt: {output_file}")
        return True
    else:
        print(f"Talsyntes misslyckades: {result.reason}")
        return False

def generate_test_audio(output_filename="test_advisory_meeting.wav", script=None):
    """
    Generate a test audio file for an advisory meeting.
    
    Args:
        output_filename (str, optional): Path to save the output audio file. Defaults to "test_advisory_meeting.wav".
        script (dict, optional): Script for the meeting. If None, a default script will be used.
        
    Returns:
        bool: True if successful, False otherwise
    """
    if output_filename is None:
        output_filename = "test_advisory_meeting.wav"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a temporary directory for audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Use default script if none provided
            if script is None:
                script = generate_meeting_script()
            
            # List to keep track of temporary files
            temp_files = []
            
            # List to keep track of speaker segments
            speaker_segments = []
            
            # Generate audio for each speaker's parts
            for i, maria_part in enumerate(script["maria_parts"]):
                file_name = os.path.join(temp_dir, f"maria_{i+1}.wav")
                success = synthesize_with_voice(maria_part, ADVISOR_VOICE, file_name)
                if not success:
                    raise Exception(f"Failed to synthesize Maria part {i+1}")
                temp_files.append(file_name)
                speaker_segments.append({"role": "advisor", "voice": ADVISOR_VOICE, "text": maria_part})
            
            for i, erik_part in enumerate(script["erik_parts"]):
                file_name = os.path.join(temp_dir, f"erik_{i+1}.wav")
                success = synthesize_with_voice(erik_part, CLIENT1_VOICE, file_name)
                if not success:
                    raise Exception(f"Failed to synthesize Erik part {i+1}")
                temp_files.append(file_name)
                speaker_segments.append({"role": "client", "voice": CLIENT1_VOICE, "text": erik_part})
            
            for i, lena_part in enumerate(script["lena_parts"]):
                file_name = os.path.join(temp_dir, f"lena_{i+1}.wav")
                success = synthesize_with_voice(lena_part, CLIENT2_VOICE, file_name)
                if not success:
                    raise Exception(f"Failed to synthesize Lena part {i+1}")
                temp_files.append(file_name)
                speaker_segments.append({"role": "client", "voice": CLIENT2_VOICE, "text": lena_part})
            
            # Combine all audio files into one
            print("Kombinerar ljudfiler...")
            with wave.open(output_filename, 'wb') as output:
                # Use parameters from the first file
                with wave.open(temp_files[0], 'rb') as w:
                    output.setparams(w.getparams())
                
                # Write audio data from each file
                for temp_file in temp_files:
                    with wave.open(temp_file, 'rb') as w:
                        output.writeframes(w.readframes(w.getnframes()))
            
            # Save speaker information to a JSON file for reference
            speaker_info_file = os.path.splitext(output_filename)[0] + "_speakers.json"
            with open(speaker_info_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "speakers": [
                        {"role": "advisor", "voice": ADVISOR_VOICE, "name": "Maria Johansson"},
                        {"role": "client", "voice": CLIENT1_VOICE, "name": "Erik Andersson"},
                        {"role": "client", "voice": CLIENT2_VOICE, "name": "Lena Karlsson"}
                    ],
                    "segments": speaker_segments
                }, f, ensure_ascii=False, indent=2)
            
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file {temp_file}: {str(e)}")
            
            print(f"Test audio file generated: {output_filename}")
            print(f"Speaker information saved to: {speaker_info_file}")
            return True
        
        except Exception as e:
            print(f"Error generating test audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("Genererar testljudfil för rådgivningsmöte med flera röster och dynamiskt innehåll...")
    success = generate_test_audio()
    if success:
        print("Generering av testljud slutfördes framgångsrikt!")
    else:
        print("Generering av testljud misslyckades.")
        sys.exit(1)
