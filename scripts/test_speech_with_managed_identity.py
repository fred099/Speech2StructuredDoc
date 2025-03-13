#!/usr/bin/env python
"""
Test script for Azure Speech service using DefaultAzureCredential authentication.
Following the WebScraper-RAG authentication pattern.
"""

import os
import sys
import json
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, AzureCliCredential
import azure.cognitiveservices.speech as speechsdk

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

def get_credential():
    """
    Get Azure credential using DefaultAzureCredential with fallback to AzureCliCredential.
    Following the WebScraper-RAG authentication pattern.
    """
    try:
        # Try DefaultAzureCredential first (primary method)
        credential = DefaultAzureCredential(additionally_allowed_tenants=["*"])
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        print("Successfully authenticated with DefaultAzureCredential")
        return credential
    except Exception as e:
        print(f"DefaultAzureCredential failed: {str(e)}")
        try:
            # Fall back to AzureCliCredential
            credential = AzureCliCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            print("Successfully authenticated with AzureCliCredential")
            return credential
        except Exception as cli_error:
            print(f"AzureCliCredential also failed: {str(cli_error)}")
            raise Exception("All authentication methods failed")

def test_azure_speech():
    """Test connection to Azure Speech service using managed identity."""
    try:
        # Get credential
        credential = get_credential()
        
        # Get access token for Azure Speech
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        
        # Set up Speech configuration
        speech_endpoint = os.getenv("SPEECH_ENDPOINT")
        speech_region = os.getenv("SPEECH_REGION", "swedencentral")
        
        print(f"Using Speech endpoint: {speech_endpoint}")
        print(f"Using Speech region: {speech_region}")
        
        # Create speech config with token
        speech_config = speechsdk.SpeechConfig(
            auth_token=token.token,
            region=speech_region
        )
        
        # Test speech synthesis (text-to-speech)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        
        # Simple text to synthesize
        text = "Hello, this is a test of the Azure Speech service using managed identity."
        
        print("Attempting to synthesize speech...")
        result = synthesizer.speak_text_async(text).get()
        
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesis succeeded!")
            return True
        else:
            print(f"Speech synthesis failed: {result.reason}")
            print(f"Detailed error: {result.error_details}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Azure Speech with DefaultAzureCredential...")
    success = test_azure_speech()
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed.")
        sys.exit(1)
