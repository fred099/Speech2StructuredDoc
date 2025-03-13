#!/usr/bin/env python
"""
Test script for Azure OpenAI and Speech services using API keys.
This script tests both services to verify they are accessible.
"""

import os
import sys
import json
from dotenv import load_dotenv
import requests
import azure.cognitiveservices.speech as speechsdk

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

def test_openai_with_api_key():
    """Test connection to Azure OpenAI using API key authentication."""
    try:
        # Get configuration from environment variables
        openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if not api_key:
            print("No API key found in environment variables. Please set AZURE_OPENAI_API_KEY.")
            return False
        
        print(f"Using OpenAI endpoint: {openai_endpoint}")
        print(f"Using API version: {api_version}")
        print(f"Using deployment: {deployment_name}")
        
        # Set up API call
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        
        # Simple completion request
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you hear me?"}
            ],
            "max_tokens": 100
        }
        
        # Make API call
        url = f"{openai_endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
        print(f"Making request to: {url}")
        
        response = requests.post(url, headers=headers, json=payload)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("Successfully connected to Azure OpenAI!")
            print("Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_speech_with_api_key():
    """Test connection to Azure Speech service using API key."""
    try:
        # Get configuration from environment variables
        speech_region = os.getenv("SPEECH_REGION")
        speech_api_key = os.getenv("SPEECH_API_KEY")
        
        if not speech_api_key:
            print("No API key found in environment variables. Please set SPEECH_API_KEY.")
            return False
        
        print(f"Using Speech region: {speech_region}")
        
        # Create speech config with API key
        speech_config = speechsdk.SpeechConfig(
            subscription=speech_api_key,
            region=speech_region
        )
        
        # Test speech synthesis (text-to-speech)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        
        # Simple text to synthesize
        text = "Hello, this is a test of the Azure Speech service using API key."
        
        print("Attempting to synthesize speech...")
        result = synthesizer.speak_text_async(text).get()
        
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesis succeeded!")
            return True
        else:
            print(f"Speech synthesis failed: {result.reason}")
            if hasattr(result, 'error_details'):
                print(f"Detailed error: {result.error_details}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Azure services with API keys...")
    
    # Test OpenAI service
    print("\n=== Testing OpenAI Service ===")
    openai_success = test_openai_with_api_key()
    
    # Test Speech service
    print("\n=== Testing Speech Service ===")
    speech_success = test_speech_with_api_key()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"OpenAI Service: {'Success' if openai_success else 'Failed'}")
    print(f"Speech Service: {'Success' if speech_success else 'Failed'}")
    
    if openai_success and speech_success:
        print("\nAll service tests succeeded!")
    elif openai_success or speech_success:
        print("\nSome service tests succeeded.")
    else:
        print("\nAll service tests failed.")
        sys.exit(1)
