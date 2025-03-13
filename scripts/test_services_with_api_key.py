#!/usr/bin/env python
"""
Test script for Azure AI Services (OpenAI and Speech) using API key authentication.
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

# API Key for testing
API_KEY = "9H25P6p1ygTk98MMdZc3gzgCW1r2meZ4GqZ9ZV9jHN1himdoOmRSJQQJ99BCACfhMk5XJ3w3AAAAACOGpLKd"

def test_azure_openai():
    """Test connection to Azure OpenAI using API key authentication."""
    try:
        # Get endpoint information
        openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        
        print(f"Using OpenAI endpoint: {openai_endpoint}")
        print(f"Using API version: {api_version}")
        print(f"Using deployment: {deployment_name}")
        
        # Set up API call
        headers = {
            "api-key": API_KEY,
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

def test_azure_speech():
    """Test connection to Azure Speech service using API key."""
    try:
        # Get endpoint information
        speech_endpoint = os.getenv("SPEECH_ENDPOINT")
        speech_region = os.getenv("SPEECH_REGION", "swedencentral")
        
        print(f"Using Speech endpoint: {speech_endpoint}")
        print(f"Using Speech region: {speech_region}")
        
        # Create speech config with API key
        speech_config = speechsdk.SpeechConfig(
            subscription=API_KEY,
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

def test_openai_deployments():
    """List available OpenAI deployments using API key."""
    try:
        # Get endpoint information
        openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        print(f"Using OpenAI endpoint: {openai_endpoint}")
        print(f"Using API version: {api_version}")
        
        # Set up API call
        headers = {
            "api-key": API_KEY,
            "Content-Type": "application/json"
        }
        
        # Make API call to list deployments
        url = f"{openai_endpoint}openai/deployments?api-version={api_version}"
        print(f"Making request to list deployments: {url}")
        
        response = requests.get(url, headers=headers)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("Successfully retrieved OpenAI deployments!")
            print("Deployments:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error listing deployments: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Error listing deployments: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Azure AI Services with API key...")
    
    # First, check available OpenAI deployments
    print("\n=== Testing OpenAI Deployments ===")
    deployments_success = test_openai_deployments()
    
    # Test OpenAI service
    print("\n=== Testing OpenAI Service ===")
    openai_success = test_azure_openai()
    
    # Test Speech service
    print("\n=== Testing Speech Service ===")
    speech_success = test_azure_speech()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"OpenAI Deployments: {'Success' if deployments_success else 'Failed'}")
    print(f"OpenAI Service: {'Success' if openai_success else 'Failed'}")
    print(f"Speech Service: {'Success' if speech_success else 'Failed'}")
    
    if deployments_success or openai_success or speech_success:
        print("\nAt least one service test succeeded!")
    else:
        print("\nAll service tests failed.")
        sys.exit(1)
