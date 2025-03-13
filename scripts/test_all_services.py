#!/usr/bin/env python
"""
Comprehensive test script for both Azure OpenAI and Speech services.
Uses hybrid authentication with fallback to API keys when needed.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, AzureCliCredential
import requests
import azure.cognitiveservices.speech as speechsdk

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION", "swedencentral")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")

class AzureOpenAIAuthenticator:
    """Class to handle Azure OpenAI authentication with proper fallback mechanisms."""
    
    def __init__(self):
        self.credential = None
        self.api_key = AZURE_OPENAI_API_KEY
        self.token = None
        self.auth_method = None
    
    def authenticate(self):
        """Try different authentication methods in order of preference."""
        auth_methods = [
            self._try_default_azure_credential,
            self._try_azure_cli_credential,
            self._try_api_key
        ]
        
        for method in auth_methods:
            success = method()
            if success:
                return True
        
        print("All authentication methods failed.")
        return False
    
    def _try_default_azure_credential(self):
        """Try authenticating with DefaultAzureCredential."""
        try:
            print("Trying DefaultAzureCredential...")
            credential = DefaultAzureCredential(additionally_allowed_tenants=["*"])
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            self.credential = credential
            self.token = token
            self.auth_method = "DefaultAzureCredential"
            print("Successfully authenticated with DefaultAzureCredential")
            return True
        except Exception as e:
            print(f"DefaultAzureCredential failed: {str(e)}")
            return False
    
    def _try_azure_cli_credential(self):
        """Try authenticating with AzureCliCredential."""
        try:
            print("Trying AzureCliCredential...")
            credential = AzureCliCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            self.credential = credential
            self.token = token
            self.auth_method = "AzureCliCredential"
            print("Successfully authenticated with AzureCliCredential")
            return True
        except Exception as e:
            print(f"AzureCliCredential failed: {str(e)}")
            return False
    
    def _try_api_key(self):
        """Try authenticating with API key."""
        if self.api_key:
            print("Using API key authentication...")
            self.auth_method = "ApiKey"
            return True
        else:
            print("No API key available.")
            return False
    
    def get_headers(self):
        """Get the appropriate headers based on the authentication method."""
        headers = {"Content-Type": "application/json"}
        
        if self.auth_method in ["DefaultAzureCredential", "AzureCliCredential"]:
            # Refresh token if needed
            if self.token.expires_on < time.time() + 300:  # Refresh if expires in less than 5 minutes
                self.token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
            
            headers["Authorization"] = f"Bearer {self.token.token}"
        elif self.auth_method == "ApiKey":
            headers["api-key"] = self.api_key
        
        return headers

def test_azure_openai():
    """Test connection to Azure OpenAI using the hybrid authentication approach."""
    try:
        print(f"\n=== Testing OpenAI Service ===")
        print(f"Using OpenAI endpoint: {AZURE_OPENAI_ENDPOINT}")
        print(f"Using API version: {AZURE_OPENAI_API_VERSION}")
        print(f"Using deployment: {AZURE_OPENAI_DEPLOYMENT}")
        
        # Try token-based authentication first
        authenticator = AzureOpenAIAuthenticator()
        if not authenticator.authenticate():
            print("Failed to authenticate with any method.")
            return False
        
        # Set up API call
        headers = authenticator.get_headers()
        
        # Simple completion request
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you hear me?"}
            ],
            "max_tokens": 100
        }
        
        # Make API call
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        print(f"Making request to: {url}")
        print(f"Using authentication method: {authenticator.auth_method}")
        
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
            
            # If token-based auth failed, try API key as fallback
            if authenticator.auth_method in ["DefaultAzureCredential", "AzureCliCredential"] and "PermissionDenied" in response.text:
                print("\nToken-based authentication failed with permission denied. Trying API key as fallback...")
                authenticator = AzureOpenAIAuthenticator()
                # Skip to API key directly
                authenticator._try_api_key()
                headers = authenticator.get_headers()
                
                # Try again with API key
                print(f"Retrying with authentication method: {authenticator.auth_method}")
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print("Successfully connected to Azure OpenAI with API key fallback!")
                    print("Response:")
                    print(json.dumps(result, indent=2))
                    return True
                else:
                    print(f"API key fallback also failed: {response.status_code}")
                    print(response.text)
                    return False
            
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_azure_speech():
    """Test connection to Azure Speech using API key authentication."""
    try:
        print(f"\n=== Testing Speech Service ===")
        print(f"Using Speech region: {SPEECH_REGION}")
        
        # Create speech configuration with API key
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        
        # Use file output for testing
        output_filename = "test_output.wav"
        file_config = speechsdk.audio.AudioOutputConfig(filename=output_filename)
        
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
        
        print("Attempting to synthesize speech...")
        result = synthesizer.speak_text_async("This is a test of the Azure Speech service.").get()
        
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesis succeeded!")
            print(f"Audio saved to {output_filename}")
            return True
        else:
            print(f"Speech synthesis failed: {result.reason}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_integrated_workflow():
    """Test an integrated workflow using both OpenAI and Speech services."""
    try:
        print(f"\n=== Testing Integrated Workflow ===")
        
        # 1. Get a response from OpenAI
        print("Step 1: Getting response from OpenAI...")
        
        # Use API key authentication for OpenAI (most reliable)
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }
        
        # Create a prompt that would generate a structured response
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides structured data."},
                {"role": "user", "content": "Generate a short paragraph about artificial intelligence."}
            ],
            "max_tokens": 200
        }
        
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"OpenAI request failed: {response.status_code}")
            print(response.text)
            return False
        
        result = response.json()
        text_response = result["choices"][0]["message"]["content"]
        print(f"OpenAI Response: {text_response}")
        
        # 2. Convert the response to speech
        print("\nStep 2: Converting response to speech...")
        
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        
        output_filename = "integrated_test_output.wav"
        file_config = speechsdk.audio.AudioOutputConfig(filename=output_filename)
        
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
        
        result = synthesizer.speak_text_async(text_response).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Successfully converted OpenAI response to speech!")
            print(f"Audio saved to {output_filename}")
            return True
        else:
            print(f"Speech synthesis failed: {result.reason}")
            return False
        
    except Exception as e:
        print(f"Error in integrated workflow: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Azure services...")
    
    # Test OpenAI service
    openai_success = test_azure_openai()
    
    # Test Speech service
    speech_success = test_azure_speech()
    
    # Test integrated workflow
    integrated_success = test_integrated_workflow()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"OpenAI Service: {'Success' if openai_success else 'Failed'}")
    print(f"Speech Service: {'Success' if speech_success else 'Failed'}")
    print(f"Integrated Workflow: {'Success' if integrated_success else 'Failed'}")
    
    if openai_success and speech_success and integrated_success:
        print("\nAll service tests succeeded!")
        sys.exit(0)
    else:
        print("\nSome tests failed. See details above.")
        sys.exit(1)
