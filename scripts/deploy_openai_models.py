#!/usr/bin/env python
"""
Script to deploy OpenAI models to Azure AI Services resource.
This script creates deployments for the models required by the Speech2StructuredDoc application.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
import requests

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION

# Load environment variables
load_dotenv()

# API Key for authentication
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

def deploy_model(model_name, deployment_name):
    """Deploy an OpenAI model to Azure AI Services."""
    try:
        print(f"Deploying model {model_name} as {deployment_name}...")
        
        # Set up API call
        headers = {
            "api-key": API_KEY,
            "Content-Type": "application/json"
        }
        
        # Deployment configuration
        payload = {
            "model": model_name,
            "capacity": 1,
            "scale_settings": {
                "scale_type": "standard"
            }
        }
        
        # Make API call to create deployment
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{deployment_name}?api-version={AZURE_OPENAI_API_VERSION}"
        print(f"Making request to: {url}")
        
        response = requests.put(url, headers=headers, json=payload)
        
        # Check response
        if response.status_code in [200, 201, 202]:
            result = response.json()
            print(f"Successfully initiated deployment of {deployment_name}!")
            print("Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error deploying model: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Error deploying model: {str(e)}")
        return False

def check_deployment_status(deployment_name):
    """Check the status of a model deployment."""
    try:
        print(f"Checking status of deployment {deployment_name}...")
        
        # Set up API call
        headers = {
            "api-key": API_KEY,
            "Content-Type": "application/json"
        }
        
        # Make API call to get deployment status
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{deployment_name}?api-version={AZURE_OPENAI_API_VERSION}"
        
        response = requests.get(url, headers=headers)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            status = result.get("status", "unknown")
            print(f"Deployment {deployment_name} status: {status}")
            return status
        else:
            print(f"Error checking deployment status: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error checking deployment status: {str(e)}")
        return None

def list_deployments():
    """List all OpenAI deployments."""
    try:
        print("Listing all deployments...")
        
        # Set up API call
        headers = {
            "api-key": API_KEY,
            "Content-Type": "application/json"
        }
        
        # Make API call to list deployments
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments?api-version={AZURE_OPENAI_API_VERSION}"
        
        response = requests.get(url, headers=headers)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("Deployments:")
            print(json.dumps(result, indent=2))
            return result.get("data", [])
        else:
            print(f"Error listing deployments: {response.status_code}")
            print(response.text)
            return []
            
    except Exception as e:
        print(f"Error listing deployments: {str(e)}")
        return []

def main():
    """Main function to deploy required models."""
    # Models to deploy based on WebScraper-RAG pattern
    models_to_deploy = [
        {"model": "gpt-4o", "deployment": "gpt-4o"},  # AZURE_OPENAI_CAPABLE_MODEL
        {"model": "gpt-4o-mini", "deployment": "gpt-4o-mini"},  # AZURE_OPENAI_FAST_MODEL
        {"model": "text-embedding-ada-002", "deployment": "text-embedding-ada-002"}  # AZURE_OPENAI_EMBEDDING_MODEL
    ]
    
    # First, list existing deployments
    print("Checking existing deployments...")
    existing_deployments = list_deployments()
    existing_deployment_names = [d.get("id") for d in existing_deployments]
    
    # Deploy models that don't already exist
    for model in models_to_deploy:
        model_name = model["model"]
        deployment_name = model["deployment"]
        
        if deployment_name in existing_deployment_names:
            print(f"Deployment {deployment_name} already exists. Skipping...")
            continue
        
        success = deploy_model(model_name, deployment_name)
        if success:
            print(f"Waiting for deployment {deployment_name} to complete...")
            # Wait a bit before checking status
            time.sleep(10)
            
            # Check status a few times
            for _ in range(5):
                status = check_deployment_status(deployment_name)
                if status == "succeeded":
                    print(f"Deployment {deployment_name} completed successfully!")
                    break
                elif status in ["failed", "canceled"]:
                    print(f"Deployment {deployment_name} failed or was canceled.")
                    break
                else:
                    print(f"Deployment {deployment_name} is still in progress. Waiting...")
                    time.sleep(30)
    
    # Final list of deployments
    print("\nFinal list of deployments:")
    list_deployments()

if __name__ == "__main__":
    main()
