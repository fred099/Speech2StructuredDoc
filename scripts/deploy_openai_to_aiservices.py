#!/usr/bin/env python
"""
Script to deploy OpenAI models to Azure AI Services resource using Azure CLI.
This script creates deployments for the models required by the Speech2StructuredDoc application.
"""

import os
import sys
import json
import subprocess
import time
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# AI Services resource details
RESOURCE_GROUP = "GraphRag"
ACCOUNT_NAME = "ai-fredrikwingren-2029"

def run_command(command):
    """Run an Azure CLI command and return the output."""
    try:
        print(f"Running command: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return result.stdout
        return None
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return None

def deploy_model(model_name, deployment_name, capacity=1):
    """Deploy an OpenAI model to Azure AI Services."""
    print(f"Deploying model {model_name} as {deployment_name}...")
    
    # Create the deployment
    command = (
        f"az cognitiveservices account deployment create "
        f"--resource-group {RESOURCE_GROUP} "
        f"--name {ACCOUNT_NAME} "
        f"--deployment-name {deployment_name} "
        f"--model-name {model_name} "
        f"--model-version latest "
        f"--model-format OpenAI "
        f"--sku-capacity {capacity} "
        f"--sku-name Standard"
    )
    
    result = run_command(command)
    
    if result:
        print(f"Successfully initiated deployment of {deployment_name}!")
        return True
    else:
        print(f"Failed to deploy {deployment_name}")
        return False

def check_deployment_status(deployment_name):
    """Check the status of a model deployment."""
    print(f"Checking status of deployment {deployment_name}...")
    
    command = (
        f"az cognitiveservices account deployment show "
        f"--resource-group {RESOURCE_GROUP} "
        f"--name {ACCOUNT_NAME} "
        f"--deployment-name {deployment_name}"
    )
    
    result = run_command(command)
    
    if result:
        status = result.get("properties", {}).get("provisioningState", "Unknown")
        print(f"Deployment {deployment_name} status: {status}")
        return status
    else:
        print(f"Failed to get status for {deployment_name}")
        return None

def list_deployments():
    """List all OpenAI deployments."""
    print("Listing all deployments...")
    
    command = (
        f"az cognitiveservices account deployment list "
        f"--resource-group {RESOURCE_GROUP} "
        f"--name {ACCOUNT_NAME}"
    )
    
    result = run_command(command)
    
    if result:
        print("Deployments:")
        print(json.dumps(result, indent=2))
        return result
    else:
        print("Failed to list deployments")
        return []

def main():
    """Main function to deploy required models."""
    # Models to deploy based on WebScraper-RAG pattern
    models_to_deploy = [
        {"model": "gpt-4o", "deployment": "gpt-4o", "capacity": 1},  # AZURE_OPENAI_CAPABLE_MODEL
        {"model": "gpt-4o-mini", "deployment": "gpt-4o-mini", "capacity": 1},  # AZURE_OPENAI_FAST_MODEL
        {"model": "text-embedding-ada-002", "deployment": "text-embedding-ada-002", "capacity": 1}  # AZURE_OPENAI_EMBEDDING_MODEL
    ]
    
    # First, list existing deployments
    print("Checking existing deployments...")
    existing_deployments = list_deployments()
    existing_deployment_names = [d.get("name") for d in existing_deployments] if existing_deployments else []
    
    # Deploy models that don't already exist
    for model in models_to_deploy:
        model_name = model["model"]
        deployment_name = model["deployment"]
        capacity = model["capacity"]
        
        if deployment_name in existing_deployment_names:
            print(f"Deployment {deployment_name} already exists. Skipping...")
            continue
        
        success = deploy_model(model_name, deployment_name, capacity)
        if success:
            print(f"Waiting for deployment {deployment_name} to complete...")
            # Wait a bit before checking status
            time.sleep(10)
            
            # Check status a few times
            for _ in range(5):
                status = check_deployment_status(deployment_name)
                if status and status.lower() == "succeeded":
                    print(f"Deployment {deployment_name} completed successfully!")
                    break
                elif status and status.lower() in ["failed", "canceled"]:
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
