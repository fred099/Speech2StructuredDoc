import os
import sys
from pathlib import Path
import logging
import subprocess
import json

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import AZURE_OPENAI_ENDPOINT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_resource_name(endpoint_url):
    """Extract the resource name from the endpoint URL."""
    if not endpoint_url:
        return None
    
    # Format: https://resourcename.openai.azure.com/
    try:
        parts = endpoint_url.replace('https://', '').split('.')
        return parts[0]
    except:
        return None

def list_openai_deployments():
    """
    List Azure OpenAI deployments using Azure CLI.
    """
    try:
        # Extract resource name from endpoint
        resource_name = extract_resource_name(AZURE_OPENAI_ENDPOINT)
        if not resource_name:
            logger.error(f"Could not extract resource name from endpoint: {AZURE_OPENAI_ENDPOINT}")
            return
        
        logger.info(f"Listing deployments for Azure OpenAI resource: {resource_name}")
        
        # Use Azure CLI to list deployments
        cmd = ["az", "cognitiveservices", "account", "deployment", "list", 
               "--name", resource_name, 
               "--resource-group", "GraphRag"]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error listing deployments: {result.stderr}")
            return
        
        # Parse and display deployments
        deployments = json.loads(result.stdout)
        
        if not deployments:
            logger.info("No deployments found")
            return
        
        logger.info(f"Found {len(deployments)} deployments:")
        for deployment in deployments:
            logger.info(f"- Name: {deployment.get('name')}")
            logger.info(f"  Model: {deployment.get('properties', {}).get('model', {}).get('name')}")
            logger.info(f"  Status: {deployment.get('properties', {}).get('provisioningState')}")
            logger.info("---")
            
    except Exception as e:
        logger.error(f"Error listing OpenAI deployments: {str(e)}")
        raise

if __name__ == "__main__":
    list_openai_deployments()
