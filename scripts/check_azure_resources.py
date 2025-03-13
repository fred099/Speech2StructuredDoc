import subprocess
import json
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_az_command(command):
    """Run an Azure CLI command and return the result as JSON."""
    try:
        logger.info(f"Running command: az {' '.join(command)}")
        result = subprocess.run(["az"] + command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Command failed with error: {result.stderr}")
            return None
        
        return json.loads(result.stdout)
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return None

def check_speech_service():
    """Check Azure Speech Service resources."""
    logger.info("Checking Azure Speech Service resources...")
    
    # List Speech Services
    speech_services = run_az_command([
        "cognitiveservices", "account", "list", 
        "--query", "[?kind=='SpeechServices']"
    ])
    
    if not speech_services:
        logger.warning("No Speech Services found")
        return
    
    logger.info(f"Found {len(speech_services)} Speech Services:")
    for service in speech_services:
        logger.info(f"- Name: {service.get('name')}")
        logger.info(f"  Resource Group: {service.get('resourceGroup')}")
        logger.info(f"  Location: {service.get('location')}")
        logger.info(f"  Endpoint: {service.get('properties', {}).get('endpoint')}")
        logger.info("---")

def check_openai_service():
    """Check Azure OpenAI Service resources and deployments."""
    logger.info("Checking Azure OpenAI Service resources...")
    
    # List OpenAI Services
    openai_services = run_az_command([
        "cognitiveservices", "account", "list", 
        "--query", "[?kind=='OpenAI']"
    ])
    
    if not openai_services:
        logger.warning("No OpenAI Services found")
        return
    
    logger.info(f"Found {len(openai_services)} OpenAI Services:")
    for service in openai_services:
        service_name = service.get('name')
        resource_group = service.get('resourceGroup')
        
        logger.info(f"- Name: {service_name}")
        logger.info(f"  Resource Group: {resource_group}")
        logger.info(f"  Location: {service.get('location')}")
        logger.info(f"  Endpoint: {service.get('properties', {}).get('endpoint')}")
        
        # List deployments for this service
        deployments = run_az_command([
            "cognitiveservices", "account", "deployment", "list",
            "--name", service_name,
            "--resource-group", resource_group
        ])
        
        if deployments:
            logger.info(f"  Deployments ({len(deployments)}):")
            for deployment in deployments:
                model_name = deployment.get('properties', {}).get('model', {}).get('name', 'Unknown')
                logger.info(f"    - {deployment.get('name')} (Model: {model_name})")
        else:
            logger.info("  No deployments found")
        
        logger.info("---")

def check_storage_accounts():
    """Check Azure Storage accounts."""
    logger.info("Checking Azure Storage accounts...")
    
    # List Storage accounts
    storage_accounts = run_az_command([
        "storage", "account", "list"
    ])
    
    if not storage_accounts:
        logger.warning("No Storage accounts found")
        return
    
    logger.info(f"Found {len(storage_accounts)} Storage accounts:")
    for account in storage_accounts:
        logger.info(f"- Name: {account.get('name')}")
        logger.info(f"  Resource Group: {account.get('resourceGroup')}")
        logger.info(f"  Location: {account.get('location')}")
        logger.info("---")

def main():
    """Main function to check all Azure resources."""
    logger.info("Starting Azure resource check...")
    
    # Check Speech Service
    check_speech_service()
    
    # Check OpenAI Service
    check_openai_service()
    
    # Check Storage accounts
    check_storage_accounts()
    
    logger.info("Azure resource check completed")

if __name__ == "__main__":
    main()
