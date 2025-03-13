import os
import sys
from pathlib import Path
import logging

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from src.config import STORAGE_ACCOUNT_NAME, CONTAINER_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_container():
    """
    Create a container in the Azure Storage account using DefaultAzureCredential.
    """
    try:
        # Get Azure credentials using DefaultAzureCredential
        logger.info("Getting Azure credentials...")
        credential = DefaultAzureCredential()
        
        # Create the blob service client
        account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
        logger.info(f"Connecting to storage account: {STORAGE_ACCOUNT_NAME}")
        blob_service_client = BlobServiceClient(account_url, credential=credential)
        
        # Create the container if it doesn't exist
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        if not container_client.exists():
            logger.info(f"Creating container: {CONTAINER_NAME}")
            container_client.create_container()
            logger.info(f"Container {CONTAINER_NAME} created successfully")
        else:
            logger.info(f"Container {CONTAINER_NAME} already exists")
            
    except Exception as e:
        logger.error(f"Error creating container: {str(e)}")
        raise

if __name__ == "__main__":
    create_container()
