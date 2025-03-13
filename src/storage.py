import logging
from azure.storage.blob import BlobServiceClient
from src.config import STORAGE_ACCOUNT_NAME

logger = logging.getLogger(__name__)

def get_blob_service_client(credential):
    """
    Get a BlobServiceClient for Azure Blob Storage using DefaultAzureCredential.
    
    Args:
        credential: Azure credential from DefaultAzureCredential or AzureCliCredential
        
    Returns:
        BlobServiceClient for interacting with Azure Blob Storage
    """
    try:
        # Construct the account URL
        account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
        
        logger.info(f"Creating BlobServiceClient for account: {STORAGE_ACCOUNT_NAME}")
        
        # Create the blob service client using token-based authentication
        blob_service_client = BlobServiceClient(account_url, credential=credential)
        
        return blob_service_client
        
    except Exception as e:
        logger.error(f"Error creating BlobServiceClient: {str(e)}")
        raise

def upload_file(blob_service_client, container_name, content, blob_name):
    """
    Upload content to a blob in Azure Blob Storage.
    
    Args:
        blob_service_client: BlobServiceClient for Azure Blob Storage
        container_name: Name of the container to upload to
        content: Content to upload (string or bytes)
        blob_name: Name of the blob to create
        
    Returns:
        URL of the uploaded blob
    """
    try:
        # Get a blob client
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        logger.info(f"Uploading to blob: {blob_name}")
        
        # Convert content to bytes if it's a string
        if isinstance(content, str):
            content = content.encode('utf-8')
            
        # Upload the content
        blob_client.upload_blob(content, overwrite=True)
        
        logger.info(f"Upload completed successfully")
        
        # Return the URL of the uploaded blob
        return blob_client.url
        
    except Exception as e:
        logger.error(f"Error uploading to blob {blob_name}: {str(e)}")
        raise

def download_file(blob_service_client, container_name, blob_name):
    """
    Download content from a blob in Azure Blob Storage.
    
    Args:
        blob_service_client: BlobServiceClient for Azure Blob Storage
        container_name: Name of the container to download from
        blob_name: Name of the blob to download
        
    Returns:
        Content of the blob as a string
    """
    try:
        # Get a blob client
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        logger.info(f"Downloading from blob: {blob_name}")
        
        # Download the blob
        download_stream = blob_client.download_blob()
        
        # Read the content
        content = download_stream.readall()
        
        # Convert to string if it's bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
            
        logger.info(f"Download completed successfully")
        
        return content
        
    except Exception as e:
        logger.error(f"Error downloading from blob {blob_name}: {str(e)}")
        raise
