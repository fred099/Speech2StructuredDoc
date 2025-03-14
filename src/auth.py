import logging
from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential, get_bearer_token_provider
from azure.core.exceptions import ClientAuthenticationError
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

def get_credential(additionally_allowed_tenants=None):
    """
    Get Azure credential using DefaultAzureCredential with fallback to AzureCliCredential.
    
    This follows the Azure AI foundry pattern for robust authentication:
    1. Try DefaultAzureCredential first (works with service principal in dev, managed identity in prod)
    2. Fall back to AzureCliCredential if DefaultAzureCredential fails
    
    Args:
        additionally_allowed_tenants: List of additional tenant IDs to allow for cross-tenant access
        
    Returns:
        An Azure credential that can be used to authenticate with Azure services
    """
    try:
        # Configure DefaultAzureCredential with additional allowed tenants if provided
        logger.info("Attempting to get DefaultAzureCredential...")
        
        credential_kwargs = {}
        if additionally_allowed_tenants:
            credential_kwargs["additionally_allowed_tenants"] = additionally_allowed_tenants
            
        # For local development, DefaultAzureCredential will automatically use
        # AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET from .env
        # For production, it will use Managed Identity
        credential = DefaultAzureCredential(**credential_kwargs)
        
        # Test the credential by getting a token
        # This will raise an exception if the credential is invalid
        credential.get_token("https://management.azure.com/.default")
        
        logger.info("Successfully obtained DefaultAzureCredential")
        return credential
        
    except ClientAuthenticationError as e:
        # Check if this is a tenant mismatch error (AADSTS700016)
        if "AADSTS700016" in str(e):
            logger.warning(f"Tenant mismatch error with DefaultAzureCredential: {str(e)}")
            logger.info("Trying AzureCliCredential as fallback...")
        else:
            logger.warning(f"Authentication error with DefaultAzureCredential: {str(e)}")
            logger.info("Falling back to AzureCliCredential...")
            
        try:
            # Try AzureCliCredential as fallback
            cli_credential = AzureCliCredential()
            # Test the credential
            cli_credential.get_token("https://management.azure.com/.default")
            logger.info("Successfully obtained AzureCliCredential")
            return cli_credential
        except Exception as cli_error:
            logger.error(f"Failed to get AzureCliCredential: {str(cli_error)}")
            # Re-raise the original error if CLI credential also fails
            raise e
    
    except Exception as e:
        logger.error(f"Unexpected error getting credential: {str(e)}")
        raise

def get_chained_credential(additionally_allowed_tenants=None):
    """
    Get a chained credential that tries DefaultAzureCredential first, then AzureCliCredential.
    
    Args:
        additionally_allowed_tenants: List of additional tenant IDs to allow for cross-tenant access
        
    Returns:
        A ChainedTokenCredential that will try multiple credential sources
    """
    logger.info("Creating chained credential...")
    
    credential_kwargs = {}
    if additionally_allowed_tenants:
        credential_kwargs["additionally_allowed_tenants"] = additionally_allowed_tenants
        
    default_credential = DefaultAzureCredential(**credential_kwargs)
    cli_credential = AzureCliCredential()
    
    return ChainedTokenCredential(default_credential, cli_credential)

def get_token(credential, scope="https://cognitiveservices.azure.com/.default"):
    """
    Get an authentication token for Azure services.
    
    Args:
        credential: The Azure credential to use
        scope: The scope to request a token for
        
    Returns:
        The authentication token as a string
    """
    return credential.get_token(scope).token

def create_azure_openai_client(endpoint, api_version, credential=None, api_key=None):
    """
    Create an Azure OpenAI client with proper authentication.
    
    This function ensures that only the required parameters are passed to the AzureOpenAI client,
    avoiding issues with proxies or other unexpected parameters.
    
    Args:
        endpoint: The Azure OpenAI endpoint
        api_version: The API version to use
        credential: The Azure credential to use for token-based authentication (optional)
        api_key: The API key to use for key-based authentication (optional)
        
    Returns:
        An initialized AzureOpenAI client
    """
    logger = logging.getLogger(__name__)
    
    # Determine which authentication method to use
    if credential is not None:
        logger.info("Creating Azure OpenAI client with token-based authentication")
        
        # Get token directly instead of using token provider
        token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        # Create client with only the required parameters
        client_kwargs = {
            "azure_endpoint": endpoint,
            "api_version": api_version,
            "azure_ad_token": token  # Use token directly instead of token provider
        }
    elif api_key is not None:
        logger.info("Creating Azure OpenAI client with API key authentication")
        
        # Create client with only the required parameters
        client_kwargs = {
            "api_key": api_key,
            "azure_endpoint": endpoint,
            "api_version": api_version
        }
    else:
        raise ValueError("Either credential or api_key must be provided")
    
    # Create the client with only the required parameters
    return AzureOpenAI(**client_kwargs)
