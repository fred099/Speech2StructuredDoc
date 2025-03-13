from openai import AzureOpenAI
import logging
import json
from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CAPABLE_MODEL

logger = logging.getLogger(__name__)

def extract_structured_data(transcription, token):
    """
    Extract structured data from transcription text using Azure OpenAI.
    
    Args:
        transcription: The transcribed text to extract data from
        token: Authentication token for Azure OpenAI
        
    Returns:
        Extracted structured data as a JSON string
    """
    try:
        # Initialize Azure OpenAI client with token-based authentication
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_ad_token=token
        )
        
        # Define system prompt for structured data extraction
        system_message = """
        Extract these fields from the transcription as JSON:
        - client_name: The name of the client or organization mentioned
        - meeting_date: The date of the meeting in YYYY-MM-DD format
        - key_points: A summary of the main points discussed
        - action_items: A list of action items or next steps mentioned
        - participants: Names of participants mentioned in the meeting
        
        Set missing fields to null. Format the output as valid JSON.
        """
        
        logger.info("Sending transcription to Azure OpenAI for structured data extraction")
        
        # Make the API call using the capable model for complex analysis
        response = client.chat.completions.create(
            model=AZURE_OPENAI_CAPABLE_MODEL,  # Using the capable model for complex extraction
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcription}
            ],
            temperature=0.3,  # Lower temperature for more deterministic output
            response_format={"type": "json_object"}
        )
        
        # Extract and validate the response
        result = response.choices[0].message.content
        
        # Validate that the result is valid JSON
        try:
            json.loads(result)
            logger.info("Successfully extracted structured data")
            return result
        except json.JSONDecodeError:
            logger.error("Azure OpenAI returned invalid JSON")
            raise ValueError("Failed to parse structured data as JSON")
            
    except Exception as e:
        logger.error(f"Error in structured data extraction: {str(e)}")
        raise
