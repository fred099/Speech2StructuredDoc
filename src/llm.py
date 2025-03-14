import logging
import json
from typing import Optional, Dict, List, Any
from pydantic import BaseModel
from src.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT
from src.auth import get_credential
from src.openai_client import create_azure_openai_client, ChatMessage, get_completion

logger = logging.getLogger(__name__)

def extract_structured_data(transcription, api_key, speaker_analysis_results=None):
    """
    Extract structured data from a transcription using Azure OpenAI.
    
    Args:
        transcription (str): The transcription text to extract data from
        api_key (str): The API key for Azure OpenAI
        speaker_analysis_results (dict, optional): Results from speaker analysis
    
    Returns:
        dict: Structured data extracted from the transcription
    """
    try:
        logger.info("Creating Azure OpenAI client")
        
        # Create Azure OpenAI client
        client = create_azure_openai_client(
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            api_key=api_key
        )
        
        # Create system message
        system_message = ChatMessage(
            role="system",
            content="""
            You are an AI assistant that extracts structured data from meeting transcriptions.
            Extract the following information from the transcription:
            - client_name: Name of the client(s)
            - meeting_type: Type of meeting (e.g., consultation, review, planning)
            - key_topics: List of key topics discussed
            - action_items: List of action items or next steps
            - financial_figures: List of financial figures mentioned
            - risk_factors: List of risk factors mentioned
            - advisor_name: Name of the financial advisor
            - client_representatives: List of client representatives present
            
            Return the data as a valid JSON object.
            """
        )
        
        # Prepare the prompt with transcription and speaker analysis if available
        prompt = f"Here is the transcription:\n\n{transcription}\n\n"
        
        if speaker_analysis_results:
            prompt += "Speaker analysis results:\n"
            for speaker, role in speaker_analysis_results.get("roles", {}).items():
                prompt += f"Speaker {speaker} is {role}\n"
                
                # Add confidence if available
                confidence = speaker_analysis_results.get("confidence", {}).get(speaker)
                if confidence:
                    prompt += f"  Confidence: {confidence}\n"
                
                # Add reasoning if available
                reasoning = speaker_analysis_results.get("reasoning", {}).get(speaker)
                if reasoning:
                    prompt += f"  Reasoning: {reasoning}\n"
        
        # Create user message with transcription
        user_message = ChatMessage(role="user", content=prompt)
        
        # Create messages list
        messages = [system_message, user_message]
        
        logger.info("Sending structured data extraction request to Azure OpenAI")
        
        # Call Azure OpenAI API
        result_text = get_completion(
            client=client,
            messages=[message.dict() for message in messages],  # Convert Pydantic models to dictionaries
            temperature=0.3,
            max_tokens=1000
        )
        
        # Try to parse the JSON
        try:
            # Find JSON content if it's wrapped in markdown code blocks
            if "```json" in result_text:
                json_content = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                json_content = result_text.split("```")[1].split("```")[0].strip()
            else:
                json_content = result_text.strip()
                
            # Parse the JSON
            structured_data = json.loads(json_content)
            return structured_data
            
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response")
            logger.debug(f"Raw response: {result_text}")
            raise
            
    except Exception as e:
        logger.error(f"Error extracting structured data: {str(e)}")
        raise
