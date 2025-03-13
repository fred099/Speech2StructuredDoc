import os
import logging
import json
from pathlib import Path
from src.auth import get_credential, get_token
from src.storage import get_blob_service_client, upload_file, download_file
from src.speech import transcribe_audio
from src.llm import extract_structured_data
from src.config import CONTAINER_NAME, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_audio_file(audio_file_path, output_dir=None):
    """
    Process an audio file to extract structured data.
    
    Args:
        audio_file_path: Path to the audio file to process
        output_dir: Directory to save output files (optional)
        
    Returns:
        Structured data extracted from the audio file
    """
    try:
        logger.info(f"Processing audio file: {audio_file_path}")
        
        # Get Azure credential with fallback mechanism
        credential = get_credential()
        
        # Get token for Azure services
        token = get_token(credential)
        
        # Step 1: Transcribe audio file
        logger.info("Step 1: Transcribing audio file")
        transcription = transcribe_audio(audio_file_path, token)
        
        # Save transcription to file if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            transcription_file = os.path.join(output_dir, "transcription.txt")
            with open(transcription_file, "w") as f:
                f.write(transcription)
            logger.info(f"Transcription saved to: {transcription_file}")
        
        # Step 2: Extract structured data from transcription
        logger.info("Step 2: Extracting structured data from transcription")
        structured_data = extract_structured_data(transcription, token)
        
        # Save structured data to file if output_dir is provided
        if output_dir:
            structured_data_file = os.path.join(output_dir, "structured_data.json")
            with open(structured_data_file, "w") as f:
                f.write(structured_data)
            logger.info(f"Structured data saved to: {structured_data_file}")
        
        # Step 3: Upload results to Azure Blob Storage
        logger.info("Step 3: Uploading results to Azure Blob Storage")
        
        # Get blob service client
        blob_service_client = get_blob_service_client(credential)
        
        # Generate base filename from the audio file
        base_filename = Path(audio_file_path).stem
        
        # Upload transcription
        transcription_blob_name = f"{base_filename}/transcription.txt"
        upload_file(
            blob_service_client, 
            CONTAINER_NAME, 
            transcription, 
            transcription_blob_name
        )
        logger.info(f"Transcription uploaded to blob: {transcription_blob_name}")
        
        # Upload structured data
        structured_data_blob_name = f"{base_filename}/structured_data.json"
        upload_file(
            blob_service_client, 
            CONTAINER_NAME, 
            structured_data, 
            structured_data_blob_name
        )
        logger.info(f"Structured data uploaded to blob: {structured_data_blob_name}")
        
        # Return the structured data
        return json.loads(structured_data)
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise

def main():
    """
    Main function to process audio files.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio files to extract structured data")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--output-dir", help="Directory to save output files")
    args = parser.parse_args()
    
    try:
        result = process_audio_file(args.audio_file, args.output_dir)
        logger.info("Audio processing completed successfully")
        logger.info(f"Extracted data: {json.dumps(result, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to process audio file: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
