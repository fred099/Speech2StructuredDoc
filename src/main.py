import os
import logging
import json
from pathlib import Path
from src.auth import get_credential, get_token
from src.storage import get_blob_service_client, upload_file, download_file
from src.speech import transcribe_audio
from src.llm import extract_structured_data
from src.config import CONTAINER_NAME, LOG_LEVEL
from src.speaker_identification import SpeakerAnalyzer, configure_diarization
from src.models import ProcessingResult, SpeakerInfo, SpeakerAnalysisResult

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_audio_file(audio_file_path, output_dir=None, enable_speaker_analysis=True):
    """
    Process an audio file to extract structured data.
    
    Args:
        audio_file_path: Path to the audio file to process
        output_dir: Directory to save output files (optional)
        enable_speaker_analysis: Whether to enable speaker identification (default: True)
        
    Returns:
        Structured data extracted from the audio file
    """
    try:
        logger.info(f"Processing audio file: {audio_file_path}")
        
        # Get API key from environment variables
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            logger.warning("AZURE_OPENAI_API_KEY not found in environment variables")
        
        # Step 1: Transcribe audio file
        logger.info("Step 1: Transcribing audio file")
        transcription_result = transcribe_audio(audio_file_path, api_key, enable_diarization=enable_speaker_analysis)
        
        # If speaker analysis is enabled, the result will be a tuple (transcription, speaker_utterances)
        # Otherwise, it will just be the transcription text
        if enable_speaker_analysis and isinstance(transcription_result, tuple):
            transcription, speaker_utterances = transcription_result
            
            # Initialize speaker analyzer
            logger.info("Analyzing speakers and their roles")
            speaker_analyzer = SpeakerAnalyzer()
            
            # Add utterances to the speaker analyzer
            for speaker_id, text in speaker_utterances:
                speaker_analyzer.add_utterance(speaker_id, text)
            
            # Analyze speaker roles
            from src.azure_openai_provider import AzureOpenAIProvider
            provider = AzureOpenAIProvider(use_token=False)
            
            def get_completion_func(request):
                return provider.get_completion(request)
            
            speaker_analyzer.analyze_with_llm(get_completion_func)
            speaker_analysis_results = speaker_analyzer.get_analysis_results()
        else:
            transcription = transcription_result
            speaker_analysis_results = None
        
        # Save transcription to file if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            transcription_file = os.path.join(output_dir, "transcription.txt")
            with open(transcription_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            logger.info(f"Transcription saved to: {transcription_file}")
        
        # Step 2: Extract structured data from transcription
        logger.info("Step 2: Extracting structured data from transcription")
        structured_data_dict = extract_structured_data(transcription, api_key, speaker_analysis_results)
        
        # Step 3: Create a complete processing result with speaker analysis
        if enable_speaker_analysis and speaker_analysis_results:
            # Create speaker info objects
            speaker_info_list = []
            for speaker_id, role in speaker_analysis_results.get("roles", {}).items():
                confidence = speaker_analysis_results.get("confidence", {}).get(speaker_id, 0.7)
                reasoning = speaker_analysis_results.get("reasoning", {}).get(speaker_id, "")
                utterance_count = speaker_analyzer.get_utterance_count(speaker_id)
                
                speaker_info = SpeakerInfo(
                    id=speaker_id,
                    name=f"Speaker {speaker_id}",
                    role=role,
                    confidence=confidence,
                    reasoning=reasoning,
                    utterance_count=utterance_count
                )
                speaker_info_list.append(speaker_info)
            
            # Create a proper SpeakerAnalysisResult object
            speaker_analysis_result = SpeakerAnalysisResult(
                speakers=speaker_info_list,
                transcription=[f"[Speaker {utterance[0]}]: {utterance[1]}" for utterance in speaker_utterances]
            )
            
            # Create a complete processing result
            processing_result = ProcessingResult(
                transcription=transcription,
                structured_data=structured_data_dict,
                speaker_analysis=speaker_analysis_result
            )
            
            # Convert to JSON
            processing_result_dict = processing_result.dict()
            
            # Save complete result to file if output_dir is provided
            if output_dir:
                complete_result_file = os.path.join(output_dir, "complete_result.json")
                with open(complete_result_file, "w", encoding="utf-8") as f:
                    # Use json.dump with ensure_ascii=False to preserve UTF-8 characters
                    json.dump(processing_result_dict, f, ensure_ascii=False, indent=2)
                logger.info(f"Complete result saved to: {complete_result_file}")
        else:
            processing_result_dict = structured_data_dict
        
        # Save structured data to file if output_dir is provided
        if output_dir:
            structured_data_file = os.path.join(output_dir, "structured_data.json")
            with open(structured_data_file, "w", encoding="utf-8") as f:
                # Use json.dump with ensure_ascii=False to preserve UTF-8 characters
                json.dump(structured_data_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Structured data saved to: {structured_data_file}")
        
        # Step 4: Upload results to Azure Blob Storage
        logger.info("Step 4: Uploading results to Azure Blob Storage")
        
        # Get blob service client
        blob_service_client = get_blob_service_client(get_credential())
        
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
            json.dumps(structured_data_dict, ensure_ascii=False), 
            structured_data_blob_name
        )
        logger.info(f"Structured data uploaded to blob: {structured_data_blob_name}")
        
        # Upload complete result if speaker analysis was enabled
        if enable_speaker_analysis and speaker_analysis_results:
            complete_result_blob_name = f"{base_filename}/complete_result.json"
            upload_file(
                blob_service_client, 
                CONTAINER_NAME, 
                json.dumps(processing_result_dict, ensure_ascii=False), 
                complete_result_blob_name
            )
            logger.info(f"Complete result uploaded to blob: {complete_result_blob_name}")
        
        # Return the structured data
        return structured_data_dict
        
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
    parser.add_argument("--disable-speaker-analysis", action="store_true", help="Disable speaker identification")
    args = parser.parse_args()
    
    try:
        result = process_audio_file(
            args.audio_file, 
            args.output_dir,
            enable_speaker_analysis=not args.disable_speaker_analysis
        )
        logger.info("Audio processing completed successfully")
        logger.info(f"Extracted data: {json.dumps(result, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to process audio file: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
