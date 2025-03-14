"""
Test script for speaker identification functionality.
This script tests the integration of speaker identification with the audio processing pipeline.
"""

import os
import sys
import logging
import json
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auth import get_credential, get_token
from src.speech import transcribe_audio
from src.speaker_identification import SpeakerAnalyzer
from src.azure_openai_provider import AzureOpenAIProvider, CompletionRequest
from src.models import SpeakerInfo, ProcessingResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_speaker_identification(audio_file_path, output_dir=None):
    """
    Test speaker identification functionality with an audio file.
    
    Args:
        audio_file_path: Path to the audio file to process
        output_dir: Directory to save output files (optional)
    """
    try:
        logger.info(f"Testing speaker identification with audio file: {audio_file_path}")
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get Azure credential with fallback mechanism
        credential = get_credential()
        
        # Get token for Azure services
        token = get_token(credential)
        
        # Step 1: Transcribe audio file with diarization enabled
        logger.info("Step 1: Transcribing audio file with speaker diarization")
        transcription, speaker_utterances = transcribe_audio(
            audio_file_path, 
            token, 
            enable_diarization=True
        )
        
        # Save transcription to file if output_dir is provided
        if output_dir:
            transcription_file = os.path.join(output_dir, "transcription_with_speakers.txt")
            with open(transcription_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            logger.info(f"Transcription with speakers saved to: {transcription_file}")
        
        # Step 2: Analyze speakers using the SpeakerAnalyzer
        logger.info("Step 2: Analyzing speakers and their roles")
        speaker_analyzer = SpeakerAnalyzer()
        
        # Add utterances to the speaker analyzer
        for speaker_id, text in speaker_utterances:
            speaker_analyzer.add_utterance(speaker_id, text)
        
        # Initialize Azure OpenAI provider with token-based authentication
        provider = AzureOpenAIProvider(use_token=True, token=token)
        
        # Analyze speaker roles
        def get_completion_func(request):
            return provider.get_completion(request)
        
        speaker_analyzer.analyze_with_llm(get_completion_func)
        
        # Get analysis results
        speaker_analysis_results = speaker_analyzer.get_analysis_results()
        
        # Save speaker analysis results to file if output_dir is provided
        if output_dir and speaker_analysis_results:
            analysis_file = os.path.join(output_dir, "speaker_analysis.json")
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(speaker_analysis_results, f, indent=2)
            logger.info(f"Speaker analysis results saved to: {analysis_file}")
        
        # Step 3: Create speaker info objects
        speaker_info_list = []
        if speaker_analysis_results:
            for speaker_id, role in speaker_analysis_results["roles"].items():
                confidence = speaker_analysis_results["confidence"].get(speaker_id, 0.7)
                reasoning = speaker_analysis_results["reasoning"].get(speaker_id, "")
                utterance_count = speaker_analyzer.get_utterance_count(speaker_id)
                
                speaker_info = SpeakerInfo(
                    id=speaker_id,
                    role=role,
                    confidence=confidence,
                    reasoning=reasoning,
                    utterance_count=utterance_count
                )
                speaker_info_list.append(speaker_info)
        
        # Create processing result
        result = ProcessingResult(
            transcription=transcription,
            speakers=speaker_info_list
        )
        
        # Save processing result to file if output_dir is provided
        if output_dir:
            result_file = os.path.join(output_dir, "processing_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            logger.info(f"Processing result saved to: {result_file}")
        
        logger.info("Speaker identification test completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in speaker identification test: {str(e)}")
        raise

def main():
    """Main function for running the test script."""
    parser = argparse.ArgumentParser(description="Test speaker identification functionality")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--output-dir", help="Directory to save output files")
    
    args = parser.parse_args()
    
    # Run the test
    test_speaker_identification(args.audio_file, args.output_dir)

if __name__ == "__main__":
    main()
