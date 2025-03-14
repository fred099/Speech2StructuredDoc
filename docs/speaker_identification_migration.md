# Speaker Identification Migration Plan

## Overview

This document outlines the migration plan for integrating speaker identification functionality into the existing advisory meeting processing workflow. The integration enables real-time speaker diarization, role analysis (advisor vs. client), and enhanced structured data extraction with speaker information.

## Key Components

1. **SpeakerAnalyzer Class**: A dedicated module for speaker identification and role analysis
2. **Enhanced Pydantic Models**: Updated models to include speaker analysis data
3. **Real-time Processing Integration**: Modified recorder to support speaker identification during live meetings
4. **Structured Data Integration**: Updated extraction to incorporate speaker roles

## Migration Steps

### 1. Model Updates

The Pydantic models have been enhanced to support speaker identification:

- **SpeakerInfo**: Added confidence scores, reasoning fields, and utterance count
- **ProcessingResult**: Added speaker_analysis field to store speaker identification results
- **SpeakerAnalysisResult**: New model for storing speaker analysis results

### 2. Speaker Identification Module

A new module `src/speaker_identification.py` has been created with:

- **SpeakerAnalyzer**: Main class for tracking utterances and analyzing speaker roles
- **configure_diarization**: Helper function to configure Azure Speech SDK for diarization
- **get_completion_with_api_key**: Utility for direct API calls to Azure OpenAI

### 3. Real-time Meeting Processor Updates

The `realtime_meeting_processor.py` script has been updated to:

- Initialize a SpeakerAnalyzer instance
- Track utterances by speaker
- Analyze speaker roles using Azure OpenAI
- Include speaker information in structured data extraction
- Save speaker analysis results alongside other data

### 4. Integration with Existing Scripts

The following scripts should be updated to use the new functionality:

- **test_swedish_advisory_processing.py**: Add speaker identification to recorded file processing
- **main.py**: Update the main processing function to include speaker analysis

## Testing Plan

1. **Unit Tests**: Test the SpeakerAnalyzer class with mock data
2. **Integration Tests**: Test the real-time meeting processor with live audio
3. **Recorded File Tests**: Test processing of recorded audio files with speaker identification

## Usage Examples

### Initializing the SpeakerAnalyzer

```python
from src.speaker_identification import SpeakerAnalyzer, configure_diarization

# Initialize the analyzer
speaker_analyzer = SpeakerAnalyzer()

# Configure speech SDK for diarization
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
configure_diarization(speech_config)
```

### Adding Utterances

```python
# When a new utterance is detected
is_new_speaker = speaker_analyzer.add_utterance(speaker_id, text)

# If this is a new speaker with enough data, analyze roles
if is_new_speaker and speaker_analyzer.get_utterance_count(speaker_id) >= 2:
    speaker_analyzer.analyze_with_llm(get_completion_func)
```

### Getting Analysis Results

```python
# Get the analysis results
results = speaker_analyzer.get_analysis_results()

# Access the roles, confidence scores, and reasoning
roles = results["roles"]
confidence = results["confidence"]
reasoning = results["reasoning"]

# Create SpeakerInfo objects
speaker_info_list = []
for speaker_id, role in roles.items():
    speaker_info = SpeakerInfo(
        id=speaker_id,
        name=f"Speaker {speaker_id}",
        role=role,
        confidence=confidence.get(speaker_id, 0.7),
        reasoning=reasoning.get(speaker_id, ""),
        utterance_count=speaker_analyzer.get_utterance_count(speaker_id)
    )
    speaker_info_list.append(speaker_info)
```

## Environment Variables

The following environment variables are required:

- **SPEECH_API_KEY**: API key for Azure Speech service
- **SPEECH_REGION**: Region for the Azure Speech service (default: "swedencentral")
- **AZURE_OPENAI_API_KEY**: API key for Azure OpenAI service
- **AZURE_OPENAI_DEPLOYMENT**: Deployment name for the Azure OpenAI model (default: "gpt-4o-mini")

## Best Practices

1. **Model Selection**: Use the "fast" model for simple speaker analysis and the "smart" model for more complex analysis
2. **Error Handling**: Always handle errors gracefully, especially during real-time processing
3. **Confidence Thresholds**: Consider the confidence score when using the speaker role analysis
4. **Parallel Analysis**: For real-time applications, use the parallel analysis thread to avoid blocking the main thread

## Future Enhancements

1. **Speaker Verification**: Add biometric verification of known speakers
2. **Role Refinement**: Enhance role detection with more specific roles (e.g., financial advisor, investment advisor)
3. **Sentiment Analysis**: Add sentiment analysis for each speaker
4. **Topic Tracking**: Track topics discussed by each speaker

## Conclusion

This migration plan provides a comprehensive guide for integrating speaker identification functionality into the existing advisory meeting processing workflow. By following these steps, you can enhance your application with real-time speaker diarization and role analysis.
