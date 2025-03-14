# Speaker Identification Integration Plan

## Overview

This document outlines the plan to integrate the dynamic speaker identification functionality into the existing Speech2StructuredDoc system. The goal is to enhance the system's ability to automatically identify speakers and their roles (advisor or client) during financial advisory meetings, both in real-time and with pre-recorded audio files.

## Current Components

1. **Main Application (`src/main.py`)**
   - Entry point for processing audio files
   - Handles authentication, transcription, and structured data extraction
   - Currently doesn't include speaker identification

2. **Test Swedish Advisory Processing (`scripts/test_swedish_advisory_processing.py`)**
   - Processes Swedish advisory meeting audio files
   - Transcribes audio and extracts structured data
   - Uses Pydantic models for validation
   - Doesn't currently identify speakers automatically

3. **Realtime Meeting Processor (`scripts/realtime_meeting_processor.py`)**
   - Records and processes meetings in real-time
   - Includes visualization and UI components
   - Has some speaker tracking but not the enhanced identification we developed

4. **Test Dynamic Speaker Identification (`scripts/test_dynamic_speaker_identification.py`)**
   - Our newly developed functionality
   - Successfully identifies multiple speakers and their roles
   - Uses Azure Speech SDK for diarization
   - Analyzes speaker roles using Azure OpenAI

5. **Generate Test Advisory Meeting (`scripts/generate_test_advisory_meeting.py`)**
   - Creates test audio files for development and testing
   - Simulates advisor and client interactions

6. **Pydantic Models (`src/models.py`)**
   - Already includes `ParticipantRole`, `SpeakerInfo`, and `SpeakerAnalysisResult` models
   - Needs to be integrated with our new speaker identification logic

## Integration Plan

### Phase 1: Extract Reusable Components

1. **Create a Speaker Identification Module**
   - Extract the `SpeakerAnalyzer` class from `test_dynamic_speaker_identification.py`
   - Move it to a new file: `src/speaker_identification.py`
   - Make it more modular and reusable

2. **Update Diarization Configuration**
   - Extract the diarization configuration code into utility functions
   - Ensure consistent configuration across all usage points

### Phase 2: Update Pydantic Models

1. **Enhance the `SpeakerInfo` Model**
   - Add confidence scores and reasoning fields
   - Ensure compatibility with our speaker analysis results

2. **Update the `AudioFormData` Model**
   - Ensure the `participants` field can be populated from speaker identification

### Phase 3: Integrate with Existing Scripts

1. **Update `test_swedish_advisory_processing.py`**
   - Add speaker identification functionality
   - Use the extracted `SpeakerAnalyzer` class
   - Populate the Pydantic models with speaker information

2. **Update `realtime_meeting_processor.py`**
   - Integrate the speaker identification functionality
   - Ensure real-time analysis works correctly
   - Update the visualization to show speaker roles

3. **Update `main.py`**
   - Add speaker identification as a step in the processing pipeline
   - Ensure it works with both real-time and file-based processing

### Phase 4: Testing and Validation

1. **Test with Generated Audio Files**
   - Use `generate_test_advisory_meeting.py` to create test files
   - Verify speaker identification works correctly

2. **Test with Real Audio Files**
   - Validate with actual advisory meeting recordings
   - Ensure accuracy in speaker role identification

3. **Test Real-time Processing**
   - Verify the system works correctly in real-time
   - Ensure performance is acceptable

### Phase 5: Documentation and Cleanup

1. **Update Documentation**
   - Document the new speaker identification functionality
   - Update the README with usage instructions

2. **Code Cleanup**
   - Remove redundant code
   - Ensure consistent error handling
   - Follow best practices for Azure service usage

## Implementation Details

### Key Functions to Implement

1. **`identify_speakers(audio_file_path)`**
   - Process an audio file to identify speakers
   - Return speaker information with roles

2. **`analyze_speaker_roles(utterances)`**
   - Analyze speaker utterances to determine roles
   - Use Azure OpenAI for analysis

3. **`configure_diarization(speech_config)`**
   - Configure speech recognition for optimal diarization
   - Set appropriate parameters for speaker separation

### Integration with Pydantic Models

The speaker identification results will be integrated with the existing Pydantic models:

```python
# Example of populated models after integration
speaker_analysis = SpeakerAnalysisResult(
    speakers=[
        SpeakerInfo(
            id="Speaker-1",
            name="Speaker 1",
            role="advisor",
            confidence=0.95,
            reasoning="Uses financial terminology and provides advice"
        ),
        SpeakerInfo(
            id="Speaker-2",
            name="Speaker 2",
            role="client",
            confidence=0.90,
            reasoning="Asks questions about investments"
        )
    ],
    transcription=[
        "[10:15:30] Speaker 1: Welcome to our advisory meeting.",
        "[10:15:35] Speaker 2: Thank you for having us."
    ]
)

# This will be used to populate the AudioFormData model
audio_form_data = AudioFormData(
    client_name="Acme Corp",
    meeting_date=date.today(),
    key_points="Discussed investment strategy",
    action_items=["Review portfolio", "Schedule follow-up"],
    participants=[
        ParticipantRole(name="Speaker 1", role="advisor"),
        ParticipantRole(name="Speaker 2", role="client")
    ]
)
```

## Timeline

1. **Phase 1 (Extract Components)**: 1 day
2. **Phase 2 (Update Models)**: 1 day
3. **Phase 3 (Integration)**: 2-3 days
4. **Phase 4 (Testing)**: 1-2 days
5. **Phase 5 (Documentation)**: 1 day

Total estimated time: 6-8 days

## Conclusion

This integration plan provides a structured approach to incorporating the dynamic speaker identification functionality into the existing Speech2StructuredDoc system. By following this plan, we'll enhance the system's ability to automatically identify speakers and their roles during financial advisory meetings, making the structured data extraction more accurate and useful.
