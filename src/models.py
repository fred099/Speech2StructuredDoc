from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict
from datetime import date

class ParticipantRole(BaseModel):
    """
    Pydantic model for participant with role information.
    """
    name: str = Field(..., description="Name of the participant")
    role: Literal["advisor", "client", "unknown"] = Field(..., description="Role of the participant (advisor, client, or unknown)")

class AudioFormData(BaseModel):
    """
    Pydantic model for validating structured data extracted from audio transcriptions.
    """
    client_name: Optional[str] = Field(None, description="Name of the client or organization")
    meeting_date: Optional[date] = Field(None, description="Date of the meeting (YYYY-MM-DD)")
    key_points: Optional[str] = Field(None, description="Summary of the main points discussed")
    action_items: Optional[List[str]] = Field(None, description="List of action items or next steps")
    participants: Optional[List[ParticipantRole]] = Field(None, description="Participants in the meeting with their roles")

class ProcessingResult(BaseModel):
    """
    Pydantic model for the overall processing result.
    """
    transcription: str = Field(..., description="The full transcription text")
    structured_data: AudioFormData = Field(..., description="Structured data extracted from the transcription")
    transcription_url: Optional[str] = Field(None, description="URL to the stored transcription in blob storage")
    json_url: Optional[str] = Field(None, description="URL to the stored JSON in blob storage")
    speaker_analysis: Optional["SpeakerAnalysisResult"] = Field(None, description="Results of speaker identification and analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transcription": "Meeting with Acme Corp on January 15th, 2025. Participants included John Doe and Jane Smith...",
                "structured_data": {
                    "client_name": "Acme Corp",
                    "meeting_date": "2025-01-15",
                    "key_points": "Discussed Q1 goals and project timeline",
                    "action_items": ["Schedule follow-up meeting", "Share project plan"],
                    "participants": [
                        {"name": "John Doe", "role": "advisor"},
                        {"name": "Jane Smith", "role": "client"}
                    ]
                },
                "transcription_url": "https://storage.blob.core.windows.net/outputs/meeting_transcription.txt",
                "json_url": "https://storage.blob.core.windows.net/outputs/meeting_data.json",
                "speaker_analysis": {
                    "speakers": [
                        {
                            "id": "Speaker-1",
                            "name": "Speaker 1",
                            "role": "advisor",
                            "confidence": 0.95,
                            "reasoning": "Uses financial terminology and provides advice"
                        },
                        {
                            "id": "Speaker-2",
                            "name": "Speaker 2",
                            "role": "client",
                            "confidence": 0.90,
                            "reasoning": "Asks questions about investments"
                        }
                    ],
                    "transcription": [
                        "[10:15:30] Speaker 1: Welcome to our advisory meeting.",
                        "[10:15:35] Speaker 2: Thank you for having us."
                    ]
                }
            }
        }

class CompletionRequest(BaseModel):
    """
    Pydantic model for OpenAI completion requests.
    """
    prompt: str = Field(..., description="The user prompt for the completion")
    system_message: Optional[str] = Field(None, description="Optional system message to guide the model")
    temperature: Optional[float] = Field(0.7, description="Temperature for the completion")
    max_tokens: Optional[int] = Field(1000, description="Maximum number of tokens to generate")

class SpeakerInfo(BaseModel):
    """
    Pydantic model for speaker information.
    """
    id: str = Field(..., description="Unique identifier for the speaker")
    name: str = Field(..., description="Name or identifier of the speaker")
    role: Literal["advisor", "client", "unknown"] = Field(..., description="Role of the speaker (advisor, client, or unknown)")
    confidence: Optional[float] = Field(None, description="Confidence score for the role classification (0.0 to 1.0)")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the role classification")
    utterance_count: Optional[int] = Field(None, description="Number of utterances by this speaker")

class SpeakerAnalysisResult(BaseModel):
    """
    Pydantic model for speaker analysis results.
    """
    speakers: List[SpeakerInfo] = Field(..., description="List of speakers with their roles")
    transcription: List[str] = Field(..., description="Transcription lines with speaker information")
