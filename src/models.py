from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date

class AudioFormData(BaseModel):
    """
    Pydantic model for validating structured data extracted from audio transcriptions.
    """
    client_name: Optional[str] = Field(None, description="Name of the client or organization")
    meeting_date: Optional[date] = Field(None, description="Date of the meeting (YYYY-MM-DD)")
    key_points: Optional[str] = Field(None, description="Summary of the main points discussed")
    action_items: Optional[List[str]] = Field(None, description="List of action items or next steps")
    participants: Optional[List[str]] = Field(None, description="Names of participants in the meeting")

class ProcessingResult(BaseModel):
    """
    Pydantic model for the overall processing result.
    """
    transcription: str = Field(..., description="The full transcription text")
    structured_data: AudioFormData = Field(..., description="Structured data extracted from the transcription")
    transcription_url: Optional[str] = Field(None, description="URL to the stored transcription in blob storage")
    json_url: Optional[str] = Field(None, description="URL to the stored JSON in blob storage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transcription": "Meeting with Acme Corp on January 15th, 2025. Participants included John Doe and Jane Smith...",
                "structured_data": {
                    "client_name": "Acme Corp",
                    "meeting_date": "2025-01-15",
                    "key_points": "Discussed Q1 goals and project timeline",
                    "action_items": ["Schedule follow-up meeting", "Share project plan"],
                    "participants": ["John Doe", "Jane Smith"]
                },
                "transcription_url": "https://storage.blob.core.windows.net/outputs/meeting_transcription.txt",
                "json_url": "https://storage.blob.core.windows.net/outputs/meeting_data.json"
            }
        }
