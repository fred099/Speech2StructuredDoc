# Speech2StructuredDoc

A Python application that processes audio files to extract structured data using Azure AI services.

## Overview

Speech2StructuredDoc transcribes audio files using Azure Speech Service and extracts structured data from the transcriptions using Azure OpenAI Service. The application validates the extracted data using Pydantic models and stores the results in Azure Blob Storage.

## Features

- Audio transcription using Azure Speech Service
- Structured data extraction using Azure OpenAI Service
- Data validation with Pydantic models
- Secure authentication using Azure DefaultAzureCredential
- Azure Blob Storage integration for storing results
- Azure Functions implementation for serverless deployment

## Prerequisites

- Python 3.8 or higher
- Azure subscription with the following services:
  - Azure Speech Service
  - Azure OpenAI Service
  - Azure Blob Storage
- Azure CLI installed (optional, for local development)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Speech2StructuredDoc.git
cd Speech2StructuredDoc
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on the provided `.env.sample`:

```bash
cp .env.sample .env
```

4. Update the `.env` file with your Azure service details.

## Configuration

The application uses environment variables for configuration. The following variables are required:

- `AZURE_TENANT_ID`: Your Azure tenant ID (for service principal authentication)
- `AZURE_CLIENT_ID`: Your Azure client ID (for service principal authentication)
- `AZURE_CLIENT_SECRET`: Your Azure client secret (for service principal authentication)
- `STORAGE_ACCOUNT_NAME`: Your Azure Storage account name
- `CONTAINER_NAME`: The container name for storing results
- `SPEECH_REGION`: The region of your Azure Speech Service
- `SPEECH_RESOURCE_NAME`: The name of your Azure Speech Service resource
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint
- `AZURE_OPENAI_API_VERSION`: The API version to use (default: 2023-12-01-preview)
- `LOG_LEVEL`: The logging level (default: INFO)

## Usage

### Process an audio file

```bash
python -m src.main audio_file.wav --output-dir ./output
```

This will:
1. Transcribe the audio file
2. Extract structured data from the transcription
3. Upload the results to Azure Blob Storage
4. Save the results locally to the specified output directory

## Key Scripts

### Test Scripts

#### Test All Services
```bash
python scripts/test_all_services.py
```
Tests connectivity to all Azure services (Speech, OpenAI, and Storage) to ensure proper configuration.

#### Test Swedish Advisory Processing
```bash
python scripts/test_swedish_advisory_processing.py
```
Processes a pre-recorded Swedish advisory meeting audio file, transcribes it, and extracts structured data. This script is useful for testing the full pipeline with Swedish language support.

#### Generate Test Advisory Meeting
```bash
python scripts/generate_test_advisory_meeting.py
```
Generates a synthetic Swedish advisory meeting audio file for testing purposes. The generated file includes all required fields for the Pydantic model validation.

### Demo Scripts

#### Demo with Recording
```bash
python scripts/demo_with_recording.py
```
Creates a video recording of the entire processing pipeline, including audio playback, transcription, and structured data extraction. This script is ideal for demonstrations to stakeholders.

#### Real-time Meeting Processor
```bash
python scripts/realtime_meeting_processor.py
```
Records and processes meetings in real-time using the microphone input. Features include:
- Live transcription of Swedish speech
- Visual interface showing transcription as it happens
- Structured data extraction during the meeting
- Video recording of the entire process
- Automatic saving of transcription and structured data

Controls:
- **P key**: Pause/resume recording and transcription
- **E key**: Extract structured data from current transcription
- **S key**: Save current transcription
- **Q key**: Quit application

Output files are saved in the `meeting_recordings` directory with timestamps:
- Video recording: `meeting_recording_YYYYMMDD_HHMMSS.mp4`
- Transcription: `transcription_YYYYMMDD_HHMMSS.txt`
- Structured data: `structured_data_YYYYMMDD_HHMMSS.json`

### Authentication Test Scripts

#### Test with API Keys
```bash
python scripts/test_with_api_keys.py
```
Tests both OpenAI and Speech services using API key authentication.

#### Test OpenAI with Managed Identity
```bash
python scripts/test_openai_with_managed_identity.py
```
Tests OpenAI with managed identity authentication (requires proper role assignments).

#### Test OpenAI with Hybrid Authentication
```bash
python scripts/test_openai_hybrid_auth.py
```
Tests OpenAI with fallback from managed identity to API key authentication.

#### Test Speech with Hybrid Authentication
```bash
python scripts/test_speech_hybrid_auth.py
```
Tests Speech service with hybrid authentication approach.

## Azure Functions Deployment

The application includes an Azure Functions implementation for serverless deployment.

1. Create an Azure Function App with Python runtime.
2. Deploy the application to the Function App.
3. Configure the application settings with the required environment variables.

## Architecture

The application consists of the following components:

- `src/main.py`: Main application logic
- `src/function_app.py`: Azure Functions implementation
- `src/auth.py`: Authentication using DefaultAzureCredential
- `src/storage.py`: Azure Blob Storage integration
- `src/speech.py`: Azure Speech Service integration
- `src/llm.py`: Azure OpenAI integration
- `src/models.py`: Pydantic models for data validation
- `src/config.py`: Configuration management

## Authentication

The application uses Azure's DefaultAzureCredential for authentication, which provides a robust authentication approach:

1. **Primary Authentication**: DefaultAzureCredential
   - Works with service principal credentials in development
   - Uses managed identity in production

2. **Fallback Authentication**: AzureCliCredential
   - Used as a fallback when DefaultAzureCredential fails
   - Useful for local development when service principal credentials might be for a different tenant

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
