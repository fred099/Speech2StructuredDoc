import azure.functions as func
import logging
import json
import tempfile
import os
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.auth import get_credential, get_token
from src.storage import get_blob_service_client, upload_file, download_file
from src.speech import transcribe_audio
from src.llm import extract_structured_data
from src.models import AudioFormData, ProcessingResult
from src.config import CONTAINER_NAME

app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", 
                 path=f"{CONTAINER_NAME}/input",
                 connection="AzureWebJobsStorage")
def process_audio_blob(myblob: func.InputStream):
    """
    Azure Function triggered when a new audio file is uploaded to the input container.
    Processes the audio file and uploads the results to the output container.
    
    Args:
        myblob: The input blob trigger binding
    """
    logging.info(f"Python blob trigger function processed blob: {myblob.name}")
    
    try:
        # Get Azure credentials
        credential = get_credential()
        token = get_token(credential)
        blob_service_client = get_blob_service_client(credential)
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(myblob.name).suffix) as temp_file:
            temp_file.write(myblob.read())
            temp_file_path = temp_file.name
        
        try:
            # Process the audio file
            transcription = transcribe_audio(temp_file_path, token)
            logging.info("Transcription completed successfully")
            
            # Extract structured data
            form_json_str = extract_structured_data(transcription, token)
            form_data = json.loads(form_json_str)
            validated_form = AudioFormData(**form_data)
            
            # Generate output blob names
            file_name = Path(myblob.name).stem
            transcription_blob_name = f"output/{file_name}_transcription.txt"
            json_blob_name = f"output/{file_name}_data.json"
            
            # Upload results to blob storage
            transcription_url = upload_file(
                blob_service_client, 
                CONTAINER_NAME, 
                transcription, 
                transcription_blob_name
            )
            
            json_url = upload_file(
                blob_service_client,
                CONTAINER_NAME,
                form_json_str,
                json_blob_name
            )
            
            # Create the processing result
            result = ProcessingResult(
                transcription=transcription,
                structured_data=validated_form,
                transcription_url=transcription_url,
                json_url=json_url
            )
            
            # Upload the full result as JSON
            result_blob_name = f"output/{file_name}_result.json"
            upload_file(
                blob_service_client,
                CONTAINER_NAME,
                result.model_dump_json(indent=2),
                result_blob_name
            )
            
            logging.info(f"Processing completed for {myblob.name}")
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logging.error(f"Error processing audio blob: {str(e)}")
        # Upload error information to blob storage
        try:
            error_blob_name = f"output/{Path(myblob.name).stem}_error.txt"
            upload_file(
                blob_service_client,
                CONTAINER_NAME,
                f"Error processing {myblob.name}: {str(e)}",
                error_blob_name
            )
        except Exception as upload_error:
            logging.error(f"Failed to upload error information: {str(upload_error)}")

@app.route(route="process", auth_level=func.AuthLevel.FUNCTION)
def process_audio_http(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger function to process an audio file uploaded via HTTP.
    
    Args:
        req: The HTTP request
        
    Returns:
        HTTP response with the processing results
    """
    logging.info("Processing audio file via HTTP trigger")
    
    try:
        # Check if the request contains a file
        audio_file = req.files.get('audio')
        if not audio_file:
            return func.HttpResponse(
                "Please upload an audio file in the request body with the key 'audio'",
                status_code=400
            )
        
        # Get Azure credentials
        credential = get_credential()
        token = get_token(credential)
        blob_service_client = get_blob_service_client(credential)
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_file.read())
            temp_file_path = temp_file.name
        
        try:
            # Process the audio file
            transcription = transcribe_audio(temp_file_path, token)
            
            # Extract structured data
            form_json_str = extract_structured_data(transcription, token)
            form_data = json.loads(form_json_str)
            validated_form = AudioFormData(**form_data)
            
            # Generate a unique filename based on timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"http_upload_{timestamp}"
            
            # Upload results to blob storage
            transcription_blob_name = f"output/{file_name}_transcription.txt"
            json_blob_name = f"output/{file_name}_data.json"
            
            transcription_url = upload_file(
                blob_service_client, 
                CONTAINER_NAME, 
                transcription, 
                transcription_blob_name
            )
            
            json_url = upload_file(
                blob_service_client,
                CONTAINER_NAME,
                form_json_str,
                json_blob_name
            )
            
            # Create the processing result
            result = ProcessingResult(
                transcription=transcription,
                structured_data=validated_form,
                transcription_url=transcription_url,
                json_url=json_url
            )
            
            # Return the result as JSON
            return func.HttpResponse(
                result.model_dump_json(),
                mimetype="application/json",
                status_code=200
            )
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logging.error(f"Error processing HTTP request: {str(e)}")
        return func.HttpResponse(
            f"Error processing audio file: {str(e)}",
            status_code=500
        )
