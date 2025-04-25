import openai
import os

# Load API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key is missing. Set OPENAI_API_KEY as an environment variable.")

client = openai.OpenAI(api_key=api_key)

def transcribe(audio_path):
    """ Transcribe the audio file using OpenAI Whisper API """
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="json",  # Ensure the response is in JSON format
                language="en"  # Set language to English
                # Removed the complex prompt here
            )

        print("Transcription received:", transcription)  # Debug output
        
        # Check response type
        if hasattr(transcription, "text"):
            return transcription.text  # Return just the text without timestamp wrapper

        return []
    
    except Exception as e:
        print("Error during transcription:", str(e))
        return []
