# Video Analysis & Description Generator (CrewAI) [📄 Read Project Description](Project_Description.pdf)


This Streamlit application utilizes a team of AI agents (built with CrewAI) to analyze video files. It extracts key information, transcribes audio, generates descriptions, and selects relevant frames, with special handling for videos not discussing products.

## Core Functionality

1.  **Video Pre-processing:** Extracts key frames at intervals and the audio track (MP3) from the input video.
2.  **Transcription & Translation:** Transcribes the extracted audio using OpenAI Whisper. Detects the language and translates the transcript to the selected target language if necessary.
3.  **Content Analysis & Description:** Analyzes the transcript using an OpenAI model to determine if the video discusses a product or general content.
    *   **Product Video:** Generates a structured description including name, specs, condition, etc., translated to the target language.
    *   **General Content:** Outputs the specific message: "There is no description for the video because it does not talk about a product."
4.  **Image Selection:** If the video is identified as a product video, uses an OpenAI vision model to select up to 5 frames that best represent the generated product description. No images are selected for general content videos.
5.  **Multi-language Support:** Supports processing and output in English, German, French, Spanish, and Italian.

## Project Structure

*   `main.py`: Runs the Streamlit web interface.
*   `agents.py`: Defines the CrewAI agents (Video Pre-processor, Transcription Specialist, Content Analyst, Visual Relevance Analyst).
*   `tasks.py`: Defines the CrewAI tasks assigned to the agents.
*   `tools.py`: Implements custom CrewAI tools used by the agents (frame/audio extraction, transcription, translation, description generation, image selection).
*   `extract_audio.py`, `extract_frames.py`, `transcribe_audio.py`: Helper scripts containing the core logic used by the tools.
*   `requirements.txt`: Lists project dependencies.
*   `outputs/`: Directory where generated files (frames, audio, transcripts, description) are saved.
*   `temp_uploads/`: Temporary storage for uploaded videos during processing.

## Installation

1.  **Clone the repository:**
    ```bash
    # Replace with your actual repository URL if applicable
    git clone <repository_url>
    cd EbayProChange 
    ```
    (If you don't have a Git repository, just ensure you are in the project directory `EbayProChange`)

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the project root or export the variable directly:
    ```
    OPENAI_API_KEY='your_openai_api_key_here'
    ```
    *Note: CrewAI automatically loads `.env` files.*

## Usage

1.  **Run the Streamlit app:**
    ```bash
    streamlit run main.py
    ```

2.  **Open the provided local URL** in your web browser.

3.  **In the web interface:**
    *   Upload a video file (Supported formats: MP4, MOV, AVI, MKV).
    *   Select the desired target language for the output.
    *   Click the "Process Video with AI Crew" button.

4.  **Wait for processing:** The app will display a spinner while the AI agents work. This may take several minutes depending on video length and API response times.

5.  **View Results:** Once complete, the app will display:
    *   The final transcript (translated if necessary).
    *   The generated description (or the specific non-product message).
    *   Relevant video frames with download buttons (only if it was identified as a product video and relevant frames were found).

## Outputs

Generated files are saved in the `outputs/` directory:
*   `outputs/frames/`: Selected image frames (JPG format, only for product videos).
*   `outputs/audio/audio.mp3`: Extracted audio track.
*   `outputs/transcripts/transcript.txt`: The raw transcript before potential translation.
*   `outputs/description/description.txt`: The final generated description text.

## Dependencies

Key dependencies are listed in `requirements.txt`:
*   `streamlit`: For the web application interface.
*   `openai`: For interacting with OpenAI models (Whisper, GPT, Vision).
*   `crewai`, `crewai-tools`: For defining and running the AI agent crew.
*   `moviepy`: For audio extraction.
*   `opencv-python`: For video frame extraction.
*   `langdetect`: For detecting the language of the initial transcript.
*   `Pillow`: For image handling (loading frames for display/download).
*   `numpy`: Numerical library (often a dependency of CV/media libraries).
*   `langchain-core`: Core components for language model interactions (used by CrewAI).

## Notes

*   Video processing involves multiple API calls and can be time-consuming.
*   Ensure your `OPENAI_API_KEY` has sufficient permissions and credits.
*   The application cleans up the temporary uploaded video and unselected frames after successful processing.
