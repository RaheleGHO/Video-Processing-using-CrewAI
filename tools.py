import os
import logging
from typing import List, Dict, Tuple, Optional
# Attempt BaseTool import directly from crewai.tools
try:
    from crewai.tools import BaseTool
except ImportError:
    # Fallback if crewai.tools doesn't have it
    try:
         from langchain.tools import BaseTool
    except ImportError:
        # Final fallback to langchain_core
        from langchain_core.tools import BaseTool

from openai import OpenAI
from langdetect import detect
from PIL import Image
import cv2 # Add OpenCV import
import numpy as np # Add numpy import
import base64 # Add base64 import
import io # Add io import

# Import functions from existing modules
from extract_frames import extract_frames as extract_frames_func
from extract_audio import extract_audio as extract_audio_func
from transcribe_audio import transcribe as transcribe_audio_func
# We'll integrate ImageSelector logic directly or adapt its core parts

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================
# --- Custom CrewAI Tools ---
# ==============================
# Each class inherits from BaseTool and implements the _run method.
# These tools interact with helper functions from other modules or APIs.

class FrameExtractionTool(BaseTool):
    """
    A tool to extract image frames from a video file at regular intervals.
    It saves the extracted frames to the 'outputs/frames/' directory.
    """
    name: str = "Frame Extraction Tool"
    description: str = "Extracts frames from a video file at a specified interval (default 120 frames) and saves them to disk. Returns a list of paths to the extracted frames."

    # Using a larger default frame_interval can speed up processing for longer videos
    # by extracting fewer frames overall.
    def _run(self, video_path: str, frame_interval: int = 120) -> List[str]:
        """
        Executes the frame extraction process.
        Args:
            video_path: Path to the input video file.
            frame_interval: Interval (number of frames) between extracted frames.
        Returns:
            A list of file paths for the successfully extracted frames.
        """
        output_dir = "outputs/frames/"
        os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists
        try:
            # Call the actual extraction logic from the imported function
            extracted_frames_data = extract_frames_func(video_path, output_dir, frame_interval)
            # The helper function returns more data; we only need the file paths.
            frame_paths = [frame[0] for frame in extracted_frames_data]
            logger.info(f"FrameExtractionTool: Extracted {len(frame_paths)} frames to {output_dir}")
            return frame_paths
        except Exception as e:
            logger.error(f"FrameExtractionTool Error: {e}")
            return [] # Return empty list on error

class AudioExtractionTool(BaseTool):
    """
    A tool to extract the audio track from a video file.
    It saves the audio as an MP3 file in 'outputs/audio/'.
    """
    name: str = "Audio Extraction Tool"
    description: str = "Extracts the audio track from a video file and saves it as an MP3 file. Returns the path to the extracted audio file."

    def _run(self, video_path: str) -> str:
        """
        Executes the audio extraction process.
        Args:
            video_path: Path to the input video file.
        Returns:
            The file path of the extracted MP3 audio file, or empty string on error.
        """
        output_audio_path = "outputs/audio/audio.mp3"
        # Ensure the directory for the audio file exists
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        try:
            # Call the actual extraction logic from the imported function
            extract_audio_func(video_path, output_audio_path)
            logger.info(f"AudioExtractionTool: Extracted audio to {output_audio_path}")
            return output_audio_path
        except Exception as e:
            logger.error(f"AudioExtractionTool Error: {e}")
            return "" # Return empty string on error

class AudioTranscriptionTool(BaseTool):
    """
    A tool to transcribe audio content from a file into text using OpenAI's Whisper model.
    Saves the transcript to 'outputs/transcripts/'.
    """
    name: str = "Audio Transcription Tool"
    description: str = "Transcribes an audio file using OpenAI Whisper. Returns the transcription text."

    def _run(self, audio_path: str) -> str:
        """
        Executes the audio transcription process.
        Args:
            audio_path: Path to the input audio file (e.g., MP3).
        Returns:
            The transcribed text as a string, or empty string on error.
        """
        try:
            transcript = transcribe_audio_func(audio_path)
            # Save transcript for potential reuse or debugging
            transcripts_path = "outputs/transcripts/transcript.txt"
            os.makedirs(os.path.dirname(transcripts_path), exist_ok=True)
            with open(transcripts_path, 'w', encoding="utf-8") as f:
                f.write(transcript)
            logger.info(f"AudioTranscriptionTool: Transcribed audio from {audio_path}")
            return transcript
        except Exception as e:
            logger.error(f"AudioTranscriptionTool Error: {e}")
            return "" # Return empty string on error

class LanguageDetectionTool(BaseTool):
    """
    A tool to detect the language of a given piece of text using the 'langdetect' library.
    """
    name: str = "Language Detection Tool"
    description: str = "Detects the language of a given text. Returns the language code (e.g., 'en', 'de')."

    def _run(self, text: str) -> Optional[str]:
        """
        Executes the language detection process.
        Args:
            text: The input text to analyze.
        Returns:
            The detected language code (e.g., 'en', 'fr') as a string, or None if detection fails.
        """
        try:
            lang = detect(text)
            logger.info(f"LanguageDetectionTool: Detected language '{lang}'")
            return lang
        except Exception as e:
            logger.error(f"LanguageDetectionTool Error: Could not detect language - {e}")
            return None # Return None on error

class TextTranslationTool(BaseTool):
    """
    A tool to translate text from one language to another using an OpenAI model.
    It defaults to returning the original text if translation fails.
    """
    name: str = "Text Translation Tool"
    description: str = "Translates text to a specified target language using an OpenAI model (e.g., gpt-4o-mini). Returns the translated text."
    client: OpenAI = OpenAI() # Initialize OpenAI client instance for this tool

    def _run(self, text: str, target_lang: str) -> str:
        """
        Executes the text translation process.
        Args:
            text: The original text to translate.
            target_lang: The language code of the desired output language (e.g., 'de', 'es').
        Returns:
            The translated text as a string, or the original text if translation fails.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", # Specify the OpenAI model to use (gpt-4o-mini is often good for translation)
                messages=[{
                    "role": "system", # System message sets the context/instruction for the AI
                    "content": f"Translate the following text to {target_lang}. Maintain the original meaning and tone accurately."
                }, {
                    "role": "user",
                    "content": text
                }],
                temperature=0.3
            )
            translated_text = response.choices[0].message.content
            logger.info(f"TextTranslationTool: Translated text to {target_lang}")
            return translated_text
        except Exception as e:
            logger.error(f"TextTranslationTool Error: {e}")
            return text # Return original text if translation fails

class DescriptionGenerationTool(BaseTool):
    """
    A tool to generate a textual description of video content based on its transcript.
    It first classifies the content as 'Product' or 'General' using an OpenAI model,
    then generates a description following a specific format.
    If the content is 'General', it outputs a predefined message.
    Handles translation to the target language if necessary for product descriptions.
    Saves the final description to 'outputs/description/'.
    """
    name: str = "Description Generation Tool"
    description: str = "Generates a description for a video based on its transcript and target language. Returns the generated description, or a specific message if the video is not about a product."
    client: OpenAI = OpenAI() # Initialize OpenAI client

    def _run(self, transcript: str, target_lang: str) -> str:
        """
        Executes the description generation process.
        Args:
            transcript: The text transcript of the video audio.
            target_lang: The desired output language code for the description.
        Returns:
            The generated description string, or a predefined message for non-product content,
            or an error message string.
        """
        try:
            # Step 1: Prompt the LLM to classify content and generate an initial description (in English).
            prompt = """
            Analyze this video transcript. First, determine if this is a product video (about an item for sale) or general content.
            Output the classification on the FIRST line as either 'Type: Product' or 'Type: General'.
            On the next line, create the description:
            If 'Type: Product', start with 'Product: ' followed by:
            - Name/Type
            - Key specifications
            - Condition (new/used/etc)
            - Pricing (if mentioned)
            remember just maintain this predefined structure.
            If 'Type: General', start with 'Content: ' followed by:
            - Primary subject/scene
            - Key actions/events
            - Notable objects/features
            - Overall context

            Only include information clearly present in the transcript. Use concise language suitable for a listing or summary.

            Transcript:
            {transcript}
            """
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", # Use faster model for initial description
                messages=[{"role": "user", "content": prompt.format(transcript=transcript)}],
                temperature=0.7,
                stop=["\n\n"] # Encourage stopping after description
            )
            raw_output_en = response.choices[0].message.content.strip()
            logger.info(f"DescriptionGenerationTool: Raw LLM Output:\n{raw_output_en}")

            # --- Step 2: Parse the LLM's Raw Output ---
            # The output is expected to have 'Type: <type>' on the first line.
            lines = raw_output_en.split('\n', 1) # Split into max 2 parts: first line and the rest
            content_type = "Unknown" # Default type if parsing fails
            description_en = raw_output_en # Default description if parsing fails

            # Check if the first line looks like a type classification
            if len(lines) > 0 and lines[0].strip().startswith("Type:"):
                content_type = lines[0].replace("Type:", "").strip() # Extract 'Product' or 'General'
                # If there's a second part (the actual description), use it. Otherwise, keep description empty.
                description_en = lines[1].strip() if len(lines) > 1 else ""

            logger.info(f"DescriptionGenerationTool: Parsed Content Type = '{content_type}'")
            logger.info(f"DescriptionGenerationTool: Parsed English Description = '{description_en[:100]}...'") # Log snippet

            # --- Step 3: Determine Final Description Based on Content Type ---
            # The specific message to return for non-product videos.
            non_product_message = "There is no description for the video because it does not talk about a product."
            final_description = "" # Initialize final description

            if content_type == "General":
                # If LLM classified as General, use the predefined message.
                logger.info("DescriptionGenerationTool: Content is General. Using predefined non-product message.")
                final_description = non_product_message
            elif content_type == "Product":
                # If LLM classified as Product, use the generated English description.
                logger.info("DescriptionGenerationTool: Content is Product. Using generated description.")
                # Translate the product description if the target language is not English.
                if target_lang != "en" and description_en: # Only translate if needed and description exists
                    logger.info(f"DescriptionGenerationTool: Translating product description from EN to {target_lang}.")
                    translation_tool = TextTranslationTool()
                    final_description = translation_tool._run(text=description_en, target_lang=target_lang)
                else:
                     # Use the English description if target is 'en' or if translation wasn't needed/possible
                     final_description = description_en
            else:
                # Handle 'Unknown' content type or parsing errors
                logger.warning(f"DescriptionGenerationTool: Could not determine content type reliably. Using default error message.")
                # Optionally return a specific error or fallback message
                return "Error: Could not reliably generate description or classify content."


            # --- Step 4: Save the Final Description ---
            # Save the result (either the product description/translation or the non-product message).
            description_path = "outputs/description/description.txt"
            os.makedirs(os.path.dirname(description_path), exist_ok=True) # Ensure directory exists
            try:
                with open(description_path, 'w', encoding="utf-8") as f:
                    f.write(final_description)
                logger.info(f"DescriptionGenerationTool: Saved final description to {description_path}")
            except IOError as io_err:
                 logger.error(f"DescriptionGenerationTool: Failed to save description file {description_path}: {io_err}")
                 # Still return the description even if saving failed

            # Return the final description string
            return final_description

        # --- Global Error Handling for the Tool ---
        except Exception as e:
            logger.error(f"DescriptionGenerationTool: Unexpected error during execution: {e}", exc_info=True)
            return "Error occurred during description generation." # Return generic error message

class ImageSelectionTool(BaseTool):
    """
    A tool to select the most relevant image frames based on a textual description.
    It uses a multi-step process:
    1. Basic filtering (e.g., file size).
    2. Scoring remaining frames against the description using an OpenAI Vision model.
    3. Selecting the top-scoring frames up to a specified maximum.
    It specifically checks if the input description is the non-product message and skips
    selection in that case.
    """
    name: str = "Image Selection Tool"
    description: str = (
        "Selects the best representative image frames based on a description. "
        "Filters, scores (using Vision AI), and selects up to 'max_images' (default 5). "
        "Returns a list of selected image file paths. Returns an empty list if the description indicates non-product content."
    )
    client: OpenAI = OpenAI() # Initialize OpenAI client

    def _run(self, description: str, frame_paths: List[str], max_images: int = 5) -> List[str]:
        """
        Executes the image selection process.
        Args:
            description: The textual description to match images against.
            frame_paths: A list of file paths for the candidate image frames.
            max_images: The maximum number of images to select.
        Returns:
            A list of file paths for the selected images, or an empty list if none are selected
            or if the description is the non-product message.
        """
        logger.info(f"ImageSelectionTool: Starting selection process for description '{description[:50]}...' with {len(frame_paths)} candidate frames.")

        # --- Early Exit for Non-Product Content ---
        # Check if the description is the specific message indicating non-product content.
        non_product_message = "There is no description for the video because it does not talk about a product."
        if description.strip() == non_product_message:
            logger.info("ImageSelectionTool: Received non-product description message. Skipping image selection and returning empty list.")
            return []
            return [] # Return empty list immediately

        # --- Input Validation ---
        if not frame_paths:
            logger.warning("ImageSelectionTool: No frame paths were provided for selection. Returning empty list.")
            return []

        # --- Step 1: Basic Filtering ---
        # Apply simple filters (like minimum file size) to remove obviously bad frames quickly.
        filtered_frames = self._filter_frames(frame_paths)
        if not filtered_frames:
            logger.warning("ImageSelectionTool: No frames remained after initial filtering. Returning empty list.")
            return []
        logger.info(f"ImageSelectionTool: {len(filtered_frames)} frames passed initial filtering.")

        # --- Step 2: Score Filtered Frames using Vision Model ---
        # Iterate through the filtered frames and score them against the description.
        scored_frames = [] # List to hold tuples of (path, score)
        for frame_path in filtered_frames:
            # Optimization: Stop scoring if we've already found enough good images.
            if len(scored_frames) >= max_images:
                 logger.info(f"ImageSelectionTool: Found sufficient high-scoring images ({len(scored_frames)}/{max_images}). Stopping scoring early.")
                 break # Exit the loop

            try:
                # Double-check existence before attempting to score
                if not os.path.exists(frame_path):
                    logger.warning(f"ImageSelectionTool: Frame path disappeared before scoring, skipping: {frame_path}")
                    continue # Go to the next frame

                # Call the helper method to get the vision score
                score = self._score_frame_with_vision(frame_path, description)

                # Define a threshold for considering a frame 'good enough'
                score_threshold = 4.0
                if score > score_threshold:
                    scored_frames.append((frame_path, score)) # Add path and score to the list
                    logger.info(f"ImageSelectionTool: Scored {os.path.basename(frame_path)} = {score:.2f} (Above threshold)")
                else:
                     # Log frames that score below the threshold
                     logger.info(f"ImageSelectionTool: Scored {os.path.basename(frame_path)} = {score:.2f} (Below threshold, skipping)")
            except Exception as e:
                 # Log errors during the scoring of a specific frame but continue with others
                logger.error(f"ImageSelectionTool: Error scoring frame {frame_path}: {e}", exc_info=True)
                continue # Skip this frame and proceed to the next

        # --- Step 3: Select Top Scoring Frames ---
        if not scored_frames:
            # If no frames scored above the threshold, provide a fallback.
            logger.warning("ImageSelectionTool: No frames scored above the threshold.")
            # Fallback: Return the first few frames from the *filtered* list instead.
            # This provides some images even if none were deemed highly relevant by the vision model.
            selected_paths = filtered_frames[:min(max_images, len(filtered_frames))]
            logger.info(f"ImageSelectionTool: Falling back to selecting the first {len(selected_paths)} filtered frames.")
            return selected_paths

        # If we have scored frames, sort them by score in descending order
        scored_frames.sort(key=lambda item: item[1], reverse=True)

        # Select the paths of the top 'max_images' frames
        selected_paths = [item[0] for item in scored_frames[:max_images]]
        logger.info(f"ImageSelectionTool: Selected top {len(selected_paths)} images based on scores: {[os.path.basename(p) for p in selected_paths]}")

        return selected_paths # Return the final list of selected image paths

    # --- Helper Methods for ImageSelectionTool ---

    def _filter_frames(self, frame_paths: List[str]) -> List[str]:
        """
        Applies basic pre-filtering to a list of frame paths.
        Currently checks if the file exists and meets a minimum size requirement.
        Args:
            frame_paths: The initial list of frame file paths.
        Returns:
            A list of frame paths that passed the filtering criteria.
        """
        min_size_bytes = 1024  # Example: 1 Kilobyte minimum size to filter out potentially empty/corrupt files
        valid_frames = []
        for path in frame_paths:
            try:
                # Check if path exists and file size is above the minimum
                if os.path.exists(path) and os.path.getsize(path) >= min_size_bytes:
                    valid_frames.append(path)
                # Optional: Add more checks here (e.g., image dimensions, aspect ratio) if needed
            except OSError as e:
                 # Log errors if a file path is inaccessible during filtering
                logger.warning(f"ImageSelectionTool Filtering: Error accessing {path}: {e}")
        return valid_frames # Return the list of paths that passed filters

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        Reads an image file from disk, ensures it's in RGB format,
        converts it to JPEG format in memory, and encodes it as a base64 string
        suitable for use in OpenAI Vision API calls.
        Args:
            image_path: The file path of the image to encode.
        Returns:
            A base64 encoded string (with data URI prefix), or None if encoding fails.
        """
        try:
            # Open the image using PIL (Pillow)
            with Image.open(image_path) as img:
                # Ensure image is in RGB format, as required by some models/processes
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Create an in-memory bytes buffer to hold the JPEG data
                buffered = io.BytesIO()
                # Save the image to the buffer in JPEG format
                img.save(buffered, format="JPEG")
                 # Get the byte value of the buffer
                img_bytes = buffered.getvalue()
                # Encode these bytes into a base64 string
                base64_string = base64.b64encode(img_bytes).decode('utf-8')
                # Prepend the necessary data URI scheme for OpenAI API
                return f"data:image/jpeg;base64,{base64_string}"
        except Exception as e:
            # Log any errors during image opening, conversion, or encoding
            logger.error(f"ImageSelectionTool: Failed to encode image {image_path} to base64: {e}", exc_info=True)
            return None # Return None on failure

    def _score_frame_with_vision(self, frame_path: str, description: str) -> float:
        """
        Scores a single image frame based on its relevance to the provided description,
        using an OpenAI Vision model (like GPT-4o mini).
        Args:
            frame_path: File path of the image frame to score.
            description: The text description to compare the image against.
        Returns:
            A relevance score between 0.0 and 10.0, or 0.0 if scoring fails.
        """
        try:
            # --- API Key Check (Optional but Recommended) ---
            # If testing without making actual API calls, you might want to return a mock score.
            # if not os.environ.get("OPENAI_API_KEY"):
            #     logger.warning("ImageSelectionTool Scoring: OPENAI_API_KEY not set. Returning mock score 5.0.")
            #     return 5.0

            # --- Prepare Image Data ---
            # Encode the image file to a base64 string for the API call.
            base64_image = self._encode_image_to_base64(frame_path)
            if not base64_image:
                 # If encoding failed, log it and return a score of 0.
                logger.error(f"ImageSelectionTool Scoring: Failed to encode image {frame_path}. Returning score 0.")
                return 0.0

            # --- Construct Vision Prompt ---
            # Create the prompt asking the model to score the image against the description.
            prompt = f"""
            Analyze how well this image frame matches the following description.
            Consider image clarity, relevance of visible elements to the description text, and overall visual quality.
            Is the main subject of the description clearly visible and well-represented?

            Description: "{description}"

            Score the match on a scale from 0 to 10, where 0 means no relevance and 10 means a perfect visual representation of the description.
            Return ONLY the numeric score (e.g., "7.5").
            """

            # --- Call OpenAI Vision API ---
            try:
                # Ensure OpenAI client is initialized
                if not hasattr(self, 'client') or self.client is None:
                     self.client = OpenAI()

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini", # A capable vision model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt}, # The text part of the prompt
                                {
                                    "type": "image_url",
                                     # Pass the base64 encoded image using the data URI scheme
                                    "image_url": {"url": base64_image}
                                },
                            ],
                        }
                    ],
                    max_tokens=10 # Requesting only a short response (the score)
                )
                # Extract the score text from the response
                score_text = response.choices[0].message.content.strip()

                # --- Parse Score ---
                # Attempt to convert the received text into a float.
                try:
                    score = float(score_text)
                except ValueError:
                     # Handle cases where the model didn't return a valid number
                    logger.warning(f"ImageSelectionTool Scoring: Could not parse score '{score_text}' from API response for {frame_path}. Defaulting score to 0.")
                    score = 0.0
            except Exception as api_err:
                 # Handle potential errors during the API call itself
                 logger.error(f"ImageSelectionTool Scoring: OpenAI API call failed for {frame_path}: {api_err}", exc_info=True)
                 score = 0.0 # Assign score 0 if API call fails

            # --- Final Score Clamping ---
            # Ensure the score is within the expected 0.0 to 10.0 range.
            final_score = max(0.0, min(10.0, score))
            return final_score

        # --- Global Error Handling for Scoring Method ---
        except Exception as e:
            # Catch any other unexpected errors during the scoring process for this frame.
            logger.error(f"ImageSelectionTool Scoring: Unexpected error for frame {frame_path}: {e}", exc_info=True)
            return 0.0 # Return 0 score on unexpected errors
