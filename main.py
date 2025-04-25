# --- Standard Library Imports ---
import os
import logging
import re  # For parsing image paths from CrewAI output string
import shutil # For removing directories (e.g., cleanup)
from typing import List, Dict, Tuple, Optional # For type hinting

# --- Third-Party Library Imports ---
import streamlit as st # The Streamlit framework for building the UI
from PIL import Image # For handling image files (loading, displaying)
from crewai import Crew, Process, Task # Core components from the CrewAI library

# --- Local Application Imports ---
# Import the defined agents and tasks from other project files
from agents import video_processor_agent, transcription_agent, description_agent, image_selector_agent
from tasks import extract_media_task, transcribe_task, generate_description_task, select_images_task

# --- Logging Configuration ---
# Set up basic logging to show informational messages during execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Get a logger instance for this module

# =====================================
# --- Streamlit Application Class ---
# =====================================

class VideoApp:
    """
    Manages the Streamlit user interface and orchestrates the video processing
    workflow using CrewAI agents and tasks.
    """

    def __init__(self):
        """Initializes the VideoApp."""
        # Define the available languages for the user to select
        self.target_languages = {
            "English": "en",
            "German": "de",
            "French": "fr",
            "Spanish": "es",
            "Italian": "it"
        }
        # Note: The old self.video_path attribute is no longer needed as paths are handled locally in run().

    # This helper method was part of an older structure and is currently unused.
    # It's kept here for potential reference or future reintegration if needed.
    # def _upload_video(self) -> Optional[str]:
    #     """Handles video file upload and returns temporary file path."""
    #     uploaded_file = st.file_uploader(
    #         "Upload a video file",
    #         type=["mp4", "mov", "avi", "mkv"],
    #         help="Supported formats: MP4, MOV, AVI, MKV"
    #     )
    #     if uploaded_file is not None:
    #         temp_dir = "temp_uploads"
    #         os.makedirs(temp_dir, exist_ok=True)
    #         file_path = os.path.join(temp_dir, uploaded_file.name)
    #         with open(file_path, "wb") as f:
    #             f.write(uploaded_file.getbuffer())
    #         logger.info(f"Video uploaded and saved to temporary path: {file_path}")
    #         return file_path
    #     return None

    def run(self):
        """
        Defines the main Streamlit interface and application flow.
        Handles file uploads, triggers the CrewAI processing, and displays results.
        """
        st.title("Video Transcription & Description Generator (CrewAI)")

        # --- Session State Initialization ---
        # Streamlit reruns the script on interaction. Session state preserves data.
        # Initialize keys if they don't exist to avoid errors on first run or after clearing.
        if 'final_transcript' not in st.session_state:
            st.session_state.final_transcript = None # Stores the final transcript text
        if 'final_description' not in st.session_state:
            st.session_state.final_description = None # Stores the final description text
        if 'selected_image_data' not in st.session_state:
            # Stores loaded PIL Image objects for display (prevents reloading from disk)
            st.session_state.selected_image_data = []
        if 'valid_selected_paths' not in st.session_state:
             # Stores the file paths of the selected images (for download buttons)
            st.session_state.valid_selected_paths = []
        if 'processing_error' not in st.session_state:
            st.session_state.processing_error = None # Stores any error message during processing
        if 'show_results' not in st.session_state:
             # Boolean flag to control whether the results section is displayed
            st.session_state.show_results = False
        if 'last_uploaded_filename' not in st.session_state:
             # Stores the name of the last uploaded file to detect new uploads
             st.session_state.last_uploaded_filename = None

        # --- File Upload Section ---
        uploaded_file_obj = st.file_uploader(
            "Upload a video file",
            type=["mp4", "mov", "avi", "mkv"],
            help="Supported formats: MP4, MOV, AVI, MKV",
            key="file_uploader" # Assigning a key helps Streamlit track the widget state
        )

        # --- Process Uploaded File ---
        temp_video_path = None # Initialize path for the current uploaded video
        if uploaded_file_obj is not None:
            # Logic to handle a newly uploaded file
            # Compare current upload name with the stored name in session state
            if st.session_state.last_uploaded_filename != uploaded_file_obj.name:
                # If it's a new file, clear out all previous results and errors
                logger.info("New file uploaded, clearing previous session state results.")
                st.session_state.final_transcript = None
                st.session_state.final_description = None
                st.session_state.selected_image_data = [] # Clear loaded image objects
                st.session_state.valid_selected_paths = [] # Clear image paths
                st.session_state.processing_error = None # Clear any previous error message
                st.session_state.show_results = False # Hide results display area
                # Update the session state with the name of the newly uploaded file
                st.session_state.last_uploaded_filename = uploaded_file_obj.name

            # Save the uploaded file to a temporary directory
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True) # Ensure the directory exists
            temp_video_path = os.path.join(temp_dir, uploaded_file_obj.name)
            try:
                # Write the uploaded file's content to the temporary path
                # Using 'with open' ensures the file is properly closed afterwards
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file_obj.getbuffer()) # getbuffer() accesses the file data
                logger.info(f"Video uploaded and saved to temporary path: {temp_video_path}")
                st.video(temp_video_path) # Display the uploaded video in the UI
            except Exception as write_err:
                 # Handle potential errors during file saving
                 st.error(f"Error saving uploaded file: {write_err}")
                 logger.error(f"Error saving uploaded file {temp_video_path}: {write_err}")
                 temp_video_path = None # Reset path to None to prevent processing a failed save

        # --- Stop if No Video is Ready ---
        # If no video was uploaded or saving failed, show a warning and stop execution for this run.
        if temp_video_path is None:
            st.warning("Please upload a video file to begin.")
            # Optional: Clean up the temporary upload directory if it's empty
            temp_upload_dir = "temp_uploads"
            if os.path.exists(temp_upload_dir) and not os.listdir(temp_upload_dir):
                 try:
                     os.rmdir(temp_upload_dir)
                     logger.info(f"Cleaned up empty temp upload directory: {temp_upload_dir}")
                 except OSError:
                     pass # Ignore error if dir is somehow not empty
            # If no valid video path, stop the current script run for this user session.
            st.stop()

        # --- UI Elements for Processing Options (Only shown if video is ready) ---
        # Language selection dropdown
        target_lang_name = st.selectbox(
            label="Select target language:", # The text displayed above the dropdown
            options=list(self.target_languages.keys()), # Use the user-friendly names from the dictionary
            index=0, # Default to English
            help="Select the desired language for the final transcription and description."
        )
        # Get the corresponding language code (e.g., 'en', 'de') for the selected name
        target_lang_code = self.target_languages[target_lang_name]

        # --- Processing Trigger ---
        # This button starts the main video processing workflow
        if st.button("Process Video with AI Crew", key="process_button"):

            # --- Start of Processing Logic ---
            logger.info(f"Processing button clicked for video: {st.session_state.last_uploaded_filename}, Target Lang: {target_lang_code}")
            # Clear any previous results or errors from session state when starting a new process
            st.session_state.final_transcript = None
            st.session_state.final_description = None
            st.session_state.selected_image_data = []
            st.session_state.valid_selected_paths = []
            st.session_state.processing_error = None
            st.session_state.show_results = False # Ensure results are hidden until processing is done

            # Use the path of the currently uploaded video
            current_video_path = temp_video_path

            # --- Input Validation ---
            # Double-check if the video path is still valid before starting expensive AI tasks
            if not current_video_path or not os.path.exists(current_video_path):
                 st.error("Uploaded video file path seems invalid or missing. Please try re-uploading.")
                 logger.error(f"Invalid video path detected before starting tasks: {current_video_path}")
                 st.session_state.processing_error = "Uploaded video file path is invalid or missing."
            else:
                # --- Task Instantiation ---
                # Define the specific tasks for the CrewAI agents.
                # These tasks use the agents and tools imported earlier.
                # Context is passed between tasks implicitly by CrewAI based on the `context` list.

                # Task 1: Media Extraction (Frames & Audio)
                task1 = Task(
                    description=f"Process the video file at '{current_video_path}'. Extract frames and audio.", # Pass the actual video path
                    expected_output="Dictionary with 'frame_paths' (list) and 'audio_path' (string).",
                    agent=video_processor_agent,
                    # Tools are referenced from the pre-defined tasks in tasks.py for consistency,
                    # though they could be instantiated directly here too.
                    tools=[extract_media_task.tools[0], extract_media_task.tools[1]]
                )

                # Task 2: Transcription and Translation
                task2 = Task(
                    description=f"Transcribe audio from context ('audio_path'), detect language, translate to '{target_lang_code}' if needed.", # Pass target language code
                    expected_output=f"Final transcript text in {target_lang_code}.",
                    agent=transcription_agent,
                    context=[task1], # Depends on the audio_path output from task1
                    tools=[transcribe_task.tools[0], transcribe_task.tools[1], transcribe_task.tools[2]]
                )

                # Task 3: Description Generation
                task3 = Task(
                    description=f"Generate description from transcript (context) in '{target_lang_code}'.", # Pass target language code
                    expected_output=f"Generated description text in {target_lang_code}.",
                    agent=description_agent,
                    context=[task2],
                    tools=[generate_description_task.tools[0], generate_description_task.tools[1]]
                )

                # Task 4: Image Selection
                task4 = Task(
                    description=f"Select up to 5 image frames from context ('frame_paths') based on description (context).",
                    expected_output="List of selected image file paths.",
                    agent=image_selector_agent,
                    context=[task1, task3], # Depends on frame_paths (task1) and description (task3)
                    tools=[select_images_task.tools[0]]
                )

                # --- Crew Execution ---
                try:
                    # Display a spinner in the UI while the crew is working
                    with st.spinner("Processing video with AI Crew... This might take several minutes..."):

                        # Instantiate the Crew with the defined agents and tasks
                        video_crew = Crew(
                            agents=[video_processor_agent, transcription_agent, description_agent, image_selector_agent],
                            tasks=[task1, task2, task3, task4], # Use the locally instantiated tasks
                            process=Process.sequential, # Tasks will run one after another
                            verbose=True # Log detailed information about agent actions to the console
                        )

                        # Define the initial inputs required for the first task(s) in the crew
                        inputs = {
                            'video_path': current_video_path, # Path to the uploaded video
                            'target_lang': target_lang_code # Desired language code
                        }
                        logger.info(f"Starting CrewAI process with inputs: {inputs} for video: {current_video_path}")

                        # Ensure all necessary output directories exist before the crew starts
                        # This prevents errors if tools try to write to non-existent folders.
                        output_dirs = ["outputs/frames", "outputs/audio", "outputs/transcripts", "outputs/description", "temp_image_uploads"]
                        for dir_path in output_dirs:
                            os.makedirs(dir_path, exist_ok=True)

                        # --- Execute the Crew ---
                        # This starts the sequential execution of the tasks defined above.
                        # The 'inputs' dictionary provides the necessary starting data.
                        result = video_crew.kickoff(inputs=inputs)
                        # The 'result' usually contains the output of the *last* task in the sequence.
                        logger.info(f"CrewAI kickoff finished. Final result/output from last task: {result}")

                        # --- Result Extraction ---
                        # It's often more reliable to get outputs directly from the task objects
                        # after the crew has finished execution.
                        # Check if the task has an 'output' attribute and if it's not empty/None.
                        final_transcript = task2.output.raw if hasattr(task2, 'output') and task2.output else "Error: Transcript not found in task output."
                        final_description = task3.output.raw if hasattr(task3, 'output') and task3.output else "Error: Description not found in task output."
                        # Image selection result might be a string or a list, handle carefully.
                        raw_image_selection_result = task4.output.raw if hasattr(task4, 'output') and task4.output else None

                        logger.info(f"Retrieved from task outputs - Transcript: {final_transcript[:100]}...")
                        logger.info(f"Retrieved from task outputs - Description: {final_description[:100]}...")
                        logger.info(f"Retrieved from task outputs - Raw Images result: {raw_image_selection_result}")

                        # --- Image Path Parsing ---
                        # The image selection task might return a string representation of a list
                        # (e.g., "['path1', 'path2']") or an actual list. Need to parse robustly.
                        selected_images_paths = []
                        if isinstance(raw_image_selection_result, str):
                            logger.info("Attempting to parse image paths from string result.")
                            try:
                                # Try using regex to find paths enclosed in single quotes
                                paths_found = re.findall(r"'(.*?)'", raw_image_selection_result)
                                if paths_found:
                                    selected_images_paths = paths_found
                                    logger.info(f"Parsed image paths using regex: {selected_images_paths}")
                                else:
                                    # If regex fails, try a simpler split (less reliable)
                                    logger.warning("Regex failed to find paths, attempting simple split fallback.")
                                    # Remove potential brackets/quotes and split by common delimiters
                                    cleaned_result = raw_image_selection_result.strip("[]'\" ")
                                    # Split by comma or space, stripping extra quotes/spaces from each part
                                    potential_paths = [p.strip("'\" ") for p in re.split(r'[,\s]+', cleaned_result) if p]
                                    selected_images_paths = potential_paths # Use this less reliable result if needed
                                    logger.info(f"Parsed image paths using split fallback: {selected_images_paths}")
                            except Exception as parse_err:
                                logger.error(f"Error attempting to parse image paths string '{raw_image_selection_result}': {parse_err}")
                        elif isinstance(raw_image_selection_result, list):
                            # If the result is already a list, use it directly
                            selected_images_paths = raw_image_selection_result
                            logger.info(f"Image selection result is already a list: {selected_images_paths}")
                        else:
                             # Log a warning if the result is not a string or list
                             logger.warning(f"Unexpected type ({type(raw_image_selection_result)}) or None value received for raw image selection result.")

                        # --- Image Loading & Validation ---
                        # Process the list of potential image paths obtained above.
                        loaded_image_data = [] # To store PIL Image objects for display
                        validated_selected_paths = [] # To store confirmed existing paths for download links
                        if selected_images_paths: # Only proceed if we have some paths
                            for img_path in selected_images_paths:
                                # Check if the path is a string and if the file actually exists
                                if isinstance(img_path, str) and os.path.exists(img_path):
                                    validated_selected_paths.append(img_path) # Add to list of valid paths
                                    try:
                                        # Open the image file using PIL
                                        image = Image.open(img_path)
                                        # Append a *copy* to the list to avoid issues with file handles
                                        loaded_image_data.append(image.copy())
                                        image.close() # Explicitly close the file handle after copying
                                        logger.info(f"Validated and loaded selected image into memory: {img_path}")
                                    except Exception as load_err:
                                        # Log error if loading fails for a valid path
                                        logger.error(f"Failed to load image from valid path {img_path}: {load_err}")
                                else:
                                    # Log warning for invalid entries or non-existent files
                                    logger.warning(f"Invalid type ({type(img_path)}) or non-existent image path found in selection result, skipping: {img_path}")
                        logger.info(f"Successfully loaded {len(loaded_image_data)} images into memory.")

                        # --- Store Results in Session State ---
                        # Save the final results to Streamlit's session state so they persist
                        # across reruns and can be displayed later in the UI.
                        st.session_state.final_transcript = str(final_transcript) # Ensure it's a string
                        st.session_state.final_description = str(final_description) # Ensure it's a string
                        st.session_state.selected_image_data = loaded_image_data # List of PIL Image objects
                        st.session_state.valid_selected_paths = validated_selected_paths # List of string paths
                        st.session_state.processing_error = None # Clear any previous error
                        st.session_state.show_results = True # Set flag to display results section
                        # Show a success message in the Streamlit UI
                        st.success("AI Crew Processing Complete!")


                        # --- Post-Processing Cleanup ---
                        # Perform cleanup only AFTER results are successfully processed and stored.

                        # 1. Remove the temporary uploaded video file
                        if current_video_path and os.path.exists(current_video_path):
                            try:
                                temp_video_dir = os.path.dirname(current_video_path)
                                os.remove(current_video_path) # Delete the file
                                logger.info(f"Removed temporary video file: {current_video_path}")
                                # Try removing the temporary directory if it's now empty
                                if os.path.exists(temp_video_dir) and not os.listdir(temp_video_dir):
                                    os.rmdir(temp_video_dir)
                                    logger.info(f"Removed empty temporary video directory: {temp_video_dir}")
                            except OSError as e:
                                logger.error(f"Error during temporary video file/directory cleanup: {e}")

                        # 2. Cleanup unselected frame images to save space
                        frames_dir = "outputs/frames" # Directory where frames were saved
                        if os.path.exists(frames_dir):
                            try:
                                # Get a list of all files currently in the frames directory
                                all_extracted_frames = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))]
                                # Create a set of the valid selected paths for efficient lookup
                                selected_paths_set = set(validated_selected_paths)
                                deleted_count = 0
                                # Iterate through all extracted frames
                                for frame_path in all_extracted_frames:
                                    # If a frame is NOT in the set of selected paths, delete it
                                    if frame_path not in selected_paths_set:
                                        try:
                                            os.remove(frame_path)
                                            deleted_count += 1
                                        except OSError as rm_err:
                                            logger.error(f"Error removing unselected frame {frame_path}: {rm_err}")
                                logger.info(f"Cleanup: Removed {deleted_count} unselected frames from {frames_dir}. Kept {len(selected_paths_set)} selected frames.")
                                # If, after deleting unselected frames, the directory is empty, remove it too
                                if not os.listdir(frames_dir):
                                    try:
                                        os.rmdir(frames_dir)
                                        logger.info(f"Cleanup: Removed empty frames directory: {frames_dir}")
                                    except OSError as rmdir_err:
                                         logger.error(f"Cleanup: Error removing empty frames directory {frames_dir}: {rmdir_err}")
                            except OSError as e:
                                logger.error(f"Cleanup: Error accessing frames directory {frames_dir} for cleanup: {e}")

                # --- Global Error Handling for Crew Execution ---
                except Exception as e:
                    # Catch any exception during the crew's kickoff or result processing
                    error_message = f"An error occurred during the AI Crew process: {str(e)}"
                    st.error(error_message) # Show error in Streamlit UI
                    logger.error(f"Application error during CrewAI execution: {str(e)}", exc_info=True) # Log full traceback
                    # Store the error message in session state and hide results area
                    st.session_state.processing_error = error_message
                    st.session_state.show_results = False # Ensure results aren't shown if an error occurred

        # --- Display Results (or Error) ---
        # This section runs on every Streamlit rerun. It checks session state
        # to decide whether to show results or a previously stored error message.

        # Display error message if one occurred during the last processing attempt
        if st.session_state.processing_error:
             st.error(st.session_state.processing_error) # Show the stored error message

        # Display results if processing completed successfully (show_results is True)
        if st.session_state.show_results:
             # Check if transcript exists in session state before displaying
            if st.session_state.final_transcript:
                 self._display_transcription(st.session_state.final_transcript)
             # Check if description exists (use 'is not None' for clarity)
            if st.session_state.final_description is not None:
                # Call the helper method to display description and images
                self._display_description(
                    st.session_state.final_description,
                    st.session_state.selected_image_data,
                    st.session_state.valid_selected_paths # Pass the list of valid image paths
                )

    # --- Helper Methods for Displaying Results ---

    def _display_transcription(self, transcript: str) -> None:
        """Displays the final transcript in a text area."""
        st.subheader("Transcription") # Section header
        # Use a text area for potentially long transcripts, allowing scrolling
        st.text_area("Full Transcript:", value=transcript, height=200, key="transcript_output")

    def _display_description(self, description: str, selected_image_data: List[Image.Image], selected_image_paths: List[str]) -> None:
        """
        Displays the generated description and the selected relevant video frames (if any).
        Handles the logic for hiding the image section if the description indicates
        non-product content or if no images were selected.

        Args:
            description: The final description text.
            selected_image_data: A list of PIL Image objects (loaded images).
            selected_image_paths: A list of file paths corresponding to the loaded images.
        """
        st.subheader("Generated Description") # Section header
        st.write(description) # Display the description text

        # Define the exact message used when content is identified as non-product
        non_product_message = "There is no description for the video because it does not talk about a product."

        # --- Conditional Image Display Logic ---
        # Only show the 'Relevant Video Frames' section if images were actually selected.
        if selected_image_data: # Check if the list of loaded images is not empty
            st.subheader("Relevant Video Frames") # Section header for images
            st.write("These frames were selected by the AI Crew as most representative:") # Informational text
            # Display images in columns for better layout (up to 3 columns)
            num_images = len(selected_image_data)
            # Calculate number of columns: minimum of 3 or the actual number of images
            num_cols = min(3, num_images) if num_images > 0 else 1
            cols = st.columns(num_cols)

            # Iterate through the loaded image data and their corresponding paths
            for i, (img_data, img_path) in enumerate(zip(selected_image_data, selected_image_paths)):
                # Distribute images evenly across the columns
                col_index = i % num_cols
                with cols[col_index]:
                    # Double-check if the image file still exists at its path (it might have been deleted)
                    if os.path.exists(img_path):
                        try:
                            # Display the image using the loaded PIL object
                            st.image(img_data, caption=f"Selected Image {i+1}", use_container_width=True)
                            # Add a download button for the image
                            # Read the image file in binary mode for the download data
                            with open(img_path, "rb") as file_bytes:
                                st.download_button(
                                    label="Download Image", # Button text
                                    data=file_bytes, # The byte data of the image
                                    file_name=os.path.basename(img_path), # Use actual filename
                                    mime="image/jpeg", # Set the appropriate MIME type
                                    key=f"download_{i}" # Unique key for the button widget
                                )
                        except FileNotFoundError:
                             # This specific error might occur if file deleted between check and open
                             st.error(f"Image file disappeared before download: {os.path.basename(img_path)}")
                             logger.error(f"Race condition? File not found during download attempt: {img_path}")
                        except Exception as e:
                            # General error handling for display/download
                            st.error(f"Error processing image {i+1} for display/download.")
                            logger.error(f"Error in image display/download loop (index {i}, path {img_path}): {str(e)}")
                    else:
                         # Show a warning if the image file doesn't exist (e.g., due to cleanup)
                         st.warning(f"Image file missing: {os.path.basename(img_path)}")
                         logger.warning(f"Attempted to display/download a non-existent file: {img_path}")

            # Add a helpful tip about downloading if images were displayed
            if num_images > 0:
                 st.write("💡 Tip: Use the download buttons to save individual images.")

        # Logic for when no images are selected/loaded:
        elif description and description.strip() == non_product_message:
            # If the description is exactly the non-product message, show nothing more.
            # The description itself is already displayed above.
            pass # Explicitly do nothing in this case
        elif description and description.strip().startswith('Product:'):
            # If it looks like a product description but no images were selected/loaded, show a warning.
            st.subheader("Relevant Video Frames") # Keep header for consistency
            st.warning("No relevant images were selected or loaded for this product description.")
        # else: # Optional fallback for other cases where description exists but no images
             # st.subheader("Relevant Video Frames")
             # st.info("Image selection was not applicable or no relevant images found.")


# ==================================
# --- Main Execution Block ---
# ==================================
# This code runs only when the script is executed directly (not imported).
if __name__ == "__main__":
    # --- API Key Check ---
    # Crucial check: Ensure the OpenAI API key is set as an environment variable.
    # The application cannot function without it.
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key (OPENAI_API_KEY) not found in environment variables. Please set it to run the app.")
        logger.error("FATAL: OPENAI_API_KEY environment variable not set.")
    else:
        # --- App Instantiation and Run ---
        # If the API key exists, create an instance of the VideoApp class...
        app = VideoApp()
        # ...and call its run() method to start the Streamlit application.
        app.run()
