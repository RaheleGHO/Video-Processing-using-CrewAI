from crewai import Task
from agents import (
    video_processor_agent,
    transcription_agent,
    description_agent,
    image_selector_agent
)
from tools import (
    FrameExtractionTool,
    AudioExtractionTool,
    AudioTranscriptionTool,
    LanguageDetectionTool,
    TextTranslationTool,
    DescriptionGenerationTool,
    ImageSelectionTool
)

# Task Definitions

# Task 1: Extract Frames and Audio
extract_media_task = Task(
    description=(
        "Process the video file located at '{video_path}'. "
        "Use the Frame Extraction Tool to extract frames at a 120-second interval (saving them to 'outputs/frames/'). " # Reverted interval
        "Then, use the Audio Extraction Tool to extract the audio track (saving it to 'outputs/audio/audio.mp3'). "
        "Return the list of extracted frame file paths and the path to the extracted audio file."
    ),
    expected_output=(
        "A dictionary containing: "
        "- 'frame_paths': A list of strings, each being the path to an extracted frame image file. "
        "- 'audio_path': A string representing the path to the extracted audio file (e.g., 'outputs/audio/audio.mp3')."
    ),
    agent=video_processor_agent,
    tools=[FrameExtractionTool(), AudioExtractionTool()]
    # No context needed initially, inputs provided via crew kickoff.
)

# Task 2: Transcribe and Translate Audio
transcribe_task = Task(
    description=(
        "Take the audio file path from the context ('audio_path'). "
        "Use the Audio Transcription Tool to transcribe the audio into text. "
        "After transcription, use the Language Detection Tool to determine the language of the transcript. "
        "Compare the detected language with the target language: '{target_lang}'. "
        "If they differ, use the Text Translation Tool to translate the transcript to the '{target_lang}'. "
        "Return the final transcript text in the target language."
    ),
    expected_output=(
        "A string containing the final transcript, accurately representing the audio content "
        "and translated to the '{target_lang}' if necessary."
    ),
    agent=transcription_agent,
    context=[extract_media_task], # Needs 'audio_path' from extract_media_task
    tools=[AudioTranscriptionTool(), LanguageDetectionTool(), TextTranslationTool()]
)

# Task 3: Generate Description
generate_description_task = Task(
    description=(
        "Analyze the transcript text provided in the context. "
        "Use the Description Generation Tool to create a concise and informative description based on the transcript. "
        "The description should be suitable for the content (product or general). "
        "Ensure the final description is in the target language: '{target_lang}'. The tool handles internal translation if needed."
        "Return the generated description text."
    ),
    expected_output=(
        "A string containing the final generated description in the target language '{target_lang}'."
    ),
    agent=description_agent,
    context=[transcribe_task], # Needs the transcript from transcribe_task
    tools=[DescriptionGenerationTool(), TextTranslationTool()] # TextTranslationTool available if needed by agent logic
)

# Task 4: Select Relevant Images
select_images_task = Task(
    description=(
        "Review the generated description text and the list of extracted frame paths provided in the context. "
        "Use the Image Selection Tool to identify and select up to 5 image frames that best visually represent the description. "
        "The tool should filter, score (using vision analysis), and rank the frames. "
        "Return the list of file paths for the selected images."
    ),
    expected_output=(
        "A list of strings, where each string is the file path to a selected image frame (maximum 5 images)."
    ),
    agent=image_selector_agent,
    context=[extract_media_task, generate_description_task], # Needs 'frame_paths' and the description
    tools=[ImageSelectionTool()]
)

# Note on Context Passing:
# CrewAI implicitly passes the output of a task listed in another task's `context` array.
# For `extract_media_task` returning a dict {'frame_paths': [...], 'audio_path': '...'},
# - `transcribe_task` can access `context['audio_path']`.
# - `select_images_task` can access `context['frame_paths']`.
# For tasks returning strings (transcript, description), subsequent tasks receive that string directly in their context.
# The task descriptions guide the agents on what inputs to expect from the context.
