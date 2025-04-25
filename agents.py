from crewai import Agent
from tools import (
    FrameExtractionTool,
    AudioExtractionTool,
    AudioTranscriptionTool,
    LanguageDetectionTool,
    TextTranslationTool,
    DescriptionGenerationTool,
    ImageSelectionTool
)
from openai import OpenAI

# Initialize the language model (e.g., OpenAI GPT-4)
# Ensure OPENAI_API_KEY is set in the environment
llm = OpenAI() # CrewAI will use the default client if none is explicitly passed to Agent

# --- Agent Definitions ---

# Agent responsible for initial video processing: frame and audio extraction
video_processor_agent = Agent(
    role='Video Pre-processor',
    goal='Extract required components (frames and audio) from the input video file efficiently.',
    backstory=(
        "An expert in multimedia file handling, specialized in quickly "
        "deconstructing video files into their core visual and auditory components "
        "for further analysis."
    ),
    tools=[FrameExtractionTool(), AudioExtractionTool()],
    verbose=True,
    allow_delegation=False,
    # llm=llm # You can explicitly pass the LLM if needed, otherwise it uses the default OpenAI client
)

# Agent responsible for transcription and translation
transcription_agent = Agent(
    role='Multilingual Transcription Specialist',
    goal=(
        'Transcribe the extracted audio accurately. If the transcription is not '
        'in the target language, detect its language and translate it.'
    ),
    backstory=(
        "A linguistics expert with a knack for understanding various accents and dialects, "
        "equipped with state-of-the-art speech-to-text technology and translation capabilities. "
        "Ensures transcriptions are accurate and adhere to the requested target language."
    ),
    tools=[AudioTranscriptionTool(), LanguageDetectionTool(), TextTranslationTool()],
    verbose=True,
    allow_delegation=False,
    # llm=llm
)

# Agent responsible for generating the description
description_agent = Agent(
    role='Content Analyst and Description Writer',
    goal='Analyze the video transcript to generate a concise and informative description, tailored for either product listings or general content summaries, in the target language.',
    backstory=(
        "A skilled analyst and writer, capable of extracting key information from text "
        "and synthesizing it into compelling descriptions. Understands the nuances between "
        "product specifications and general content narratives. Ensures the final description is accurate and in the correct language."
    ),
    tools=[DescriptionGenerationTool(), TextTranslationTool()], # Translation might be needed if description tool generates English first
    verbose=True,
    allow_delegation=False,
    # llm=llm
)

# Agent responsible for selecting the best images
image_selector_agent = Agent(
    role='Visual Relevance Analyst',
    goal='Select the most relevant and high-quality image frames from the extracted video frames that best match the generated description.',
    backstory=(
        "An AI agent with a keen eye for visual detail and relevance. Uses advanced vision models "
        "to compare image content against textual descriptions, selecting frames that are clear, "
        "representative, and align perfectly with the product or content described."
    ),
    tools=[ImageSelectionTool()],
    verbose=True,
    allow_delegation=False,
    # llm=llm # Vision capabilities might be implicitly handled by the tool if using GPT-4V
)
