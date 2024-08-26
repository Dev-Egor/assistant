import os
import prompt_generator
import configparser

# Display available personas and store selected directory
print("Available personas:")
personas = [f for f in os.listdir("personas")]
for i, folder in enumerate(personas, start=1):
    print(f"{i}. {folder.capitalize()}")
selected = personas[int(input("Enter a number to select a persona: ")) - 1]
persona_dir = os.path.join("personas", selected)

# Parse global and persona configs
persona_config = configparser.ConfigParser()
persona_config.read(os.path.join(persona_dir, "config.ini"))
global_config = configparser.ConfigParser()
global_config.read("global_config.ini")

# User input section
stt_enabled = input("Enter 'y' to enable STT capabilities: ") == "y"
tts_enabled = input("Enter 'y' to enable TTS capabilities: ") == "y"
if input("Enter 'y' to generate new conversation history: ") == "y":
    prompt_generator.generate(persona_dir)

# Get rvc and language details from config
language = persona_config.get("Common", "Language")
rvc_enabled = persona_config.getboolean("TTS", "Rvc")

# Only import necessary clients
print("Importing dependencies...")
from llm_client import LLMClient
if tts_enabled:
    from xtts_client import XTTSClient
if tts_enabled and rvc_enabled:
    from rvc_client import RVCClient
if stt_enabled:
    from stt_client import STTClient

# Other imports
import json
import prompt_generator
import logging


class Initializer:
    def startup():
        # Create client for LLM
        llm_client =  LLMClient(
            client_url = persona_config.get("LLM", "Url"),
            api_key =  persona_config.get("LLM", "ApiKey"),
            model_name = persona_config.get("LLM", "Model"),
            temperature = persona_config.getfloat("LLM", "Temperature"),
            history = json.load(open(os.path.join(persona_dir, "history.json")))
        )

        # Create client for persona's TTS type if enabled
        if tts_enabled:
            if rvc_enabled:
                rvc_client = RVCClient(
                    model_path = os.path.join(persona_dir, "rvc/model.pth"),
                    index_path = os.path.join(persona_dir, "rvc/index.index")
                    )
            else:
                rvc_client = None
            if persona_config.get("TTS", "Type") == "xtts":
                tts_client = XTTSClient(
                    model_dir = os.path.join(persona_dir, "tts"),
                    config_path = os.path.join(persona_dir, "tts/config.json"),
                    voice_path = os.path.join(persona_dir, "tts/reference.wav"),
                    temperature = persona_config.getfloat("TTS", "Temperature"),
                    output_language = language,
                    rvc_client = rvc_client
                )

        # Create client for STT if enabled
        if stt_enabled:
            stt_client = STTClient(
                sample_rate = global_config.getint("STT", "SampleRate"),
                model_type = global_config.get("STT", "ModelEnglish") if language == "en" else global_config.get("STT", "ModelOther"),
                device = global_config.get("STT", "Device"),
                compute_type = global_config.get("STT", "ComputeType"),
                chunk_interval = global_config.getfloat("STT", "ChunkInterval"),
                chunk_duration = global_config.getfloat("STT", "ChunkDuration"),
                prompt_delay = global_config.getfloat("STT", "PromptDelay"),
                no_merge_delay = global_config.getfloat("STT", "NoMergeDelay"),
                vad_threshhold =  global_config.getfloat("STT", "VadThreshold"),
                input_language = language
            ) 

        print("LOADING COMPLETE!")
        logging.getLogger().setLevel(logging.WARNING) # fuck off chinese rvc lib bastards
        return llm_client, tts_client if tts_enabled else None, stt_client if stt_enabled else None