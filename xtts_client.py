import sounddevice as sd
import numpy as np
import queue
import threading
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class XTTSClient:
    def __init__(self, model_dir, config_path, voice_path, temperature, output_language, rvc_client):
        # Get directores of files used in this class
        print("Getting TTS model and voice file directories...")
        self.model_dir = model_dir
        self.config_path = config_path
        self.voice_path = voice_path
        self.temperature = temperature
        self.output_language = output_language
        self.rvc_client = rvc_client

        print("Loading and configuring TTS model with DeepSpeed...")
        self.model = self.load_model()

        print("Cloning voice from wav file...")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path = [self.voice_path], load_sr = 48000)

        # Create audio playing thread
        print("Creating and starting audio playing thread...")
        self.play_queue = queue.Queue()
        self.play_thread = threading.Thread(target=self.play_audio_stream, daemon=True)
        self.play_thread.start()

        print("Warming up TTS model...")
        self.warmup()


    # Create TTS stream for given text segment
    def queue_tts(self, segment):
        tts_stream = self.model.inference_stream(
            text = segment,
            language = self.output_language,
            gpt_cond_latent = self.gpt_cond_latent,
            speaker_embedding = self.speaker_embedding,
            temperature = self.temperature
        )
        for audio_chunk in tts_stream:
            if self.rvc_client:
                self.play_queue.put(self.rvc_client.process(audio_chunk))
            else:
                self.play_queue.put(audio_chunk.cpu().numpy())

                
    # Audio playing stream
    def play_audio_stream(self):
        output_stream = sd.OutputStream(samplerate=48000 if self.rvc_client else 24000,
                                        channels=1,
                                        dtype=np.int16 if self.rvc_client else np.float32)
        with output_stream:
            while True:
                processed_audio_chunk = self.play_queue.get()
                output_stream.write(processed_audio_chunk)


    # Load and configure TTS model
    def load_model(self):
        config = XttsConfig()
        config.load_json(self.config_path) 
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=self.model_dir, use_deepspeed=True)
        model.cuda()
        return model
    

    # Warm up the model
    def warmup(self):
        tts_stream = self.model.inference_stream(
            "warmup",
            "en",
            self.gpt_cond_latent,
            self.speaker_embedding,
            self.temperature
        )
        for _ in tts_stream:
            pass