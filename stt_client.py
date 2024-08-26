import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading
import time


class STTClient:
    def __init__(self, sample_rate, model_type, device, compute_type, chunk_interval, chunk_duration, prompt_delay, no_merge_delay, vad_threshhold, input_language):
        self.sample_rate = sample_rate
        self.chunk_interval = chunk_interval
        self.chunk_duration = chunk_duration
        self.prompt_delay = prompt_delay
        self.no_merge_delay = no_merge_delay
        self.vad_threshhold = vad_threshhold
        self.input_language = input_language
        self.speech_detected = False

        # Whisper model type
        print("Loading and configuring Whisper STT model...")
        self.model = WhisperModel(model_type, device=device, compute_type=compute_type)

        # Queue for audio chunks to be transcribed
        self.transcription_queue = queue.Queue()

        # Queue for transcribed overlapping text to be merged
        self.text_merge_queue = queue.Queue()

        # Ring buffer for storing continuous mic input
        print("Creating ring buffer for recording mic input...")
        self.audio_buffer = self.AudioRingBuffer(sample_rate = self.sample_rate,
                                                 chunk_interval = self.chunk_interval,
                                                 chunk_duration = self.chunk_duration,
                                                 transcription_queue = self.transcription_queue)

        # separate thread for transcriptions to run
        print("Creating and starting audio transcribing thread...")
        self.transcribe_thread = threading.Thread(target=self.transcribe_queue)
        self.transcribe_thread.start()


    class AudioRingBuffer:
        def __init__(self, sample_rate, chunk_interval, chunk_duration, transcription_queue):
            callback_size = sample_rate * 0.026 # what the fuck?
            self.chunk_interval = int(round(sample_rate * chunk_interval / callback_size) * callback_size)
            self.chunk_duration = int(round(sample_rate * chunk_duration / callback_size) * callback_size)
            self.transcription_queue = transcription_queue
            self.capacity = self.chunk_interval + self.chunk_duration
            self.buffer = np.zeros(self.capacity)
            self.generated = 0
            self.index = 0


        # Reset buffer
        def clear(self):
            self.buffer = np.zeros(self.capacity)


        # Add audio data to transcription buffer
        def queue_audio_data(self, data):
            remaining_capacity = self.capacity - self.index
            if data.shape[0] > remaining_capacity:
                first_part = data[:remaining_capacity]
                second_part = data[remaining_capacity:]
                self.buffer[self.index:] = first_part
                self.buffer[:second_part.shape[0]] = second_part
                self.index = second_part.shape[0]
            else:
                self.buffer[self.index : self.index + data.shape[0]] = data
                self.index += data.shape[0]
            self.generated += data.shape[0]

            if self.generated >= self.chunk_interval:
                self.generate_chunk()
            
            
        # Transcribe chunk of size interval + context
        def generate_chunk(self):
            start_index = self.index - self.generated - self.chunk_duration
            end_index = self.index
            self.generated = 0

            if start_index >= 0:
                chunk = self.buffer[start_index:end_index]
            else:
                chunk = np.concatenate([self.buffer[start_index:], self.buffer[:end_index]])
            self.transcription_queue.put(chunk)


    # Prompt model and continuously yield full output
    def prompt(self):
        # Clear existing data
        self.audio_buffer.clear()
        self.transcription_queue.queue.clear()
        self.text_merge_queue.queue.clear()
        
        # Start the input audio stream with callback for adding audio data to ring buffer
        mic_stream = sd.InputStream(samplerate=self.sample_rate,
                                    channels=1,
                                    callback=lambda indata, *_: self.audio_buffer.queue_audio_data(indata.squeeze()))
        with mic_stream:
            # Wait for voice before starting
            while not self.speech_detected:
                time.sleep(0.01)
        
            # Iterations to avoid merging transcriptions
            no_merge_iterations = round(self.no_merge_delay / self.chunk_interval)

            output = ""
            last_output = ""
            last_output_time = time.time()

            while True:
                new_string = self.text_merge_queue.get()
                if new_string is not None:
                    new_string = new_string.strip().replace("\n", " ")
                    if no_merge_iterations > 0:
                        no_merge_iterations -= 1
                        output = new_string
                    else:
                        output = self.merge_string(output, new_string)
                self.text_merge_queue.task_done()

                if output != last_output:
                    last_output = output
                    last_output_time = time.time()
                    yield output
                elif time.time() - last_output_time >= self.prompt_delay:
                    return


    # Merge strings based on overlap
    def merge_string(self, s1, s2):
        m = len(s1)
        n = len(s2)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        longest = 0
        end = 0
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    dp[i][j] = 0
                elif s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] >= longest:
                        longest = dp[i][j]
                        end = i - 1
                else:
                    dp[i][j] = 0
        lcs = s1[end-longest+1 : end+1]
        overlap_merge = s1[:end-longest+1] + lcs + s2[s2.find(lcs) + longest:]
        return overlap_merge


    # Transcribe audio chunks from queue
    def transcribe_queue(self):
        while True:
            audio_chunk = self.transcription_queue.get()
            self.vad(audio_chunk)

            if self.speech_detected:
                segments, _ = self.model.transcribe(audio_chunk, language=self.input_language)
                text = ""
                for segment in segments:
                    text += segment.text
                self.text_merge_queue.put(text)
            else:
                self.text_merge_queue.put(None)
            self.transcription_queue.task_done()


    # Use RMS to detect voice activity in last second of audio chunk very primitively TODO silero maybe
    def vad(self, audio_chunk):
        self.speech_detected = np.sqrt(np.mean(np.square(audio_chunk[-self.sample_rate:]))) > self.vad_threshhold