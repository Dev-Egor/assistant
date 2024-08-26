import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import time
import threading
from silero_vad import load_silero_vad
from torch import from_numpy


class STTClient:
    def __init__(self, sample_rate, stream_blocksize, model_type, device, compute_type, chunk_interval, chunk_duration, prompt_delay, no_merge_delay, vad_threshhold, input_language):
        
        self.sample_rate = sample_rate
        self.stream_blocksize = stream_blocksize
        self.chunk_interval = chunk_interval
        self.chunk_duration = chunk_duration
        self.prompt_delay = prompt_delay
        self.no_merge_delay = no_merge_delay
        self.vad_threshhold = vad_threshhold
        self.input_language = input_language
        self.speech_detected = False

        # Whisper model type
        print("Loading and configuring Whisper STT model...")
        self.stt_model = WhisperModel(model_type, device=device, compute_type=compute_type)

        # Load Silero VAD model
        print("Loading Silero VAD model...")
        self.vad_model = load_silero_vad()

        # Queue for audio chunks to be transcribed
        self.transcription_queue = queue.Queue()

        # Queue for transcribed overlapping text to be merged
        self.text_merge_queue = queue.Queue()

        # Ring buffer for storing continuous mic input
        print("Creating ring buffer for recording mic input...")
        self.audio_buffer = self.AudioRingBuffer(sample_rate = self.sample_rate,
                                                 stream_blocksize = self.stream_blocksize,
                                                 chunk_interval = self.chunk_interval,
                                                 chunk_duration = self.chunk_duration,
                                                 transcription_queue = self.transcription_queue)
        
        # Separate thread for transcriptions to run
        print("Creating and starting audio transcribing thread...")
        self.transcribe_thread = threading.Thread(target=self.transcribe_queue)
        self.transcribe_thread.start()


    class AudioRingBuffer:
        def __init__(self, sample_rate, stream_blocksize, chunk_interval, chunk_duration, transcription_queue):
            self.chunk_interval = int(round(sample_rate * chunk_interval / stream_blocksize) * stream_blocksize)
            self.chunk_duration = int(round(sample_rate * chunk_duration / stream_blocksize) * stream_blocksize)
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
                self.transcription_queue.put(self.buffer[start_index:end_index])
            else:
                self.transcription_queue.put(np.concatenate([self.buffer[start_index:], self.buffer[:end_index]]))


    # Merge strings based on overlap
    # TODO Use similarity scores because identical matches are unreliable
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


    # Callback for input stream
    def inputstream_callback(self, indata, *_):
        self.audio_buffer.queue_audio_data(indata.squeeze())
        self.speech_detected = self.vad_model(from_numpy(indata.squeeze()), self.sample_rate).item() > self.vad_threshhold


    # Prompt stt model and continuously yield full output
    # TODO Test no_merge_iterations
    def prompt(self):
        mic_stream = sd.InputStream(samplerate=self.sample_rate,
                                    blocksize=self.stream_blocksize,
                                    channels=1,
                                    callback=self.inputstream_callback)
        
        self.audio_buffer.clear()
        self.transcription_queue.queue.clear()
        no_merge_iterations = round(self.no_merge_delay / self.chunk_interval)
        
        with mic_stream:
            while not self.speech_detected:
                time.sleep(0.001)
            time.sleep(self.chunk_interval)

            self.text_merge_queue.queue.clear()
            last_speech_detected = time.time()

            while True:
                new_string = self.text_merge_queue.get().strip().replace("\n", " ")
                if no_merge_iterations > 0:
                    no_merge_iterations -= 1
                    output = new_string
                else:
                    output = self.merge_string(output, new_string)
                self.text_merge_queue.task_done()
                yield output

                if self.speech_detected:
                    last_speech_detected = time.time()
                elif time.time() - last_speech_detected >= self.prompt_delay:
                    return


    # Transcribe audio chunk
    def transcribe_queue(self):
        while True:
            audio_chunk = self.transcription_queue.get()
            segments, _ = self.stt_model.transcribe(audio_chunk, language=self.input_language)
            text = ""
            for segment in segments:
                text += segment.text
            self.text_merge_queue.put(text)
            self.transcription_queue.task_done()