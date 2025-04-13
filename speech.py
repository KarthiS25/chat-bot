import speech_recognition as sr
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO
r = sr.Recognizer() 

# with sr.Microphone() as source:
#     print("Adjusting for ambient noise... Please wait.")
#     r.adjust_for_ambient_noise(source, duration=1)
#     print("Listening...")

#     audio_data = r.record(source, duration=5)
#     try:
#         print("Recognizing...")
#         text = r.recognize_google(audio_data, language="en-IN")  
#         print("You said:", text)
#     except sr.UnknownValueError:
#         print("Google Speech Recognition could not understand the audio.")
#     except sr.RequestError as e:
#         print(f"Could not request results from Google Speech Recognition service; {e}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# Python program to show 
# how to convert text to speech 
# import pyttsx3
# import time

# def text_to_speech(response):
#     converter = pyttsx3.init()
#     converter.setProperty('rate', 150)
#     converter.setProperty('volume', 0.8)
#     converter.say(response)
#     # time.sleep(0.5)
#     converter.runAndWait()

# text_to_speech("Hello")

# import whisper
# import sounddevice as sd
# import numpy as np
# import queue

# model = whisper.load_model("base")

# # Define audio parameters
# SAMPLE_RATE = 16000  # Whisper works best with 16kHz
# CHANNELS = 1  # Mono
# BLOCKSIZE = 1024
# DURATION = 5

# def record_audio():
#     print("Recording... Press ENTER to stop.")
#     audio_data = []
    
#     def callback(indata, frames, time, status):
#         audio_data.append(indata.copy())
    
#     stream = sd.InputStream(
#         samplerate=SAMPLE_RATE,
#         channels=CHANNELS,
#         dtype='int16',
#         callback=callback
#     )
#     stream.start()
    
#     input()  # Waits for ENTER key
#     stream.stop()
#     stream.close()
    
#     return np.concatenate(audio_data, axis=0).flatten()

# def transcribe_audio(audio_data):
#     """Transcribe audio using Whisper"""
#     audio = np.array(audio_data, dtype=np.float16) / np.iinfo(np.int16).max  # Normalize
#     result = model.transcribe(audio)  # <-- Automatically handles everything
#     print(f"Transcription: {result['text']}")

# audio_data = record_audio()  # Capture live audio from mic
# transcribe_audio(audio_data)

# from openai import OpenAI
# client = OpenAI(api_key="")

# audio_file = open("harvard.wav", "rb")
# transcript = client.audio.transcriptions.create(
#   file=audio_file,
#   model="whisper-1",
#   response_format="text",
#   timestamp_granularities=["word"]
# )

import whisper

def record_audio():
  with sr.Microphone() as source:
      print("Adjusting for ambient noise... Please wait.")
      r.adjust_for_ambient_noise(source, duration=1)
      print("Listening...")

      audio_data = r.record(source, duration=3)
      return audio_data

model = whisper.load_model("small")  # or "small", "medium", "large"
audio_data = record_audio()
wav_file = audio_data.get_wav_data()
audio_np, sa = sf.read(BytesIO(wav_file), dtype='float32')
print("Done")
if sa != 16000:
   import librosa
   audio_np = librosa.resample(audio_np, orig_sr=sa, target_sr=16000)
# print("DONE", audio_np)
# print("DONE", sa)
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("HF_TOKEN"), "sdfdgfdfh")

audio_30s = np.zeros(480000, dtype=np.float32)
length = min(len(audio_np), 480000)
audio_30s[:length] = audio_np[:length]

# result = model.transcribe(audio_np)
# print(result, "Text")
# print(result["text"], "Text")