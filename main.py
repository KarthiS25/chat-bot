import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from streamlit_mic_recorder import mic_recorder, speech_to_text
import time
import pyttsx3

import speech_recognition as sr
from pydub import AudioSegment

import whisper
import sounddevice as sd
import numpy as np
import wave
import streamlit as st
import threading
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from IPython.display import Audio
from datasets import load_dataset
import torch
import io
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
audio_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
print("testtt")

model = whisper.load_model("base")
load_dotenv()

# Audio recording settings
SAMPLE_RATE = 16000  # Sample rate
CHANNELS = 1  # Mono audio
FILENAME = "recorded_audio.wav"

# Transcribe audio
def transcribe_audio(audio_data):
    print("Transcribing...")
    result = model.transcribe(audio_data)
    print("Extracted Text:", result["text"])
    return result["text"]

def text_to_speech(response):
    converter = pyttsx3.init()
    converter.setProperty('rate', 200)
    # converter.setProperty('volume', 0.8)
    converter.setProperty('voice', 'ta-IN')
    converter.say(response)
    # time.sleep(0.5)
    converter.runAndWait()
    # inputs = processor(text=response, return_tensors="pt")
    # speech = audio_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    # # st.audio(audio_bytes, format='audio/wav')
    # sd.play(speech.numpy(), samplerate=16000)

def get_audio_to_text():
    print("Recording...")
    r = sr.Recognizer()
    with sr.Microphone() as source:
      r.adjust_for_ambient_noise(source, duration=5)

      audio_data = r.record(source, duration=5)
      try:
          data = transcribe_audio(audio_data)
          return data
      except sr.UnknownValueError:
          print("Google Speech Recognition could not understand the audio.")
      except sr.RequestError as e:
          print(f"Could not request results from Google Speech Recognition service; {e}")
      except Exception as e:
          print(f"An error occurred: {e}")

model_id="mistralai/Mistral-7B-Instruct-v0.3"

def get_llm_hf_inference(model_id=model_id, max_new_tokens=128, temperature=0.1):
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token = os.getenv("HF_TOKEN")
    )
    return llm

# Configure the Streamlit app
st.set_page_config(page_title="Voice ChatBot", page_icon="🤖")
st.title("🤖 Voice ChatBot")
reset_history = st.button("Reset Chat History")
# st.markdown(f"*This is a simple chatbot that uses the HuggingFace transformers library to generate responses to your text input. It uses the {model_id}.*")

# Initialize session state for avatars
if "avatars" not in st.session_state:
    st.session_state.avatars = {'user': None, 'assistant': None}

# Initialize session state for user text input
if 'user_text' not in st.session_state:
    st.session_state.user_text = None

# Initialize session state for model parameters
if "max_response_length" not in st.session_state:
    st.session_state.max_response_length = 256

if "system_message" not in st.session_state:
    st.session_state.system_message = "friendly AI conversing with a human user"

if "starter_message" not in st.session_state:
    st.session_state.starter_message = "Hello, there! How can I help you today?"
    
st.session_state.avatars['assistant'] = "💬"
st.session_state.avatars['user'] = "👤"

# Initialize or reset chat history
if "chat_history" not in st.session_state or reset_history:
    st.session_state.chat_history = [{"role": "assistant", "content": st.session_state.starter_message}]

def get_response(system_message, chat_history, user_text, 
                eos_token_id=['User'], max_new_tokens=256, get_llm_hf_kws={}):
    # Set up the model
    hf = get_llm_hf_inference(max_new_tokens=max_new_tokens, temperature=0.1)

    # Create the prompt template
    prompt = PromptTemplate.from_template(
        (
            "[INST] {system_message}"
            "\nCurrent Conversation:\n{chat_history}\n\n"
            "\nUser: {user_text}.\n [/INST]"
            "\nAI:"
        )
    )
    # Make the chain and bind the prompt
    chat = prompt | hf.bind(skip_prompt=True) | StrOutputParser(output_key='content')

    # Generate the response
    response = chat.invoke(input=dict(system_message=system_message, user_text=user_text, chat_history=chat_history))
    response = response.split("AI:")[-1]

    # Update the chat history
    chat_history.append({'role': 'user', 'content': user_text})
    chat_history.append({'role': 'assistant', 'content': response})
    print("Testtt")
    return response, chat_history

# Chat interface
chat_interface = st.container(border=True)

with chat_interface:
  output_container = st.container()
  col1, col2 = st.columns([10, 1])
  with col1:
    user_input = st.chat_input(
      placeholder="Enter your text here or speak..."
    )
  with col2:
    audio_input = speech_to_text(language='en-IN', start_prompt="🎤", stop_prompt="🔴", just_once=False, use_container_width=False,callback=None,args=(),kwargs={},key=None)

  if audio_input:
    st.session_state.user_text = audio_input

  if user_input:
    st.session_state.user_text = user_input

# Display chat messages
with output_container:
  # For every message in the history
  for message in st.session_state.chat_history:
    # Skip the system message
    if message['role'] == 'system':
      continue
        
    # Display the chat message using the correct avatar
    with st.chat_message(message['role'], 
                          avatar=st.session_state['avatars'][message['role']]):
      st.markdown(message['content'])

    if len(st.session_state.chat_history) == 1 and not st.session_state.user_text:
      text_to_speech(st.session_state.starter_message)
            
  # When the user enter new text:
  if st.session_state.user_text:
    # Display the user's new message immediately
    with st.chat_message("user", 
                          avatar=st.session_state.avatars['user']):
      st.markdown(st.session_state.user_text)
        
    # Display a spinner status bar while waiting for the response
    with st.chat_message("assistant", 
                          avatar=st.session_state.avatars['assistant']):

      with st.spinner("Thinking..."):
        # Call the Inference API with the system_prompt, user text, and history
        response, st.session_state.chat_history = get_response(
            system_message=st.session_state.system_message, 
            user_text=st.session_state.user_text,
            chat_history=st.session_state.chat_history,
            max_new_tokens=st.session_state.max_response_length,
        )
        st.markdown(response)
        print(response, "Response")
      text_to_speech(response)