import os
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
import pyttsx3
import speech_recognition as sr
import whisper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from streamlit_mic_recorder import mic_recorder, speech_to_text
from dotenv import load_dotenv

import asyncio
import edge_tts
import playsound

model = whisper.load_model("base")
load_dotenv()

async def _text_to_speech_async(text, voice="en-IN-NeerjaExpressiveNeural", output="output.mp3"):
    voice = st.session_state.voice
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output)
    if text != st.session_state.starter_message:
      st.markdown(text)
    playsound.playsound(output)
    os.remove(output)

def text_to_speech(text, voice="en-IN-NeerjaExpressiveNeural"):
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      asyncio.run(_text_to_speech_async(text, voice))
    else:
      # If an event loop is already running (e.g., Streamlit), use create_task
      asyncio.create_task(_text_to_speech_async(text, voice))

# Transcribe audio
def transcribe_audio(audio_data):
  print("Transcribing...")
  result = model.transcribe(audio_data)
  print("Extracted Text:", result["text"])
  return result["text"]

model_id="mistralai/Mistral-7B-Instruct-v0.3"

def get_llm_hf_inference(model_id=model_id, max_new_tokens=128, temperature=0.1):
  llm = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation",
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    token = os.getenv("HF_TOKEN")
  )
  return llm

# Configure the Streamlit app
st.set_page_config(page_title="Voice ChatBot", page_icon="🤖")
st.title("Voice ChatBot")
col1, col2 = st.columns([10, 5])
with col1:
  reset_history = st.button("Reset Chat History")
with col2:
  col3, col4 = st.columns(2)
  with col3:
    gender = st.selectbox(
      "*Select Voice:*", options=["Male", "Female"], index=0, placeholder="Choose an option", disabled=False, label_visibility="visible"
    )

    if gender == "Male":
      st.session_state.voice = "en-IN-PrabhatNeural"
    else:
      st.session_state.voice = "en-IN-NeerjaExpressiveNeural"


# Initialize avatars
if "avatars" not in st.session_state:
  st.session_state.avatars = {'user': None, 'assistant': None}

# Initialize user text input
if 'user_text' not in st.session_state:
  st.session_state.user_text = None

# Initialize model parameters
if "max_response_length" not in st.session_state:
  st.session_state.max_response_length = 256

if "system_message" not in st.session_state:
  st.session_state.system_message = "friendly AI conversing with a human user"

if "starter_message" not in st.session_state:
  st.session_state.starter_message = "Hey! Ready to dive in? Tell me what you need."

if "voice"not in st.session_state:
  st.session_state.voice = "Male"

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
      # st.markdown(response)
      text_to_speech(response)
        # print(response, "Response")
      # text_to_speech(response)