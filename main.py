import time as pytime
import os
import json
import sounddevice as sd
import numpy as np
import soundfile as sf
import assemblyai as aai
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import queue
from gtts import gTTS
import simpleaudio as sa
import speech_recognition as sr  # Google Speech Recognition

# Replace with your AssemblyAI API key
aai.settings.api_key = ""

# Initialization for OpenAI
client = OpenAI(api_key='')

# Define the queue at the top level
q = queue.Queue()


# Function to capture audio
def capture_audio():
    noise_threshold = 0.0001  # Adjust this value based on your noise level

    # Query the default sample rate of the input device
    default_samplerate = int(sd.query_devices(kind='input')['default_samplerate'])
    print(f"Using default sample rate: {default_samplerate} Hz")  # Debug statement

    # Initialize a list to store audio frames
    audio_frames = []
    is_speaking = False
    silence_start = None

    def callback(indata, frames, callback_time, status):
        nonlocal is_speaking, silence_start
        audio_frames.append(indata.copy())  # Store audio data for debugging
        volume_norm = (indata ** 2).mean() ** 0.5

        # Limit the frequency of debug output
        if len(audio_frames) % 15 == 0:
            print(f"Measured volume norm: {volume_norm}")  # Debug statement

        if volume_norm > noise_threshold:
            is_speaking = True
            silence_start = None
        else:
            if is_speaking:
                if silence_start is None:
                    silence_start = pytime.time()
                elif pytime.time() - silence_start > 1.0:  # Adjust the silence duration as needed
                    is_speaking = False
                    audio_data = np.concatenate(audio_frames, axis=0)
                    q.put(audio_data)
                    audio_frames.clear()
                    silence_start = None

    with sd.InputStream(callback=callback, channels=1, samplerate=default_samplerate):
        while True:
            try:
                audio_data = q.get(timeout=1)
                if audio_data is not None:
                    # Write audio data to a file for debugging
                    audio_file_path = "recorded_audio.wav"
                    sf.write(audio_file_path, audio_data, default_samplerate)
                    print(f"Saved recorded audio to {audio_file_path}")
                    return audio_file_path
            except queue.Empty:
                continue
 # Adding a small delay to reduce loop frequency


# Function to transcribe audio using AssemblyAI
def transcribe_audio(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
        return None
    else:
        return transcript.text


# Function to transcribe audio using Google Speech Recognition for low latency check
def quick_transcribe_google(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""


# Function to process text with GPT-4
def process_with_gpt(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# Function to convert text to speech using gtts and play using pydub and simpleaudio
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save('output.mp3')
    audio = AudioSegment.from_mp3('output.mp3')
    play(audio)


# Main loop
def main():
    conversation_history = ""
    pause_flag = False

    try:
        while True:
            if not pause_flag:
                print("Listening...")
                audio_file_path = capture_audio()
                print("Quick transcription check with Google STT...")
                quick_text = quick_transcribe_google(audio_file_path)
                print(quick_text)
                if quick_text == "":
                    print("No speech detected by Google STT.")
                    continue
                print("Transcribing audio with AssemblyAI...")
                text = transcribe_audio(audio_file_path)
                if text is None:
                    print("Error in transcription")
                    continue
                if text == "":
                    print("Empty transcription from AssemblyAI")
                    continue
                print(f"Heard: {text}")  # Debug statement
                if text.lower() in ["clear memory", "erase memory"]:
                    conversation_history = ""
                    text_to_speech("Memory cleared")
                    continue
                if text.lower() in ["stop"]:
                    text_to_speech("Stopping")
                    pause_flag = True
                    continue
                if text.lower() in ["start"]:
                    text_to_speech("Resuming")
                    pause_flag = False
                    continue

                conversation_history += f"User: {text}\n"
                response = process_with_gpt(conversation_history)
                conversation_history += f"Assistant: {response}\n"
                text_to_speech(response)
            else:
                print("Paused")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
