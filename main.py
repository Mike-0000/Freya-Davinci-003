import time as pytime
import os
import json
import sounddevice as sd
import numpy as np
import soundfile as sf
import webrtcvad
import queue
import simpleaudio as sa
from pydub import AudioSegment
import noisereduce as nr
from pynput import keyboard
from openai import OpenAI
import speech_recognition as sr
from threading import Thread

# Initialize OpenAI client with API key
client = OpenAI(api_key="")

# Define the queue at the top level for audio data
q = queue.Queue()

# Global variables
pause_flag = False  # Flag to pause and resume listening
stop_playback = False  # Flag to stop audio playback
vad_mode = 3  # VAD (Voice Activity Detection) aggressiveness mode (0-3)
vad = webrtcvad.Vad(vad_mode)

# Set the sample rate and frame duration for audio capture
sample_rate = 16000  # 16kHz sample rate
frame_duration_ms = 20  # Frame duration in milliseconds (10ms, 20ms, or 30ms)
frame_length = int(sample_rate * frame_duration_ms / 1000)  # Frame length in samples


# Function to play audio files using simpleaudio
def play_audio(file_path):
    global stop_playback

    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()

    while play_obj.is_playing():
        if stop_playback:
            play_obj.stop()
            break
        pytime.sleep(0.1)
    stop_playback = False


# Function to capture audio from the microphone
def capture_audio():
    global pause_flag

    # Initialize variables for audio capture
    audio_frames = []
    is_speaking = False
    silence_start = None
    buffer = []  # Buffer to handle brief periods of silence
    buffer_duration = 2  # Buffer duration in seconds

    # Callback function to process audio frames
    def callback(indata, frames, callback_time, status):
        nonlocal is_speaking, silence_start, buffer

        if frames != frame_length:
            print(f"Unexpected frame length: {frames}")
            return  # Skip frames that do not match the expected length

        # Convert audio frame to byte array for VAD
        byte_data = (indata.flatten() * 32767).astype(np.int16).tobytes()

        try:
            if vad.is_speech(byte_data, sample_rate=sample_rate):
                if not is_speaking:
                    print("Speech detected")
                is_speaking = True
                # Append the buffer to audio_frames when speech is detected
                audio_frames.extend(buffer)
                buffer.clear()
                audio_frames.append(indata.copy())
                silence_start = None
            else:
                if is_speaking:
                    if silence_start is None:
                        silence_start = pytime.time()
                        print("Silence started")
                    elif pytime.time() - silence_start > 3.0:  # Adjust silence duration as needed
                        is_speaking = False
                        print("End of speech detected")
                        audio_data = np.concatenate(audio_frames, axis=0)
                        q.put(audio_data)  # Add the audio data to the queue
                        audio_frames.clear()
                        silence_start = None
        except Exception as e:
            print(f"Error during VAD processing: {e}")

        # Maintain a buffer of the last 2 seconds of audio
        buffer.append(indata.copy())
        if len(buffer) * frames / sample_rate > buffer_duration:
            buffer.pop(0)

    # Open an input stream to capture audio
    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, blocksize=frame_length, dtype='float32'):
        while not pause_flag:
            try:
                audio_data = q.get(timeout=1)
                if audio_data is not None:
                    # Apply noise reduction to the captured audio
                    reduced_noise_audio = nr.reduce_noise(y=audio_data.flatten(), sr=sample_rate, stationary=True,
                                                          prop_decrease=0.3)
                    audio_file_path = "recorded_audio.wav"
                    sf.write(audio_file_path, reduced_noise_audio, sample_rate)
                    print(f"Saved recorded audio to {audio_file_path}")
                    return audio_file_path
            except queue.Empty:
                continue

# Function to transcribe audio using OpenAI's Whisper API
def transcribe_audio_openai(file_path):
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text
# Function to transcribe audio using Google's Speech Recognition API
def transcribe_audio(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio)
            if text.strip() == "":
                print("No speech detected.")
                return None
            else:
                print("Google Speech Recognition thinks you said: " + text)
                return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None


# Function to process text with GPT-4
def process_with_gpt(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# Function to convert text to speech using OpenAI's TTS API and play it
def text_to_speech(text):
    global stop_playback

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    # Assuming the response content needs to be read directly as a binary stream
    speech_file_path = "output.mp3"
    with open(speech_file_path, "wb") as audio_file:
        audio_file.write(response.read())  # Use .read() to get the binary content

    # Add a brief silent segment and convert MP3 to WAV using pydub
    audio = AudioSegment.from_mp3(speech_file_path)
    silent_segment = AudioSegment.silent(duration=800)  # 800 ms silent segment
    audio = silent_segment + audio
    wav_file_path = "output.wav"
    audio.export(wav_file_path, format="wav")

    # Play the WAV file using simpleaudio in a separate thread
    play_thread = Thread(target=play_audio, args=(wav_file_path,))
    play_thread.start()

    # Check for "Stop" command while playing audio
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=sample_rate) as source:
        while play_thread.is_alive():
            try:
                audio = r.listen(source, timeout=1, phrase_time_limit=2)
                text = r.recognize_google(audio)
                if "stop" in text.lower():
                    stop_playback = True
                    break
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                break

    play_thread.join()


# Main loop to capture, transcribe, process, and respond to audio input
def main():
    global pause_flag
    conversation_history = ""

    try:
        while True:
            if not pause_flag:
                print("Listening...")
                audio_file_path = capture_audio()
                if audio_file_path:
                    print("Transcribing audio with Google Speech Recognition...")
                    text = transcribe_audio(audio_file_path)

                    if text is None or text.strip() == "":
                        print("Error in transcription or empty transcription")
                        continue
                    openai_text = transcribe_audio_openai(audio_file_path)
                    if openai_text is None or openai_text.strip() == "":
                        print("Error in transcription or empty transcription")
                        continue
                    print(f"Heard: {openai_text}")

                    # Commands to clear memory, stop, and start listening
                    if openai_text.lower() in ["clear memory.", "erase memory.", "clear memory", "erase memory"]:
                        conversation_history = ""
                        text_to_speech("Memory cleared!")
                        continue
                    if openai_text.lower() in ["stop.", "stop"]:
                        text_to_speech("Stopped!")
                        pause_flag = True
                        continue
                    if openai_text.lower() in ["start.", "start"]:
                        text_to_speech("Resuming!")
                        pause_flag = False
                        continue

                    # Process user input with GPT-4 and generate a response
                    conversation_history += f"User: {openai_text}\n"
                    response = process_with_gpt(conversation_history)
                    conversation_history += f"Assistant: {response}\n"
                    text_to_speech(response)
            else:
                pytime.sleep(1)
    except KeyboardInterrupt:
        pass


# Function to handle media play/pause key press to pause and resume listening
def on_press(key):
    global pause_flag
    try:
        if key == keyboard.Key.media_play_pause:
            pause_flag = not pause_flag
            if pause_flag:
                print("Listening paused.")
                text_to_speech("Listening paused!")
            else:
                print("Listening resumed.")
                text_to_speech("Listening resumed!")
    except AttributeError:
        pass


if __name__ == "__main__":
    # Start the keyboard listener for media play/pause key
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    main()
