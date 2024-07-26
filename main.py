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

# Initialize OpenAI client
client = OpenAI(api_key="")

# Define the queue at the top level
q = queue.Queue()

# Global variables
pause_flag = False
vad_mode = 2  # VAD aggressiveness mode (0-3)
vad = webrtcvad.Vad(vad_mode)

# Set the sample rate and frame duration
sample_rate = 16000  # 16kHz sample rate
frame_duration_ms = 20  # Frame duration in milliseconds (10ms, 20ms, or 30ms)
frame_length = int(sample_rate * frame_duration_ms / 1000)  # Frame length in samples

# Function to play audio files using simpleaudio
def play_audio(file_path):
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

# Function to capture audio
def capture_audio():
    global pause_flag

    # Initialize variables
    audio_frames = []
    is_speaking = False
    silence_start = None
    buffer = []  # Buffer to handle brief periods of silence
    buffer_duration = 2  # Buffer duration in seconds
    debug_counter = 0  # Counter for limiting debug messages

    def callback(indata, frames, callback_time, status):
        nonlocal is_speaking, silence_start, buffer, debug_counter

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
                    reduced_noise_audio = nr.reduce_noise(y=audio_data.flatten(), sr=sample_rate,
                                                          stationary=True, prop_decrease=0.3)
                    audio_file_path = "recorded_audio.wav"
                    sf.write(audio_file_path, reduced_noise_audio, sample_rate)
                    print(f"Saved recorded audio to {audio_file_path}")
                    return audio_file_path
            except queue.Empty:
                continue

# Function to transcribe audio using OpenAI's Whisper API
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text

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

# Function to convert text to speech using OpenAI's TTS API and play it
def text_to_speech(text):
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

    # Play the WAV file using simpleaudio
    wave_obj = sa.WaveObject.from_wave_file(wav_file_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

# Main loop
def main():
    global pause_flag
    conversation_history = ""

    try:
        while True:
            if not pause_flag:
                print("Listening...")
                audio_file_path = capture_audio()
                if audio_file_path:
                    print("Transcribing audio with OpenAI Whisper API...")
                    text = transcribe_audio(audio_file_path)
                    if text is None or text.strip() == "":
                        print("Error in transcription or empty transcription")
                        continue

                    print(f"Heard: {text}")

                    if text.lower() in ["clear memory.", "erase memory.", "clear memory", "erase memory"]:
                        conversation_history = ""
                        text_to_speech("Memory cleared!!")
                        continue
                    if text.lower() in ["stop.", "stop"]:
                        text_to_speech("Stopped!!")
                        pause_flag = True
                        continue
                    if text.lower() in ["start.", "start"]:
                        text_to_speech("Resuming!!")
                        pause_flag = False
                        continue

                    conversation_history += f"User: {text}\n"
                    response = process_with_gpt(conversation_history)
                    conversation_history += f"Assistant: {response}\n"
                    text_to_speech(response)
            else:
                pytime.sleep(1)
    except KeyboardInterrupt:
        pass

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
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    main()
