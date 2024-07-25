import time as pytime
import os
import json
import sounddevice as sd
import numpy as np
import soundfile as sf
from openai import OpenAI
import queue
import simpleaudio as sa
from pydub import AudioSegment
import noisereduce as nr
from pynput import keyboard
from scipy.signal import medfilt

client = OpenAI(api_key="")

# Define the queue at the top level
q = queue.Queue()

# Global variables
pause_flag = False
noise_threshold = 0.0
amplification_factor = 1.0  # Adjust this value to amplify the input


# Function to measure ambient noise with band-pass filtering
def measure_ambient_noise(duration=10, samplerate=44100, kernel_size=3):
    print("Measuring ambient noise...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()  # Wait until recording is finished

    # Apply band-pass filtering
    low_cutoff = 300.0  # Lower cutoff frequency in Hz
    high_cutoff = 3400.0  # Higher cutoff frequency in Hz
    filtered_recording = bandpass_filter(recording.flatten(), samplerate, low_cutoff, high_cutoff)

    rms_values = np.sqrt(np.mean(filtered_recording ** 2, axis=0))

    # Ensure kernel_size does not exceed the extent of rms_values
    if np.isscalar(rms_values):
        rms_values = np.array([rms_values])
    kernel_size = min(kernel_size, len(rms_values) // 2 * 2 + 1)  # kernel_size must be odd and <= len(rms_values)

    # Apply median filter to smooth the rms values
    filtered_rms_values = medfilt(rms_values, kernel_size=kernel_size)

    volume_max = np.max(filtered_rms_values)

    print(f"Measured maximum ambient noise level: {volume_max}")
    text_to_speech("Listening!")
    return volume_max


def bandpass_filter(data, samplerate, low_cutoff, high_cutoff):
    from scipy.signal import butter, lfilter
    nyquist = 0.5 * samplerate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(1, [low, high], btype='band')
    return lfilter(b, a, data)


# Function to capture audio
def capture_audio():
    global pause_flag, noise_threshold

    # Query the default sample rate of the input device
    default_samplerate = int(sd.query_devices(kind='input')['default_samplerate'])
    print(f"Using default sample rate: {default_samplerate} Hz")

    # Initialize a list to store audio frames
    audio_frames = []
    is_speaking = False
    silence_start = None

    def callback(indata, frames, callback_time, status):
        nonlocal is_speaking, silence_start
        # Amplify the input signal
        amplified_data = indata * amplification_factor
        audio_frames.append(amplified_data)
        volume_norm = (amplified_data ** 2).mean() ** 0.5

        if len(audio_frames) % 15 == 0:
            print(f"Measured volume norm: {volume_norm}")

        if volume_norm > noise_threshold:
            is_speaking = True
            silence_start = None
        else:
            if is_speaking:
                if silence_start is None:
                    silence_start = pytime.time()
                elif pytime.time() - silence_start > 3.0:  # Changed from 1.0 to 3.0 seconds
                    is_speaking = False
                    audio_data = np.concatenate(audio_frames, axis=0)
                    q.put(audio_data)
                    audio_frames.clear()
                    silence_start = None

    with sd.InputStream(callback=callback, channels=1, samplerate=default_samplerate):
        while not pause_flag:
            try:
                audio_data = q.get(timeout=1)
                if audio_data is not None:
                    # Apply noise reduction with adjusted parameters
                    reduced_noise_audio = nr.reduce_noise(y=audio_data.flatten(), sr=default_samplerate,
                                                          stationary=True, prop_decrease=0.8)
                    audio_file_path = "recorded_audio.wav"
                    sf.write(audio_file_path, reduced_noise_audio, default_samplerate)
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
    global pause_flag, noise_threshold
    conversation_history = ""

    try:
        # Measure ambient noise level to set the noise threshold
        ambient_noise_level = measure_ambient_noise()
        noise_threshold = ambient_noise_level * 1.5  # Adjust the multiplier based on your environment

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
                    if text.lower() in ["remeasure noise.", "remeasure background noise."]:
                        text_to_speech("Measuring background noise.")
                        ambient_noise_level = measure_ambient_noise()
                        noise_threshold = ambient_noise_level * 1.5
                        text_to_speech("Background noise measurement complete.")

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
