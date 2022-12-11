import sys

from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError

import os
from datetime import datetime
import base64
import openai
import playsound
import pyttsx3
import requests
import speech_recognition as recognition
import soundfile as sf
from contextlib import closing
from tempfile import gettempdir

# CONFIG

openAIToken = os.getenv('openAIToken')
myname = "Michael"
NameandSpace = "Michael: "  # Change to the individuals name
session = Session(profile_name="default", region_name='us-west-2') #Need to set up your profile in ~\.aws\
polly = session.client("polly")
MAX_CONVERSATION_LENGTH = 200

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Female voice, change to 0 for male voice

volume = engine.getProperty('volume')
engine.setProperty('volume', 0.6)  # set to 60% volume
speech = recognition.Recognizer()

name = "Polly"
date = str(datetime.today()).split(" ")[0]

content = ""


def gpt3(stext):
    openai.api_key = openAIToken
    response = openai.Completion.create(
        engine="text-davinci-003",  # Latest model as of 12/2022
        prompt=stext,
        temperature=0.7,
        max_tokens=135,
        top_p=0.99,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[NameandSpace, f"{name}: "]
    )
    return response.choices[0].text


def read_file(filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data


def doPolly(response):
    try:
        # Request speech synthesis
        response = polly.synthesize_speech(Text=response, OutputFormat="mp3",
                                           VoiceId="Joanna", Engine="neural")
    except (BotoCoreError, ClientError) as error:
        # The service returned an error, exit gracefully
        print(error)
        sys.exit(-1)
    if "AudioStream" in response:
        # Note: Closing the stream is important because the service throttles on the
        # number of parallel connections. Here we are using contextlib.closing to
        # ensure the close method of the stream object will be called automatically
        # at the end of the with statement's scope.
        with closing(response["AudioStream"]) as stream:
            output = os.path.join("speech.mp3")

            try:
                # Open a file for writing the output as a binary stream
                with open(output, "wb") as file:
                    file.write(stream.read())
            except IOError as error:
                # Could not write to file, exit gracefully
                print(error)
                sys.exit(-1)

    else:
        # The response didn't contain audio data, exit gracefully
        print("Could not stream audio")
        sys.exit(-1)
    playsound.playsound(output)
    os.remove("speech.mp3")


def doassembly():
    headers = {'authorization': os.getenv('AssemblyToken')}
    json = {"audio_data": enc1, "punctuate": True}
    response1 = requests.post('https://api.assemblyai.com/v2/stream',
                              headers=headers,
                              json=json,
                              )
    print(response1.json())
    print(response1.json())
    print(response1.json())
    return (response1.json()['text'])


content += f"Your name is {name}\n{name}: Heyo.\n"
pauseFlag = False
try:
    while (True):
        while (True):
            with recognition.Microphone() as source:
                speech.energy_threshold = 4000
                print("Listening..")
                audio = speech.listen(source)  # Starts Listening for words
                try:
                    text3 = speech.recognize_google(audio)#Only used during pre-checks
                except Exception as e:
                    print(e)
                    continue

                with open("microphone-results.flac", "wb") as f:
                    f.write(audio.get_flac_data())
                # with open("microphone-results.wav", "wb") as q:
                #     q.write(audio.get_wav_data())
                os.system(
                    "ffmpeg -y -i microphone-results.flac -ar 8000 -rematrix_maxval 1.0 -minrate 128k -maxrate 128k -bufsize 128k -ac 1 -b:a 128k new.flac")
                data, samplerate = sf.read('new.flac')
                sf.write('new1.RAW', data, 8000, subtype='PCM_16', format='RAW')
                enc = base64.b64encode(open("new1.RAW", "rb").read())

                enc1 = enc.decode('UTF-8')
                # print(enc) #DEBUG
                # print(base64.b64decode(enc)) #DEBUG


            try:

                if text3 == "":
                    print("CAUGHT A BAD LISTEN")
                    continue
                text = doassembly()

                text = text.strip()
                # preGenerationCheckouts()
                print(text)

                if text == "Clear memory.":
                    content = f"Your name is {name}\n"
                    doPolly("Memory Cleared")
                    engine.runAndWait()
                    continue
                if text == "Stop." or text == "Stop!":
                    doPolly(name+" is Stopped")
                    pauseFlag = True
                    continue
                if text == "Make ready." or text == "start" or text == "START." or text == "start":
                    doPolly(name+" is Resumed")

                    pauseFlag = False
                    continue
                if text == "" or len(text) < 4:
                    print("CAUGHT A BAD LISTEN")
                    continue
                if pauseFlag == True:
                    engine.runAndWait()
                    print("PAUSED")
                    continue

                break
            except Exception as e:
                print(e)
                print("I did not catch that, let's try again.")
                pass
        if len(content) > MAX_CONVERSATION_LENGTH:

            counter = 0
            x = content.split(NameandSpace)
            content = ""
            for y in x:
                if counter > 0:
                    # if counter == 1:
                    content += NameandSpace
                    content += y
                counter += 1

        print(content)
        content += f"{NameandSpace}{text}\n"+name+": "
        response = gpt3(content).lstrip()
        content += f"{response}\n"
        doPolly(response)

        # engine.say()
        # engine.runAndWait()

        print(content)
except KeyboardInterrupt:
    pass
