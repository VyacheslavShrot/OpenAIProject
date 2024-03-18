import ssl

import openai
from openai import OpenAI
from pytube import YouTube

from utils.get_env import OPENAI_API_KEY

ssl._create_default_https_context = ssl._create_stdlib_context

client = OpenAI(api_key=OPENAI_API_KEY)


def youtube_audio_downloader(link: str) -> str:
    yt = YouTube(link)

    audio = yt.streams.filter(only_audio=True).first()

    audio_file_path = audio.download()
    return audio_file_path


def transcribe(audio_file_path: str, translate=False) -> str:
    if translate:
        with open(audio_file_path, 'rb') as file:
            translations = openai.audio.translations.create(
                model='whisper-1',
                file=file
            )
            return translations.text

    with open(audio_file_path, 'rb') as file:
        transcription = openai.audio.transcriptions.create(
            model='whisper-1',
            file=file
        )
        return transcription.text


def summary(text: str) -> str:
    system_prompt = f"""
    You'll get text from some youtube video and you'll have to write briefly what it's about and tell it in a nutshell.
    Follow through on the following commitments:
    1. Answer as well as you can.
    2. If you have something to add from yourself, only using verified sites like WIKI and others.
    """
    messages = [
        {
            "role": "system", "content": system_prompt
        },
        {
            "role": "user", "content": text
        }
    ]

    summary_text = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=1
    )
    return summary_text.choices[0].message.content


_link = "https://www.youtube.com/watch?v=x38DO4fc2VE&t=10s"

downloaded_audio = youtube_audio_downloader(_link)
text_transcribed = transcribe(downloaded_audio, translate=True)
summary_response = summary(text_transcribed)
print(summary_response)
