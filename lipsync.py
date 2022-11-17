import time
import json

from gtts import gTTS
import requests

# from utils import RequestInput
# from flask import Flask, Blueprint, current_app, jsonify, request
from flask import Flask, jsonify, request

# lypsync_api = Blueprint('lypsync_api', __name__)

import os
import whisper
import moviepy.editor
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import time
import pandas as pd
import base64

model_audio = whisper.load_model("large")
model_translate = MBartForConditionalGeneration.from_pretrained(
"facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained(
"facebook/mbart-large-50-many-to-many-mmt")
# model_audio, model_translate, tokenizer = 1, 2, 3





def text_to_speech(indic_translated, filename):
    parameters = {
        "input": [
            {
            "source": indic_translated
            }
        ],
        "config": {
            "gender": "male",
            "language": {
            "sourceLanguage": 'ta'
            }
        }
    }
    response = requests.post("https://tts-api.ai4bharat.org", json=parameters)
    audio_str = response.json()['audio'][0]['audioContent']
    wav_file = open(filename, "wb")
    decode_string = base64.b64decode(audio_str)
    wav_file.write(decode_string)


def speech(my_text,language,filename):
	tts = gTTS(text=my_text, lang=language)
	tts.save(filename) 

def extract_audio(source_video, dest_filename):
    mp4_filename = source_video.split(".")[0] + ".mp4"
    print(mp4_filename)
    os.system(f'ffmpeg -fflags +genpts -i {source_video} -r 24 -y {mp4_filename}')
    print("Conversion process done")
    video = moviepy.editor.VideoFileClip(mp4_filename)
    audio = video.audio
    audio.write_audiofile(dest_filename)
    print("Write to file complete")


def audio_to_text(dest_filename):
    print(dest_filename)
    audio_text = model_audio.transcribe(dest_filename)
    print(audio_text)
    return audio_text


def translate_to_dest_lang(source_text,
                           source_lang="hi_IN",
                           target_lang="en_XX"):
    tokenizer.src_lang = source_lang
    encoded_hi = tokenizer(source_text, return_tensors="pt")
    generated_tokens = model_translate.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
    translated_text = tokenizer.batch_decode(generated_tokens,
                                             skip_special_tokens=True)
    print(translated_text)
    parameters_trans = {
        "data": [
            translated_text[0],
            "Tamil"
        ]
        }
    response_trans = requests.post("https://hf.space/embed/ai4bharat/IndicTrans-English2Indic/+/api/predict/", json=parameters_trans)
    print(response_trans)
    indic_translated = response_trans.json()['data'][0]
    print(indic_translated)
    return indic_translated


def deep_fake(source_video, audio_file):
    cmd = f"python inference.py --checkpoint_path /home/chiragagrawal/Wav2Lip/checkpoints/wav2lip_gan.pth --face \"{source_video}\" --audio \"{audio_file}\""
    print(cmd)
    os.system(cmd)


# creating a Flask app
app = Flask(__name__)

# import requests
# a = requests.post("http://localhost:9010/generate_lypsync",data={"gcs_link":"/home/adjjkef/abc.mp4"})


@app.route("/extract_audio", methods=['POST'])
def extract_audio_ep():
    video_file_inp = request.form.get("video_file")
    print("inp: ", video_file_inp)
    # video_file_upload_location = os.path.join("tempDir",video_file_inp.name)
    # video_file = open(f'{video_file_inp.name}', 'rb')
    # video_bytes = video_file.read()
    video_filename = video_file_inp.split(".")[0]
    print("trunc:", video_filename)
    # print(video_filename)
    extract_audio(video_file_inp, video_filename + ".mp3")
    return "success"

@app.route("/audio_to_text", methods=['POST'])
def audio_to_text_ep():
    video_file_inp = request.form.get("video_file")
    video_filename = video_file_inp.split(".")[0]
    obtained_text = audio_to_text(video_filename + ".mp3")
    #open text file
    print("Obtained text: ", obtained_text)
    text_file = open("/home/chiragagrawal/Wav2Lip/tempDir/data.txt", "w")
    text_file.write(obtained_text["text"])
    text_file.close()
    return "success"

@app.route("/translate_text", methods=['POST'])
def translate_text_ep():
    with open('/home/chiragagrawal/Wav2Lip/tempDir/data.txt', 'r') as f:
        data = f.read().replace('\n', '')
    print(data)
    translated_text = translate_to_dest_lang(data)
    text_file = open("/home/chiragagrawal/Wav2Lip/tempDir/translated_data.txt", "w")
    # if len(translated_text) > 0:
    text_file.write(translated_text)
    return "success"

@app.route("/prepare_audio", methods=['POST'])
def prepate_audio_ep():
    with open('/home/chiragagrawal/Wav2Lip/tempDir/translated_data.txt', 'r') as f:
        data = f.read().replace('\n', '')

    text_to_speech(data, "/home/chiragagrawal/Wav2Lip/tempDir/translated_audio.wav")
    return "success"

@app.route("/create_video", methods=['POST'])
def create_video_ep():
    video_file_inp = request.form.get("video_file")
    audio_file = "/home/chiragagrawal/Wav2Lip/tempDir/translated_audio.wav"
    deep_fake(video_file_inp, audio_file)
    return "success"

if __name__ == '__main__':
    app.run('0.0.0.0', 9010, debug=True)
