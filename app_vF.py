import streamlit as st
import streamlit.components.v1 as stc
import os
import whisper
import moviepy.editor
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import time
# File Processing Pkgs
import pandas as pd
import requests


def extract_audio(source_video, dest_filename):
    mp4_filename = source_video.split(".")[0] + ".mp4"
    print(mp4_filename)
    os.system(f'ffmpeg -fflags +genpts -i {source_video} -r 24 -y {mp4_filename}')
    print("Conversion process done")
    video = moviepy.editor.VideoFileClip(mp4_filename)
    audio = video.audio
    audio.write_audiofile(dest_filename)

def audio_to_text(dest_filename, model_audio):
    
    audio_text = model_audio.transcribe(dest_filename)
    print(audio_text)
    return audio_text

def translate_to_dest_lang(source_text, model_translate, tokenizer, source_lang="hi_IN", target_lang="en_XX"):
    tokenizer.src_lang = source_lang
    encoded_hi = tokenizer(source_text, return_tensors="pt")
    generated_tokens = model_translate.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(translated_text)

def deep_fake(source_video, audio_file):
    cmd = f"python /home/chiragagrawal/Wav2Lip/inference.py --checkpoint_path /home/chiragagrawal/Wav2Lip/checkpoints/wav2lip_gan.pth --face \"{source_video}\" --audio \"{audio_file}\""
    print(cmd)
    os.system(cmd)
    

def read_pdf(file):
	pdfReader = PdfFileReader(file)
	count = pdfReader.numPages
	all_page_text = ""
	for i in range(count):
		page = pdfReader.getPage(i)
		all_page_text += page.extractText()

	return all_page_text

def read_pdf_with_pdfplumber(file):
	with pdfplumber.open(file) as pdf:
	    page = pdf.pages[0]
	    return page.extract_text()


def main():

    st.title("Deep Fake creative Generation!!")
    video_file = st.file_uploader("Upload Video",type=['webm'])
    if st.button("Process"):
        if video_file is not None:
            file_details = {"Filename":video_file.name,"FileType":video_file.type,"FileSize":video_file.size}
            st.write(file_details)
            video_file_upload_location = os.path.join("/home/chiragagrawal/Wav2Lip/tempDir",video_file.name)
            if not os.path.isdir("tempDir"):
                os.makedirs("tempDir")
            with open(video_file_upload_location,"wb") as f: 
                f.write(video_file.getbuffer())         
                st.success("Saved File")

            video_file = open(f'tempDir/{video_file.name}', 'rb')
            video_bytes = video_file.read()

            col1, col2 = st.columns(2, gap="large")

            with col1:	
                st.header("Original")	
                st.video(video_bytes)

            with col2:
                st.header("Converted!")

                with st.spinner('Churning out source language of video....'):
                    time.sleep(5)
                    field1 = st.success('Language detected!')

                video_filename = video_file.name.split(".")[0]
                with st.spinner('Extracting audio...'):
                    print("V File:", video_file_upload_location)
                    start = time.time()
                    a = requests.post("http://localhost:9010/extract_audio",data={"video_file":video_file_upload_location})
                    print(a)
                    end = time.time()
                    print("Extract Audio time: ", end-start)
                    
                    
                    field2 = st.success('Audio Extracted!')
            
                with st.spinner('Converting audio to text...'):
                    # 2. Convert audio to text 
                    start = time.time()
                    a = requests.post("http://localhost:9010/audio_to_text",data={"video_file":video_file_upload_location})
                    end = time.time()
                    print("Audio_to_text time: ", end-start)
                    field3 = st.success('Text is ready!')

                with st.spinner('Translating text...'):
                    # 3. Translate text from source lang to target
                    start = time.time()
                    a = requests.post("http://localhost:9010/translate_text",data={})
                    end = time.time()
                    print("Translation time time: ", end-start)
                    field4 = st.success('Translation done!')

                with st.spinner('Re-creating audio in desired language.'):
                    a = requests.post("http://localhost:9010/prepare_audio",data={})
                    field5 = st.success('Audio prepared!')

                with st.spinner('Creating you video...Hold on buddy almost there!!'):
                    a = requests.post("http://localhost:9010/create_video",data={"video_file": video_file_upload_location})
                    field6 = st.success('Done!')

                output_file = "/home/chiragagrawal/Wav2Lip/results/result_voice.mp4"
                video_file = open(f'{output_file}', 'rb')
                video_bytes = video_file.read()

                
                    
                

                field1.empty()
                field2.empty()
                field3.empty()
                field4.empty()
                field5.empty()
                field6.empty()
                st.video(video_bytes)
                # st.header("Converted!")
                

            # col1.header("Original")
            # col1.video(video_bytes)

            # grayscale = original.convert('LA')
            # col2.header("After Magic!")

            # model_audio = whisper.load_model("large")
            # model_translate = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            # tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

            # Sequential calls to various process
            
        
            # 2. Convert audio to text 
            # start = time.time()
            # obtained_text = audio_to_text(video_filename+".mp3", model_audio)
            # end = time.time()
            # print("Audio_to_text time: ", end-start)
            # print(obtained_text)

            # 3. Translate text from source lang to target
            # start = time.time()
            # translate_to_dest_lang(obtained_text["text"], model_translate, tokenizer)
            # end = time.time()
            # print("Translation time time: ", end-start)

            # Obtain audio from text

            # Deep Fake
            

            


            # col2.image(grayscale, use_column_width=True)
    
    
        
            st.subheader("Team Learners")
            st.info("Built with Streamlit")
        else:
            st.warning("Please upload video")



if __name__ == '__main__':
   
    main()