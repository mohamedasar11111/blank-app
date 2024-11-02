import torch
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import soundfile as sf
from PIL import Image
from IPython.display import display
import pytesseract
import streamlit as st

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")   

def show_image(uploaded_file):
    if uploaded_file is not None:
        # Read the uploaded file as bytes
        image_bytes = uploaded_file.read()
        # Open the image using PIL
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        return img
    else:
        return None

def ocr_image(src_img):
    if src_img is not None:
        return pytesseract.image_to_string(src_img)
    else:
        return ""
picture = st.file_uploader('Upload a photo')
print(picture)
picture_text = show_image(picture)  # Use snake_case for variable names
exported_text = ocr_image(picture_text)
print(exported_text)

synthesizer = pipeline("text-to-speech", "microsoft/speecht5_tts") 
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesizer(exported_text, forward_params={"speaker_embeddings": speaker_embedding})
sf.write("exported_text.wav", speech["audio"], samplerate=speech["sampling_rate"])
