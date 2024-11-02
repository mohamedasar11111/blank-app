import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return model

model = load_model()

st.title("TrOCR for Image-to-Text")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
     try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        st.subheader("Extracted Text:")
        st.write(extracted_text)
    except Exception as e:
        st.error(f"An error occurred: {e}")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


    st.subheader("Extracted Text:")
    st.write(extracted_text)
