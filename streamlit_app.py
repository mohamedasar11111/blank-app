import torch
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import soundfile as sf
from PIL import Image
from IPython.display import display
import pytesseract

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

def show_image(path_str):  # Use snake_case for function names
    img = Image.open(path_str).convert("RGB")
    display(img)
    return img

def ocr_image(src_img):
    return pytesseract.image_to_string(src_img)

picture_text = show_image('./handwriting.png')  # Use snake_case for variable names
exported_text = ocr_image(picture_text)
print(exported_text)

synthesizer = pipeline("text-to-speech", "microsoft/speecht5_tts") 
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesizer(exported_text, forward_params={"speaker_embeddings": speaker_embedding})
sf.write("exported_text.wav", speech["audio"], samplerate=speech["sampling_rate"])
