import requests
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from tqdm import tqdm
import urllib.parse as parse
import os

# LitServe API Integration
import litserve as ls


# Verify URL function
def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


# Load an image from a URL or local path
def load_image(image_path):
    if check_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)


# HuggingFace API class for image captioning
class ImageCaptioningLitAPI(ls.LitAPI):
    def setup(self, device):
        # Assign available GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the ViT Encoder-Decoder Model
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)

        # Load the corresponding Tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

        # Load the Image Processor
        self.image_processor = ViTImageProcessor.from_pretrained(model_name)

    # Decode request payload to extract image URL or path
    def decode_request(self, request):
        return request["image_path"]

    # Generate image caption
    def predict(self, image_path):
        image = load_image(image_path)

        # Preprocessing the Image
        img = self.image_processor(image, return_tensors="pt").to(self.device)

        # Generating captions
        output = self.model.generate(**img)

        # Decode the output to generate the caption
        caption = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return caption

    # Encode the response back to the client
    def encode_response(self, output):
        return {"caption": output}


# Running the LitServer
if __name__ == "__main__":
    api = ImageCaptioningLitAPI()
    server = ls.LitServer(api, accelerator="auto", devices=1, workers_per_device=1)
    server.run(port=8000)
