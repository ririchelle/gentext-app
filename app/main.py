import io

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.cnn_model import CNNClassifier
from app.gan_model import Generator

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
cnn_classifier = CNNClassifier(num_classes=10)
cnn_classifier.load_state_dict(torch.load("app/cnn_weights.pth", map_location="cpu"))
cnn_classifier.eval()

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

GAN_Z_DIM = 100
gan_generator = Generator(z_dim=GAN_Z_DIM)
gan_generator.load_state_dict(torch.load("app/gan_generator_weights.pth", map_location="cpu"))
gan_generator.eval()

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.get_embedding(request.word)
    return {"word": request.word, "embedding": embedding}

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = cnn_classifier(tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = CIFAR10_CLASSES[predicted.item()]
    return {"predicted_class": predicted_class}

@app.get("/generate-digit")
def generate_digit():
    with torch.no_grad():
        noise = torch.randn(1, GAN_Z_DIM)
        fake_image = gan_generator(noise)  # shape (1, 1, 28, 28), values in [-1, 1]

    # Undo the Tanh normalization: [-1, 1] -> [0, 1] -> [0, 255]
    image_tensor = (fake_image.squeeze(0).squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    image_array = (image_tensor.numpy() * 255).astype("uint8")
    image = Image.fromarray(image_array, mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")