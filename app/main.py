import io

import torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.cnn_model import CNNClassifier

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()

# Load the trained CNN classifier
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
cnn_classifier = CNNClassifier(num_classes=10)
cnn_classifier.load_state_dict(torch.load("app/cnn_weights.pth", map_location="cpu"))
cnn_classifier.eval()  # IMPORTANT: inference mode, not training mode

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


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
    tensor = image_transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = cnn_classifier(tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = CIFAR10_CLASSES[predicted.item()]
    return {"predicted_class": predicted_class}