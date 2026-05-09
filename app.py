from fastapi import FastAPI
from pydantic import BaseModel

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# MODEL
MODEL_NAME = "SanjuTuni/bengali_sentiment_distilbert"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded!")

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

print("Model loaded!")

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

model.eval()


# FASTAPI
app = FastAPI(
    title="Bengali Sentiment Analysis API"
)


# REQUEST FORMAT
class TextRequest(BaseModel):
    text: str


# PREDICTION FUNCTION
def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    if prediction == 1:
        return "Positive"

    return "Negative"


# HOME ROUTE
@app.get("/")

def home():

    return {
        "message":
        "Bengali Sentiment API Running!"
    }


# PREDICT ROUTE
@app.post("/predict")

def predict(request: TextRequest):

    sentiment = predict_sentiment(
        request.text
    )

    return {
        "text": request.text,
        "sentiment": sentiment
    }