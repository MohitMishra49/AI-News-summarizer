from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware
MODEL_PATH = "my_t5_summarizer"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

app = FastAPI(
    title="News Summarizer API",
    description="T5 Fine-Tuned News Summarization API",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SummaryRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "message": "News Summarizer API Running"
    }

@app.post("/summarize")
def summarize(request: SummaryRequest):

    article = request.text

    inputs = tokenizer(
        "summarize: " + article,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    summary_ids = model.generate(
        **inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return {
        "summary": summary
    }
