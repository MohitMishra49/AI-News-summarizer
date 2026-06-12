import gradio as gr
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_PATH = "."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()


def summarize(text):
    if not text.strip():
        return "Please enter a news article."

    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste your news article here...",
        label="News Article"
    ),
    outputs=gr.Textbox(
        lines=5,
        label="Generated Summary"
    ),
    title="AI News Summarizer",
    description="Generate concise summaries from news articles using a fine-tuned T5 model."
)

if __name__ == "__main__":
    demo.launch()