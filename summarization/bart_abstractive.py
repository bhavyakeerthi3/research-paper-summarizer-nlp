from transformers import pipeline

bart_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn"
)

def bart_summary(text: str, max_length: int = 150) -> str:
    result = bart_summarizer(
        text,
        max_length=max_length,
        min_length=50,
        do_sample=False
    )
    return result[0]["summary_text"]
