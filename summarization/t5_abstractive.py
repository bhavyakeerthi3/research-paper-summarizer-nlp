from transformers import pipeline

t5_summarizer = pipeline(
    "summarization",
    model="t5-small",
    tokenizer="t5-small"
)

def t5_summary(text: str, max_length: int = 150) -> str:
    result = t5_summarizer(
        text,
        max_length=max_length,
        min_length=50,
        do_sample=False
    )
    return result[0]["summary_text"]
