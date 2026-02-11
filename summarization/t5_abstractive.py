from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model and tokenizer lazily
_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        # Reverting to t5-small but using direct model control
        model_name = "t5-small"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(_device)
    return _model, _tokenizer

def t5_summary(text: str, max_length: int = 150) -> str:
    """Generates an abstractive summary using the T5 model with direct model control."""
    model, tokenizer = get_model_and_tokenizer()
    
    # Prefix for T5
    input_text = "summarize: " + text
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    ).to(_device)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
