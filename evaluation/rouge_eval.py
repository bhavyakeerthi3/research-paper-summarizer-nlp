from rouge_score import rouge_scorer

def compute_rouge(reference: str, predicted: str) -> dict:
    """Computes ROUGE scores for a predicted summary against a reference."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        use_stemmer=True
    )
    scores = scorer.score(reference, predicted)
    
    # Format scores for easier reading
    return {
        "rouge1": {
            "precision": scores["rouge1"].precision,
            "recall": scores["rouge1"].recall,
            "fmeasure": scores["rouge1"].fmeasure
        },
        "rougeL": {
            "precision": scores["rougeL"].precision,
            "recall": scores["rougeL"].recall,
            "fmeasure": scores["rougeL"].fmeasure
        }
    }
